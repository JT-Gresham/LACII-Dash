"""Offline unit test for #gguf-split — multi-part GGUF ingestion (no torch, no network, no fleet).

Covers the parts of the split-GGUF pipeline that can be wrong SILENTLY: which files make up a set,
whether an incomplete set is refused, and that a single-file GGUF still resolves to exactly the
call the old code made. The dequantization itself (transformers' GGUF reader) needs torch and is
NOT exercised here — its guard is gguf_convert._verify_fully_loaded, tested below on synthetic
loading_info because the real thing needs a GPU box with transformers.

Run: python3 test_gguf_split.py
"""
import sys
import types

import gguf_convert
import model_store


# --------------------------------------------------------------------------- part-name derivation
def test_split_info_and_part_names():
    assert model_store.gguf_split_info("Model-Q4_K_M.gguf") is None
    assert model_store.gguf_split_info("") is None
    assert model_store.gguf_split_info("Model-00001-of-00003.gguf") == ("Model", 1, 3)
    assert model_store.gguf_split_info("Q4_K_M/m-00002-of-00002.gguf") == ("Q4_K_M/m", 2, 2)
    # out-of-range series are NOT interpreted (see the helper: guessing a part set is how you
    # silently fetch the wrong tensors)
    assert model_store.gguf_split_info("Model-00000-of-00003.gguf") is None
    assert model_store.gguf_split_info("Model-00004-of-00003.gguf") is None
    # 4-digit / 6-digit widths are not llama.cpp's convention -> not a split name
    assert model_store.gguf_split_info("Model-0001-of-0003.gguf") is None

    assert model_store.gguf_part_names("Model-Q4_K_M.gguf") == ["Model-Q4_K_M.gguf"]
    assert model_store.gguf_part_names("M-00002-of-00003.gguf") == [
        "M-00002-of-00003.gguf".replace("00002", "00001"),
        "M-00002-of-00003.gguf",
        "M-00002-of-00003.gguf".replace("00002", "00003")]
    # any part yields the SAME set, in order, starting at 1
    for i in (1, 2, 3):
        assert model_store.gguf_part_names(f"x/B-{i:05d}-of-00003.gguf") == [
            "x/B-00001-of-00003.gguf", "x/B-00002-of-00003.gguf", "x/B-00003-of-00003.gguf"]
    # extension CASE is preserved: a synthesized ".gguf" would 404 on a ".GGUF" repo
    assert model_store.gguf_part_names("B-00001-of-00002.GGUF") == [
        "B-00001-of-00002.GGUF", "B-00002-of-00002.GGUF"]
    print("OK gguf_split_info / gguf_part_names")


def test_converter_derivation_matches_controller():
    """gguf_convert keeps its OWN copy (isolated subprocess — see its module comment). The two must
    agree on every name, or the controller would validate one set and the converter fetch another."""
    names = ["Model-Q4_K_M.gguf", "M-00001-of-00003.gguf", "M-00003-of-00003.gguf",
             "sub/dir/M-00002-of-00011.gguf", "M-00000-of-00003.gguf", "M-00004-of-00003.gguf",
             "B-00001-of-00002.GGUF", "plain.gguf", "no-extension"]
    for n in names:
        parts, idx = gguf_convert._split_part_names(n)
        assert parts == model_store.gguf_part_names(n), (n, parts)
        info = model_store.gguf_split_info(n)
        assert idx == (info[1] if info else 1), (n, idx)
    print("OK converter/controller part derivation agree")


# --------------------------------------------------------------------------------- quant selection
def test_pick_prefers_and_accepts_split():
    import downloads
    pick, err = downloads._pick_gguf_quant(["M-Q8_0.gguf", "M-Q4_K_M.gguf", "M-F16.gguf"])
    assert (pick, err) == ("M-Q4_K_M.gguf", None), (pick, err)

    # a COMPLETE split set is a first-class candidate now, registered as its part 1
    full = [f"M-Q4_K_M-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    pick, err = downloads._pick_gguf_quant(full)
    assert (pick, err) == ("M-Q4_K_M-00001-of-00003.gguf", None), (pick, err)

    # better quant only available as a split set -> the split set wins on quant rank
    pick, err = downloads._pick_gguf_quant(full + ["M-Q2_K.gguf"])
    assert pick == "M-Q4_K_M-00001-of-00003.gguf", pick

    # same quant rank as a single file -> the single file wins (fewer moving parts)
    pick, err = downloads._pick_gguf_quant(full + ["M-Q4_K_M.gguf"])
    assert pick == "M-Q4_K_M.gguf", pick

    # only float dumps -> still translatable (unchanged behaviour)
    pick, err = downloads._pick_gguf_quant(["M-f16.gguf"])
    assert (pick, err) == ("M-f16.gguf", None), (pick, err)
    print("OK _pick_gguf_quant single/split ranking")


def test_pick_refuses_incomplete_split():
    """The load-bearing refusal: part 2 of 3 absent must FAIL, not quietly convert 2/3 of a model."""
    import downloads
    gapped = ["M-Q4_K_M-00001-of-00003.gguf", "M-Q4_K_M-00003-of-00003.gguf"]
    pick, err = downloads._pick_gguf_quant(gapped)
    assert pick is None, pick
    assert err and "INCOMPLETE" in err and "00002" in err, err

    # a usable candidate alongside a broken set -> pick the usable one, don't refuse the repo
    pick, err = downloads._pick_gguf_quant(gapped + ["M-Q5_K_M.gguf"])
    assert (pick, err) == ("M-Q5_K_M.gguf", None), (pick, err)

    # two DIFFERENT sets, one complete one not -> the complete one is picked
    ok = [f"M-Q6_K-{i:05d}-of-00002.gguf" for i in (1, 2)]
    pick, err = downloads._pick_gguf_quant(gapped + ok)
    assert pick == "M-Q6_K-00001-of-00002.gguf", pick

    # no .gguf at all -> (None, None): not an error, just not a GGUF repo
    assert downloads._pick_gguf_quant(["config.json"]) == (None, None)
    print("OK _pick_gguf_quant refuses an incomplete set")


# ------------------------------------------------------------------- resolver (stubbed hub listing)
def _stub_hub(listing_by_repo):
    """Install a fake huggingface_hub whose HfApi.list_repo_files serves `listing_by_repo`.
    KeyError -> raise, so the resolver's 'listing failed' branch can be exercised too."""
    mod = types.ModuleType("huggingface_hub")

    class HfApi:
        def __init__(self, *a, **k):
            pass

        def list_repo_files(self, repo, token=None):
            return list(listing_by_repo[repo])          # KeyError == repo not found

    mod.HfApi = HfApi
    sys.modules["huggingface_hub"] = mod


def test_resolver_accepts_and_refuses_split_repos():
    import downloads
    downloads.HF_TOKEN = None                            # normally injected by state.bind()
    downloads._friendly_from_hf = lambda hf: hf.split("/")[-1].lower()

    full = [f"M-Q4_K_M-{i:05d}-of-00004.gguf" for i in (1, 2, 3, 4)]
    _stub_hub({"org/M-GGUF": full + ["README.md"]})       # no safetensors twin -> KeyError on org/M
    r = downloads._resolve_download_source("org/M-GGUF")
    assert r["error"] is None, r
    assert r["gguf_file"] == "M-Q4_K_M-00001-of-00004.gguf", r
    assert "split set" in r["note"], r

    _stub_hub({"org/M-GGUF": full[:2] + full[3:]})        # part 3 missing
    r = downloads._resolve_download_source("org/M-GGUF")
    assert r["error"] and "INCOMPLETE" in r["error"], r
    assert r["gguf_file"] == "", r

    # explicit override naming a MIDDLE part: normalized to part 1, set verified complete
    _stub_hub({"org/M-GGUF": full})
    r = downloads._resolve_download_source("org/M-GGUF", "M-Q4_K_M-00003-of-00004.gguf")
    assert r["error"] is None and r["gguf_file"] == "M-Q4_K_M-00001-of-00004.gguf", r

    # explicit override on an incomplete set: refused (the old code honoured it blindly)
    _stub_hub({"org/M-GGUF": full[:3]})
    r = downloads._resolve_download_source("org/M-GGUF", "M-Q4_K_M-00001-of-00004.gguf")
    assert r["error"] and "INCOMPLETE" in r["error"], r

    # explicit override, listing unavailable: honoured, and SAYS it is unverified
    _stub_hub({})
    r = downloads._resolve_download_source("org/M-GGUF", "M-Q4_K_M-00002-of-00004.gguf")
    assert r["error"] is None and r["gguf_file"] == "M-Q4_K_M-00001-of-00004.gguf", r
    assert "NOT verified" in r["note"], r

    # explicit single-file override: byte-for-byte the old behaviour (no probe, verbatim)
    r = downloads._resolve_download_source("org/M-GGUF", "M-Q4_K_M.gguf")
    assert r == {"target": "org/M-GGUF", "gguf_file": "M-Q4_K_M.gguf", "friendly_hint": "",
                 "note": "explicit GGUF file M-Q4_K_M.gguf", "error": None}, r

    # a repo WITH safetensors is still passed through untouched
    _stub_hub({"org/Plain": ["model.safetensors", "config.json"]})
    r = downloads._resolve_download_source("org/Plain")
    assert r == {"target": "org/Plain", "gguf_file": "", "friendly_hint": "", "note": "",
                 "error": None}, r
    print("OK _resolve_download_source split accept/refuse/passthrough")


# ------------------------------------------------------------------------ the load-coverage guard
def test_verify_fully_loaded():
    cfg_tied = types.SimpleNamespace(tie_word_embeddings=True)
    cfg_untied = types.SimpleNamespace(tie_word_embeddings=False)
    f = gguf_convert._verify_fully_loaded

    assert f({"missing_keys": []}, cfg_tied, 3) == []
    assert f(None, cfg_tied, 1) == []
    # benign families must not trip the gate
    assert f({"missing_keys": ["model.layers.0.self_attn.rotary_emb.inv_freq",
                               "model.rotary_emb.inv_freq", "lm_head.weight"]}, cfg_tied, 3) == []
    # an UNtied lm_head genuinely missing is not benign
    assert f({"missing_keys": ["lm_head.weight"]}, cfg_untied, 3) == ["lm_head.weight"]
    # the failure this whole feature has to catch: tail layers never read from parts 2..N
    tail = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(20, 32)]
    assert f({"missing_keys": tail + ["model.layers.0.self_attn.rotary_emb.inv_freq"]},
             cfg_tied, 4) == tail
    print("OK _verify_fully_loaded benign/fatal classification")


def test_single_file_path_is_unchanged():
    """A non-split name must produce exactly the old one-file plan: one part, index 1, and the same
    src_dir/basename split the previous code did."""
    parts, idx = gguf_convert._split_part_names("Model-Q4_K_M.gguf")
    assert (parts, idx) == (["Model-Q4_K_M.gguf"], 1)
    assert model_store.gguf_part_names("Model-Q4_K_M.gguf") == ["Model-Q4_K_M.gguf"]
    # convert_gguf_to_model_dir's log/derivation treats it as 1 part -> no split branch anywhere
    assert len(model_store.gguf_part_names("Model-Q4_K_M.gguf")) == 1
    print("OK single-file path unchanged")


if __name__ == "__main__":
    test_split_info_and_part_names()
    test_converter_derivation_matches_controller()
    test_pick_prefers_and_accepts_split()
    test_pick_refuses_incomplete_split()
    test_resolver_accepts_and_refuses_split_repos()
    test_verify_fully_loaded()
    test_single_file_path_is_unchanged()
    print("\nALL GGUF-SPLIT TESTS PASSED")
