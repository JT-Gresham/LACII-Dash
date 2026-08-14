# Shared secret patterns for the pre-commit and pre-push hooks. Sourced, never executed directly.
# Sets $SECRET_PAT (an extended regex). Keep this the ONE definition — two hooks with two
# hand-maintained pattern lists drift, and the one that drifts is the one that lets a secret past.
#
# NOTHING SECRET GOES IN THIS FILE. It is tracked and pushed to a PUBLIC mirror, so writing a
# literal here to detect it would publish exactly what the hooks exist to prevent. Patterns match
# by SHAPE. Site-specific literals belong in an untracked .git/secret-patterns (one ERE per line).

# 1. Forge tokens. The glpat- rule also mechanically enforces "never push the android branch
#    (old tokened history) to GitHub as-is".
SECRET_PAT='(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{20,})'

# 2. Cleartext passwords passed to sshpass inline. A shell variable ($PW, ${PW}) is fine and is
#    NOT matched — only a bare literal is. This comment deliberately does not spell the invocation
#    out verbatim: an earlier draft did, the hook matched ITSELF, and it blocked every push.
SECRET_PAT="$SECRET_PAT|sshpass[[:space:]]+-p[[:space:]]*[\"']?[^\$\"'[:space:]]{4,}"

# 3. Site literals, kept OUT of the repo. Optional; absent is fine.
_sp="$(git rev-parse --git-dir 2>/dev/null)/secret-patterns"
if [ -r "$_sp" ]; then
  while IFS= read -r _line; do
    case "$_line" in ''|'#'*) continue ;; esac
    SECRET_PAT="$SECRET_PAT|$_line"
  done < "$_sp"
fi
unset _sp _line
