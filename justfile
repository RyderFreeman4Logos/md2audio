check:
    TMPDIR="$HOME/tmp" TMP="$HOME/tmp" TEMP="$HOME/tmp" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
    PYTHONPYCACHEPREFIX="$HOME/tmp/md2audio-bytecode" .venv/bin/python -m py_compile *.py tests/*.py
    for hook in scripts/hooks/*.sh scripts/hooks/pre-push; do bash -n "$hook"; done

install-hooks:
    git config --local core.hooksPath scripts/hooks
