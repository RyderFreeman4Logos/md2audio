check:
    TMPDIR="$HOME/tmp" TMP="$HOME/tmp" TEMP="$HOME/tmp" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
    PYTHONPYCACHEPREFIX="$HOME/tmp/md2audio-bytecode" .venv/bin/python -m py_compile md2audio.py md2audio_dialogue.py tests/test_multi_speaker.py tests/test_synthesis.py
    bash -n scripts/hooks/review-check.sh
