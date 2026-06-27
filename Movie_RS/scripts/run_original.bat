@echo off
REM Start Flask with the original baseline (kNN + multi-hot content) and the
REM original template. Used as condition A in the A/B test.
set REC_ALGO=original
set UI_MODE=original
flask --app flaskr run --debug
