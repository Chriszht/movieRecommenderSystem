@echo off
REM Start Flask with the enhanced hybrid recommender and the new UI.
REM Run from an Anaconda Prompt where `conda activate lab3` has been executed.
set REC_ALGO=enhanced
set UI_MODE=enhanced
flask --app flaskr run --debug
