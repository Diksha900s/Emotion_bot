# Emotion Detection Chatbot

This project already uses the bundled dataset in `data/emotions.csv` with the required `text,emotion` columns.

Run it from the project folder or the repo root:

    pip install -r requirements.txt
    python train.py     # trains + saves model.joblib
    python chat.py      # type sentences interactively

The scripts resolve file paths relative to the project folder, so they work even when launched outside the current working directory.
