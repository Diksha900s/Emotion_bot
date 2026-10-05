"""Interactive chatbot: type a sentence -> detected emotion + reply. Type 'quit' to exit."""
import random
from pathlib import Path

import joblib

from preprocess import clean

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.joblib"

REPLIES = {
    "sad": ["I'm really sorry you're going through this. I'm here to listen."],
    "happy": ["That's wonderful! I'm so happy for you!"],
    "angry": ["That sounds frustrating. Do you want to vent more?"],
    "fearful": ["That's a lot to carry. Take it one step at a time."],
    "surprised": ["Wow, that's quite a twist! How are you feeling about it?"],
    "disgusted": ["Ugh, that sounds unpleasant. I'm sorry you had to see that."],
    "neutral": ["Got it. Anything else you'd like to tell me?"],
}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run 'python train.py' first.")

model = joblib.load(MODEL_PATH)
print("Emotion chatbot ready. Type a sentence (or 'quit').")

while True:
    try:
        text = input("\nYou: ").strip()
    except EOFError:
        break
    if text.lower() in {"quit", "exit"}:
        break
    if not text:
        continue

    emo = model.predict([clean(text)])[0]
    print(f"[detected emotion: {emo}]")
    print("Bot:", random.choice(REPLIES.get(emo, ["I hear you. Tell me more."])))
