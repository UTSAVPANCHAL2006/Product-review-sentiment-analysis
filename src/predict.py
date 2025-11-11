from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model('models/text_classification_model.h5')

with open('saved_models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 200

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    
    label = np.argmax(pred, axis=1)[0]
    labels = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😄"}
    
    print("📝 Text:", text)
    print(f"🎯 Sentiment: {labels[label]}")
    print(f"📊 Confidence: {pred[0][label]*100:.2f}%\n")
    return label, pred[0][label]

if __name__ == "__main__":
    print("\n🚀 Sentiment Predictor Ready!\n")

    examples = [
        "This product is absolutely amazing, loved it!",
        "It’s okay, nothing special.",
        "Worst purchase I’ve ever made.",
        "Excellent quality and super fast delivery!",
        "Terrible service, I want a refund.",
        "Average product but good packaging."
    ]
    
    for text in examples:
        predict_sentiment(text)
