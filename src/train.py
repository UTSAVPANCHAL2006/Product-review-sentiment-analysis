import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from data_split import load_data
from preproces import prepare_tokenizer, preprocess_texts
from model import build_model
import os

def train_model():
    X_train, X_test, y_train, y_test = load_data()

    print(f"✅ Loaded data: {len(X_train)} train / {len(X_test)} test")
    print("Unique labels:", np.unique(y_train))

    num_classes = len(np.unique(y_train))
    print(f"Detected {num_classes} classes")

    tokenizer = prepare_tokenizer(X_train)
    X_train_seq = preprocess_texts(X_train, tokenizer)
    X_test_seq = preprocess_texts(X_test, tokenizer)

    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")

    model = build_model(vocab_size=vocab_size, num_classes=num_classes, input_length=X_train_seq.shape[1])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(
        X_train_seq, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )

    y_pred = model.predict(X_test_seq)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred_classes))

    os.makedirs('models', exist_ok=True)
    model.save('models/text_classification_model.h5')
    print("✅ Model saved to models/text_classification_model.h5")

if __name__ == "__main__":
    train_model()
