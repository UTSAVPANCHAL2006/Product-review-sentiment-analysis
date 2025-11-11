import os
import pickle
from data_split import load_data
from preproces import prepare_tokenizer

# Step 1: Load your dataset
X_train, X_test, y_train, y_test = load_data()

# Step 2: Prepare tokenizer
tokenizer = prepare_tokenizer(X_train)

# Step 3: Save tokenizer
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

with open('saved_models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer saved successfully at saved_models/tokenizer.pkl")
