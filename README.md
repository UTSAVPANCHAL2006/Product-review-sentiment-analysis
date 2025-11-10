
#  🧠 Product Review Sentiment Analysis (Deep Learning + NLP )


This project uses a **BiLSTM Deep Learning model** to classify product reviews into  
**Positive 😄**, **Neutral 😐**, and **Negative 😠** sentiments.  
It’s built using **TensorFlow**, **Keras**, **NLTK**, and **scikit-learn**, with a clean modular ML pipeline.

---

## 📁 Project Structure
Product-review-sentiment-analysis/

```
─ 📁 data/
    ├── Reviews.csv
    └── clean-data.csv
─ 📁 models/
    ├── text_classification_model.h5
    └── tokenizer.pkl
─ 📁 src/
    ├── data_split.py
    ├── preproces.py
    ├── model.py
    ├── train.py
    ├── predict.py
    ├── save_tokenizer.py
    └── init.py
─ 📁 notebooks/
    └── eda.ipynb
─ .gitignore
─ requirements.txt
─ README.md

```
---

## 🚀 Features

✅ Clean text preprocessing (stopwords, punctuation, lowercase)  
✅ Tokenization & padding for sequence modeling  
✅ BiLSTM model for contextual sentiment understanding  
✅ EarlyStopping for efficient training  
✅ Model & tokenizer saving for later inference  
✅ Real-time review prediction  
✅ Fully modular pipeline (easy to extend or deploy)

---

## 🧩 Technologies Used

| Component | Library |
|------------|----------|
| **Language** | Python 3.11 |
| **Deep Learning** | TensorFlow / Keras |
| **Data Processing** | Pandas, NumPy |
| **Text Cleaning** | NLTK |
| **Model Evaluation** | scikit-learn |
| **Visualization** | Matplotlib |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Product-review-sentiment-analysis.git
cd Product-review-sentiment-analysis
```


### 2️⃣ Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # (Mac / Linux)
# OR
.venv\Scripts\activate       # (Windows)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare the Dataset
```bash
The full dataset (~300 MB) is not uploaded due to GitHub's size limit.
You can download it from here:

Then place it inside your data/ folder:
data/clean-data.csv
```
### 5️⃣ Train the Model
```bash
python src/train.py

You’ll see output like:

Epoch 1/5
accuracy: 0.87 - val_accuracy: 0.89
✅ Model saved to models/text_classification_model.h5
```
### 6️⃣ Predict Sentiment (Real-Time)
```bash
python src/predict.py

Example Output:
📝 Text: This product is absolutely amazing, loved it!
🎯 Sentiment: Positive 😄
📊 Confidence: 99.53%

📝 Text: Worst purchase ever.
🎯 Sentiment: Negative 😠
📊 Confidence: 98.92%
```

👨‍💻 Author :-

Utsav

📧 [utsavpanchal2756@gmail.com]

🌐 github.com/UTSAVPANCHAL2006
