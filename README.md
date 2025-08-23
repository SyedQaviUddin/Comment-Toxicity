## 🛡️ Deep Learning for Comment Toxicity Detection with Streamlit

# 📖 Project Overview

This project develops a deep learning-based model to automatically detect toxic comments such as hate speech, harassment, or offensive language.
It combines NLP preprocessing, a BiLSTM neural network, and a Streamlit web app for real-time moderation.

The system helps online platforms and communities flag toxic content in real-time, supporting healthier and safer online discussions.

# 🎯 Objectives
Train a deep learning model to classify comments as toxic/non-toxic.
Build an interactive Streamlit app for real-time predictions.
Support bulk predictions via CSV upload.
Provide tools for content moderation in social media, forums, e-learning, and more.

# 🛠️ Skills Gained
Deep Learning (BiLSTM, NLP)
Model Training, Evaluation & Optimization
Streamlit Web App Development
Model Deployment

# 🧩 Problem Statement
Online communities are plagued by toxic comments.
The goal is to build an automated system to:
Detect toxic comments in real-time.
Assist moderators in filtering, warning, or reviewing harmful content.

# 🔑 Business Use Cases
Social Media Platforms → Auto-remove toxic comments.
Forums & Communities → Efficient moderation.
Brand Safety → Protect advertisers from offensive environments.
E-learning → Keep online classrooms safe.
News Platforms → Clean article comment sections.

## 🗂️ Project Structure
project/
│── toxicity_detection.ipynb    # Notebook (model training & evaluation)

│── app.py                      # Streamlit app (interactive UI)

│── train.csv                   # Training dataset

│── test.csv                    # Test dataset

│── model_bilstm.keras          # Saved trained model

│── tokenizer.joblib            # Saved tokenizer

│── meta.joblib                 # Metadata (threshold, config)

│── requirements.txt            # Dependencies

│── README.md                   # Project documentation

# ⚙️ Approach
-> EDA & Preprocessing
-> Clean comments (remove URLs, symbols, stopwords).
-> Tokenization & padding.
-> Model Development
-> Train a BiLSTM deep learning model.
-> Evaluate with Precision, Recall, F1-score.
->Save model + tokenizer + metadata.

Streamlit App
Enter a comment → get toxicity prediction.
Upload CSV → get batch predictions.
View probability scores and decisions.

# 📊 Results

Model trained with good F1-score (balance between precision & recall).
Real-time predictions available via Streamlit.
CSV bulk processing supported.

# 🚀 Deployment Guide
Run Locally
pip install -r requirements.txt
streamlit run app.py

App runs at: http://localhost:8501
Deploy Online (Streamlit Cloud)
Push repo to GitHub.

Go to https://share.streamlit.io
.

Select repo → choose app.py.

Deploy → get public app link.

For detailed setup, see Deployment Guide
 (if included).

# 📦 Deliverables

✅ toxicity_detection.ipynb → Notebook (training + evaluation)
✅ app.py → Streamlit app (real-time + CSV predictions)
✅ requirements.txt → Dependencies
✅ README.md → Documentation
✅ Demo Video → Walkthrough of notebook + app

## 🏁 Conclusion

This project demonstrates how Deep Learning + NLP + Streamlit can create a practical solution for comment toxicity detection, useful for real-world online platforms.


### Model File
Download trained model from Google Drive:
[model_bilstm.keras](https://drive.google.com/file/d/1cM0LTLDqcIEd3RQbP8kZspQBl11lCbLs/view?usp=sharing)



