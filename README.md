# 150K Lyrics Classifier 🎶

**Large-scale NLP project for emotion/genre classification from song lyrics. Built with Python, scikit-learn, and Streamlit. [Demo app included!]**

## 🚀 Table of Contents

- Overview
- Demo
- Motivation & Impact
- Tech Stack
- Installation & Usage
- Results
- Project Structure
- About Me
- License

## 🪄 Overview

The **150K Lyrics Classifier** is an advanced natural language processing (NLP) project designed to automatically classify song lyrics by genre and emotion. Leveraging a robust dataset of over 150,000 unique lyrics, this project tackles the challenge of large-scale text classification in the music domain, where linguistic diversity and subtle emotional markers are key.

Built using Python and industry-standard ML libraries (scikit-learn, pandas), the project offers:
- End-to-end pipeline from data cleaning to model evaluation
- Customizable feature engineering (TF-IDF), multi-model training (Logistic Regression, SVM)
- Interactive Streamlit app for live predictions, results visualization, and user engagement

By automating lyric analysis, this project demonstrates applied machine learning in entertainment analytics—showcasing skills across NLP, model deployment, and user-focused app development. Recruiters see best practices in code organization, reproducibility, and impactful communication.

## 🎮 Demo
[Live Streamlit App](YOUR-APP-LINK)

![Demo Screenshot](img/demo.png)

## 🤔 Motivation & Impact

The explosion of digital music platforms has made it crucial to understand song content at scale—for recommendation systems, playlist curation, and music analytics. Manual tagging is slow, inconsistent, and impractical for vast lyric databases. This project automates the process by applying advanced NLP and machine learning, enabling:

- Fast, reliable prediction of song genre and emotional tone from raw lyrics
- Insights into songwriter trends, audience preferences, and commercial potential
- Improved music discovery and user engagement on streaming platforms

By bridging the gap between raw text and actionable metadata, this project highlights how machine learning can power better entertainment experiences and business decisions.

## ⚙️ Tech Stack

- Python
- scikit-learn, pandas, numpy
- Streamlit (for interactive demo app)
- Visualization: matplotlib, plotly
- Deployment: Streamlit Cloud

## 📦 Installation & Usage

```bash
Clone repo
git clone https://github.com/yukta1103/150K-Lyrics-Classifier.git
cd 150K-Lyrics-Classifier

Install dependencies
pip install -r requirements.txt

Download dataset from Kaggle
Place file at: data/lyrics.csv
Train model
python src/train.py --lyrics_col lyrics --title_col title --artist_col artist

Launch Streamlit app
streamlit run streamlit_app/app.py
```


## 📊 Results

| Model                     | Accuracy | F1 Score |
|---------------------------|----------|----------|
| TFIDF + Logistic Regression | 87%      | 0.82     |

Sample Output:
- “I walked across an empty land…” → **Genre: Pop / Emotion: Nostalgic**
- “I got the horses in the back…” → **Genre: Country / Emotion: Angry**

Confusion matrix and visualizations are available in the `results/` folder.

## 📁 Project Structure

```bash
├── data/ # Lyrics dataset
├── src/ # Training and model scripts
├── streamlit_app/ # Streamlit interactive app
├── results/ # Plots, metrics, and outputs
├── README.md
└── requirements.txt
```
