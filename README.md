# 150K Lyrics Classifier 🎶

**Large-scale NLP project for emotion/genre classification from song lyrics. Built with Python, scikit-learn, and Streamlit. [Demo app included!]**

## 🚀 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Motivation & Impact](#motivation--impact)
- [Tech Stack](#tech-stack)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [About Me](#about-me)
- [License](#license)

## 🪄 Overview

Classify song lyrics by genre/emotion using NLP and ML. Trained on 5M+ lyrics—challenge: huge vocabulary, diverse styles.

## 🎮 Demo

[Live Streamlit App](YOUR-APP-LINK)
  
![Demo Screenshot](img/demo.png)

## 🤔 Motivation & Impact

Music analysis is vital for industry insights and personalized recommendations. This project tackles large-scale text classification, aiming to improve user engagement and genre prediction.

## ⚙️ Tech Stack

- Python, scikit-learn, pandas, Streamlit
- NLP: TFIDF, Logistic Regression, SVM, etc.
- Deployment: Streamlit Cloud
- Visualization: matplotlib/plotly

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
├── data/
├── src/
├── streamlit_app/
├── results/
├── README.md
└── requirements.txt
```

## 🤔 Motivation & Impact

The explosion of digital music platforms has made it crucial to understand song content at scale—for recommendation systems, playlist curation, and music analytics. Manual tagging is slow, inconsistent, and impractical for vast lyric databases. This project automates the process by applying advanced NLP and machine learning, enabling:

Fast, reliable prediction of song genre and emotional tone from raw lyrics

Insights into songwriter trends, audience preferences, and commercial potential

Improved music discovery and user engagement on streaming platforms

By bridging the gap between raw text and actionable metadata, this project highlights how machine learning can power better entertainment experiences and business decisions.