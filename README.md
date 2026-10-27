# Social Media Usage and Emotional Well-Being

## Project Overview
This machine learning project analyzes social media usage patterns and predicts the dominant emotion recorded in the dataset.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit

## Dataset Features
- Age
- Gender
- Social media platform
- Daily usage time
- Posts per day
- Likes received per day
- Comments received per day
- Messages sent per day

**Target variable:** `Dominant_Emotion`

## Machine Learning Model
A Random Forest Classifier is trained to predict the recorded dominant emotion based on the user's social media usage patterns.

## Model Evaluation
The project evaluates model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## Run the Project

Install the dependencies:

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

Run the machine learning script:

```bash
python main.py
```

Launch the web application:

```bash
python -m streamlit run app.py
```

## Disclaimer
Predictions reflect patterns in the available dataset. They are not psychological diagnoses and do not establish that social media usage causes particular emotions.
