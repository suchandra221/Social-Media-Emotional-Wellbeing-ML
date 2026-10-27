
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. LOAD THE DATASET

df = pd.read_csv("test.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove completely empty rows
df = df.dropna(how="all")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# 2. DATA CLEANING

# Remove rows without a target emotion
df = df.dropna(subset=["Dominant_Emotion"]).copy()

# Remove extra spaces from text values
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# Convert numerical columns to numeric values
numeric_columns = [
    "Age",
    "Daily_Usage_Time (minutes)",
    "Posts_Per_Day",
    "Likes_Received_Per_Day",
    "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing feature values
df = df.dropna(
    subset=numeric_columns + ["Gender", "Platform"]
)

print("\nMissing values handled.")
print("Clean dataset shape:", df.shape)

# 3. SEPARATE FEATURES AND TARGET

# User_ID is an identifier, so it is not used as a feature
X = df.drop(columns=["User_ID", "Dominant_Emotion"])
y = df["Dominant_Emotion"].astype(str)

print("\nFeatures:")
print(X.columns.tolist())

print("\nEmotion categories:")
print(y.value_counts())

# 4. PREPROCESS THE DATA

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# 5. CREATE THE RANDOM FOREST MODEL

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# 6. SPLIT THE DATASET

# Stratification preserves emotion proportions where possible
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 7. TRAIN THE MODEL

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# 8. MAKE PREDICTIONS

y_pred = model.predict(X_test)

# 9. EVALUATE THE MODEL

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# 10. CONFUSION MATRIX

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()
plt.show()

# 11. SAMPLE USER PREDICTION

sample_user = X.iloc[[0]]

predicted_emotion = model.predict(sample_user)[0]

print("\n========== SAMPLE PREDICTION ==========")
print("User details:")
print(sample_user.to_string(index=False))
print("Predicted Dominant Emotion:", predicted_emotion)

# 12. GRAPH 1: EMOTION DISTRIBUTION

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Dominant_Emotion"
)

plt.title("Distribution of Dominant Emotions")
plt.xlabel("Dominant Emotion")
plt.ylabel("Number of Users")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 13. GRAPH 2: DAILY USAGE BY EMOTION

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Dominant_Emotion",
    y="Daily_Usage_Time (minutes)",
    errorbar=None
)

plt.title("Average Daily Social Media Usage by Emotion")
plt.xlabel("Dominant Emotion")
plt.ylabel("Daily Usage (minutes)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 14. GRAPH 3: PLATFORM DISTRIBUTION

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Platform"
)

plt.title("Social Media Platform Distribution")
plt.xlabel("Social Media Platform")
plt.ylabel("Number of Users")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 15. SAVE THE TRAINED MODEL

import joblib

joblib.dump(model, "emotion_model.pkl")

print("\nTrained model saved as emotion_model.pkl")
