
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Social Media & Emotional Well-Being",
    page_icon="📱"
)

model = joblib.load("emotion_model.pkl")

st.title("📱 Social Media Usage & Emotional Well-Being")
st.write("Explore how social media usage patterns relate to recorded emotions.")

st.subheader("Enter User Details")

age = st.number_input("Age", min_value=13, max_value=100, value=22)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Non-binary"]
)

platform = st.selectbox(
    "Social Media Platform",
    ["Instagram", "Facebook", "Snapchat", "Telegram", "LinkedIn"]
)

daily_usage = st.number_input(
    "Daily Usage (minutes)", min_value=0, value=120
)

posts = st.number_input(
    "Posts Per Day", min_value=0, value=3
)

likes = st.number_input(
    "Likes Received", min_value=0, value=40
)

comments = st.number_input(
    "Comments", min_value=0, value=15
)

messages = st.number_input(
    "Messages Sent", min_value=0, value=20
)


if st.button("Predict Dominant Emotion"):

    user_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Platform": platform,
        "Daily_Usage_Time (minutes)": daily_usage,
        "Posts_Per_Day": posts,
        "Likes_Received_Per_Day": likes,
        "Comments_Received_Per_Day": comments,
        "Messages_Sent_Per_Day": messages
    }])

    prediction = model.predict(user_data)[0]

    st.success(f"Predicted Emotion: {prediction}")

    st.caption(
        "This is a machine-learning prediction based on "
        "patterns in the dataset, not a psychological assessment."
    )


    prediction = model.predict(user_data)[0]

    st.success(f"Predicted Emotion: {prediction}")

    st.caption(
        "This is a machine-learning prediction based on patterns "
        "in the dataset, not a psychological assessment."
    )
