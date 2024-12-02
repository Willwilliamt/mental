import streamlit as st
import tensorflow as tf
import numpy as np
import random
import pickle

# Load the trained model
model = tf.keras.models.load_model("mentalhealth_chatbot_model_final.h5.keras")

# Load the tokenizer and label encoder
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pickle", "rb") as enc:
    label_encoder = pickle.load(enc)

# Load the responses dictionary
with open("response.pickle", "rb") as resp:
    responses = pickle.load(resp)

# Function to predict the response
def get_response(user_input):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding="post", maxlen=20)
    prediction = model.predict(padded)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])
    return random.choice(responses[tag[0]])

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", page_icon=":brain:", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 50px;
    }
    .header {
        text-align: center;
        color: #2d6187;
        margin-bottom: 50px;
    }
    .chatbox {
        border: 2px solid #2d6187;
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
    }
    .input {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #2d6187;
        width: 100%;
    }
    .button {
        background-color: #2d6187;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='header'>Mental Health Chatbot</h1>", unsafe_allow_html=True)
st.write("Type your message in the chatbox below and get responses from the chatbot. Remember, this is a supportive tool, not a substitute for professional help.")
user_input = st.text_input("You:", key="input", help="Type your message here", placeholder="How are you feeling today?")
if st.button("Send", key="send"):
    if user_input:
        response = get_response(user_input)
        st.session_state.conversation.append((user_input, response))
        for user_msg, bot_msg in st.session_state.conversation:
            st.markdown(f"<div class='chatbox'><b>You:</b> {user_msg}<br><b>Chatbot:</b> {bot_msg}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please type a message before sending.")

st.markdown("<h2 class='header'>Our Mission</h2>", unsafe_allow_html=True)
st.write("""
    Our chatbot aims to provide initial support and information to individuals seeking help with their mental health.
    Please consult a professional for personalized advice and treatment.
""")

