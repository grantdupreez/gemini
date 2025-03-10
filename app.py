import os
import streamlit as st
import google.generativeai as genai
import hmac
from functions import*

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()

API_KEY = st.secrets['auth_key']

st.set_page_config(
    page_title="Chat with Gemini!",
    page_icon=":robot_face:",  
    layout="wide",  
)

gpt.configure(api_key=API_KEY)
model = gpt.GenerativeModel('gemini-2.0-flash-001')

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

st.title("🤖 Chat with Gemini 2.0")

# Display the chat history
for msg in st.session_state.chat_session.history:
    with st.chat_message(map_role(msg["role"])):
        st.markdown(msg["content"])

# Input field for user's message
user_input = st.chat_input("Ask Gemini...")
if user_input:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_input)

    # Send user's message to Gemini and get the response
    gemini_response = fetch_gemini_response(user_input)

    # Display Gemini's response
    with st.chat_message("assistant"):
        st.markdown(gemini_response)

    # Add user and assistant messages to the chat history
    st.session_state.chat_session.history.append({"role": "user", "content": user_input})
    st.session_state.chat_session.history.append({"role": "model", "content": gemini_response})
    
