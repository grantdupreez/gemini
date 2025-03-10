import streamlit as st
import os
from streamlit_option_menu import option_menu
from PIL import Image
import google.generativeai as genai
import hmac

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

# Set up Google Gemini-Pro AI model
genai.configure(api_key=API_KEY)

# load gemini-pro model
def gemini_mod():
    model = genai.GenerativeModel('gemini-2.0-flash-001')
    return model

# Load gemini vision model
def gemini_vision():
    model = genai.GenerativeModel('gemini-pro-vision')
    return model

# get response from gemini pro vision model
def gemini_visoin_response(model, prompt, image):
    response = model.generate_content([prompt, image])
    return response.text

# Set page title and icon

st.set_page_config(
    page_title="Chat With Gem",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    user_picked = option_menu(
        "Google Gemini AI",
        ["ChatBot", "Image Captioning"],
        menu_icon="robot",
        icons = ["chat-dots-fill", "image-fill"],
        default_index=0
    )

def roleForStreamlit(user_role):
    if user_role == 'model':
        return 'assistant'
    else:
        return user_role
    

if user_picked == 'ChatBot':
    model = gemini_mod()
    
    if "chat_history" not in st.session_state:
        st.session_state['chat_history'] = model.start_chat(history=[])

    st.title("🤖TalkBot")

    #Display the chat history
    for message in st.session_state.chat_history.history:
        with st.chat_message(roleForStreamlit(message.role)):    
            st.markdown(message.parts[0].text)

    # Get user input
    user_input = st.chat_input("Message TalkBot:")
    if user_input:
        st.chat_message("user").markdown(user_input)
        reponse = st.session_state.chat_history.send_message(user_input)
        with st.chat_message("assistant"):
            st.markdown(reponse.text)


if user_picked == 'Image Captioning':
    model = gemini_vision()

    st.title("🖼️Image Captioning")

    image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    user_prompt = st.text_input("Enter the prompt for image captioning:")

    if st.button("Generate Caption"):
        load_image = Image.open(image)

        colLeft, colRight = st.columns(2)

        with colLeft:
            st.image(load_image.resize((800, 500)))

        caption_response = gemini_visoin_response(model, user_prompt, load_image)

        with colRight:
            st.info(caption_response)
