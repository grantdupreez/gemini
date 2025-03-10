import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
import PyPDF2
import os
import io
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


st.title("Chat Your PDFs")  # Updated title

# Retrieve API key from environment variable
google_api_key = st.secrets['auth_key']

# Check if the API key is available
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()

# File Upload with user-defined name
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.text("PDF File Uploaded Successfully!")

    pdf_data = uploaded_file.read()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
    pdf_pages = pdf_reader.pages

    # Create Context
    context = "\n\n".join(page.extract_text() for page in pdf_pages)

    # Split Texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(context)

    # Chroma Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

    # Get User Question
    user_question = st.text_input("Ask a Question:")

    if st.button("Get Answer"):
        if user_question:
            # Get Relevant Documents
            docs = vector_index.get_relevant_documents(user_question)

            # Define Prompt Template
            prompt_template = """
            Answer the question as detailed as possible from the provided context,
            make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context",
            don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n
            Answer:
            """

            # Create Prompt
            prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

            # Load QA Chain
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=google_api_key)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

            # Get Response
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            # Display Answer
            st.subheader("Answer:")
            st.write(response['output_text'])

        else:
            st.warning("Please enter a question.")
