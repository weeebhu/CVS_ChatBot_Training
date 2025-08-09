import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config.config import GEMINI_API_KEY

st.set_page_config(page_title = "Chatbot")

st.title("Langchain Chatbot")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GEMINI_API_KEY
)

# User input
user_input = st.text_input("Type your message here...")

if st.button("Send") and user_input:

    prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a friendly assistant. Respond in plain natural language: {question}"
    )

    query = prompt.format(question=user_input)
    response = llm.invoke(query)  # LangChain's ChatGoogleGenerativeAI returns natural text

    st.markdown(f"**ChatBot:** {response.content}")