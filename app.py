import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from config.config import GEMINI_API_KEY

# loading file 
file_path = "data/BMW M.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 80,
    add_start_index = True
)
all_splits = text_splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector storage
vectorstore = FAISS.from_documents(all_splits, embeddings)

# Define Retrieval Tool
def retrieve_info(query: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

retrieval_tool = Tool(
    name="BMW_PDF_Search",
    func=retrieve_info,
    description="Use this to search the BMW M Series PDF for relevant information."
) 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY
)

# Agent
agent = initialize_agent(
    tools = [retrieval_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.set_page_config(page_title = "BMW Chatbot")
st.title("BMW Chatbot")

# User input
user_input = st.text_input("Ask about the BMW M Series...")

if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        response = agent.run(user_input)
    st.markdown(f"**ChatBot:** {response}")