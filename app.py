import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

st.set_page_config(page_title = "BMW Chatbot")

st.title("BMW Chatbot")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY
)

# User input
user_input = st.text_input("Type your message here...")

if st.button("Send") and user_input:
    # Retrive relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    relevent_docs = retriever.get_relevant_documents(user_input)

    def score_doc(doc, query):
        # Simple prompt to score relevance 0-100 (example)
        scoring_prompt = f"""
        On a scale from 0 to 100, how relevant is the following text to the question?  
        Text: \"\"\"{doc.page_content}\"\"\"
        Question: \"\"\"{query}\"\"\"
        Relevance score (only a number):
        """
        response = llm.invoke(scoring_prompt)
        try:
            score = int(''.join(filter(str.isdigit, response.content)))
        except:
            score = 0
        return score
    
    # Score all candidates
    scored_docs = [(doc, score_doc(doc, user_input)) for doc in relevent_docs]
    # Sort by score descending
    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # Step 3: Take top 3 reranked docs
    top_docs = [doc for doc, score in scored_docs[:3]]

    # Combine top docs into context
    context_text = "\n\n".join([doc.page_content for doc in top_docs])

    # # combine docs into context
    # context_text ="\n\n".join([doc.page_content for doc in relevent_docs])

    # Create final prompt

    prompt_template="""
    you are a helpful assistant. use the following context from the PDF to answer the question.
    Context:
    {context}

    Question: {question}
"""
    prompt = prompt_template.format(context=context_text, question=user_input)

    # Get answer
    response = llm.invoke(prompt)  # LangChain's ChatGoogleGenerativeAI returns natural text
    st.markdown(f"**ChatBot:** {response.content}")