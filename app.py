import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import Tool, tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config.config import GEMINI_API_KEY, WEATHER_API_KEY, QDRANT_API_KEY
import requests
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Load and split PDF 
file_path = "data/BMW M.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Embeddings 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Store
client = QdrantClient(
    url="https://d4cf864f-232d-4dfa-850f-c7132069f31c.us-west-1-0.aws.cloud.qdrant.io",
    api_key=QDRANT_API_KEY,
    timeout=120
)

vectorstore = Qdrant(
    client=client,
    collection_name="bmw_docs",
    embeddings=embeddings
)

batch_size = 32
for i in range(0, len(all_splits), batch_size):
    batch = all_splits[i:i + batch_size]
    vectorstore.add_documents(batch)
    print(f"Inserted batch {i//batch_size + 1} / {len(all_splits)//batch_size + 1}")

# Reranker 
hf_cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
compressor = CrossEncoderReranker(
    model=hf_cross_encoder,
    top_n=3
)

retriever = vectorstore.as_retriever(search_kwargs={"k":5})
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Tools
def retrieve_info(query: str):
    """Retrieve relevant sections from BMW PDF using the retriever."""
    docs = rerank_retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

retrieval_tool = Tool(
    name="BMW_PDF_Search",
    func=retrieve_info,
    description="Search the BMW M Series PDF for relevant information."
)

# Math tool
def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b

# Weather tool
@tool
def get_weather(city: str) -> str:
    """
    Fetch the current weather for a given city using OpenWeatherMap API.
    Example: get_weather("Berlin")
    """
    try:
        url=f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            return f"Error: {data.get('message','unknown error')}"

        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]

        return f"The current weather in {city} is {weather_desc} with a temperature of {temp}°C (feels like {feels_like}°C)."
    except Exception as e:
        return f"Failed to fetch weather data: {str(e)}"

# Gender tool
@tool
def get_gender(name: str) -> str:
    """
    Predict the gender for a given name using the genderize.io API.
    Example: get_gender("Alice")
    """
    try:
        url=f"https://api.genderize.io?name={name}"
        response = requests.get(url)
        data = response.json()
        gender = data.get("gender")
        probab = data.get("probability")
        if gender is None:
            return f"Sorry, I couldn't determine the gender for {name}."
        return f"I think {name} has a {float(probab)*100:.1f}% probability of being {gender}."
    except Exception as e:
        return f"Failed to fetch gender data: {str(e)}"

# Random joke tool
@tool
def random_joke(query: str) -> str:
    """
    Return a random joke from the official joke API.
    Example: random_joke("Tell me a joke")
    """
    try:
        url="https://official-joke-api.appspot.com/random_joke"
        response = requests.get(url)
        data = response.json()
        typeOfJoke = data["type"]
        setup = data["setup"]
        punchline = data["punchline"]
        return f"Here is a {typeOfJoke} joke for you:\n{setup}\n{punchline}"
    except Exception as e:
        return f"Failed to fetch joke: {str(e)}"

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY
)

# Agents
bmw_agent = create_react_agent(
    model=llm,
    tools=[retrieval_tool],
    prompt=(
        "You are a BMW expert. Always use the BMW_PDF_Search tool to answer user queries.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with BMW-related query\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."),
    name="bmw_agent"
)

math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)

weather_agent = create_react_agent(
    model = llm,
    tools=[get_weather],
    prompt=(
        "You are a weather assistant. Fetch and explain current weather for cities.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with weather realated query\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="weather_agent"
)

gender_agent = create_react_agent(
    model=llm,
    tools=[get_gender],
    prompt=(
        "You are a gender guesser.\n\n"
        "INSTRUCTIONS:\n"
        "- Predict the gender for a given name using the genderize.io API.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="gender_agent"
)

joke_agent = create_react_agent(
    model=llm, 
    tools = [random_joke], 
    prompt=(
        "You are funny agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Tell a random joke whenever asked.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="joke_agent"
)

# Supervisor
workflow = create_supervisor(
    model=llm,
    agents=[bmw_agent,math_agent,weather_agent,gender_agent,joke_agent],
    prompt=(
        "You are a supervisor managing five agents:\n"
        "- a bmw agent. Knows everythin about bmw m cars"
        "- a math agent. Knows everything about math.\n"
        "- a weather agent. given the name of city you know what weather is there.\n"
        "- a gender agent. you are expert in guessing gender by names.\n"
        "- joke agent. you are a funny assistant and can tell a joke when asked.\n"
        "Do not do any work yourself."
        )
)

supervisor = workflow.compile(name="supervisor")

# Streamlit UI
import streamlit as st

st.set_page_config(page_title="BMW Chatbot")
st.title("BMW Chatbot")

# Input
user_input = st.text_input("Ask about BMW M Series, weather, math, or other info...")

# Query & response
if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        # Invoke supervisor with the user input
        final_state = supervisor.invoke({"messages": [{"role": "user", "content": user_input}]})

    # Get response
    messages = final_state.get("messages", [])
    if messages:
        bot_reply = messages[-1].content
    else:
        bot_reply = "Sorry, I didn’t get a reply."

    # Display response
    st.markdown(bot_reply)

