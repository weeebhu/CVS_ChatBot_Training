import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY =os.getenv("WEATHER_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
