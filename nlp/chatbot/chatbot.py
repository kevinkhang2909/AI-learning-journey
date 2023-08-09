from pathlib import Path
from langchain.document_loaders import TextLoader


file_path = Path.home() / 'PycharmProjects/AI-learning-journey/nlp/chatbot/temp/file.txt'

with open(str(file_path)) as f:
    text = f.read()

open(str(file_path)).read()

docs = TextLoader(str(file_path)).load()
