import os
import tempfile
import pickle
from docx import Document
import pandas as pd

import constants
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def read_excel_or_csv(file_path):
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    content = []
    for _, row in df.iterrows():
        content.append("\t".join(map(str, row)))
    return "\n".join(content)

os.environ["OPENAI_API_KEY"] = constants.APIKEY
temp_dir = tempfile.mkdtemp()

for file_path in os.listdir("."):
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    elif file_path.endswith(".docx"):
        doc_content = read_docx(file_path)
        temp_file_path = os.path.join(temp_dir, os.path.basename(file_path) + ".txt")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
    elif file_path.endswith(('.xls', '.xlsx', '.csv')):
        file_content = read_excel_or_csv(file_path)
        temp_file_path = os.path.join(temp_dir, os.path.basename(file_path) + ".txt")
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

loader = DirectoryLoader(temp_dir)
index = VectorstoreIndexCreator().from_loaders([loader])

chat_history_path = "chat_history.pkl"
if os.path.exists(chat_history_path):
    with open(chat_history_path, 'rb') as f:
        chat_history = pickle.load(f)
else:
    chat_history = []

while True:
    query = input("Please enter your query (or 'q', 'quit', 'exit' to stop): ")
    if query.lower() in ["q", "quit", "exit"]:
        # Save chat history before exiting
        with open(chat_history_path, 'wb') as f:
            pickle.dump(chat_history, f)
        break

    chat_history.append(("User", query))

    response = index.query(query, llm=ChatOpenAI(model="gpt-3.5-turbo"))
    print(response, "\n")
    chat_history.append(("Bot", response))
