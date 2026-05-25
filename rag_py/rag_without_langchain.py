import os
import re
import time
import json
import math
import sys
import requests
import pypdf
import sqlite3
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

load_dotenv()


def clearMsg():
    sys.stdout.write(f"\r{' ':<100}\r")
    sys.stdout.flush()


def print_status(message=""):
    sys.stdout.write(f"\r{message:<100}")
    sys.stdout.flush()


def dot_product(vec_a, vec_b):
    total_sum = 0
    for i in range(len(vec_a)):
        total_sum += vec_a[i] * vec_b[i]
    return total_sum


def magnitude(vec):
    return math.sqrt(sum(x**2 for x in vec))


def cosine_similarity(vec_a, vec_b):
    """
    cosθ = (A . B)/(|A||B|)
    """
    mag_a = magnitude(vec_a)
    mag_b = magnitude(vec_b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product(vec_a, vec_b) / (mag_a * mag_b)


# Reads PDF
def readFile(file_path):
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# Creates chunks out of the extracted text
def createChunks(text, chunk_size, overlap):
    words = re.findall(r"\b[\w'-]+\b", text)
    text_len = len(words)
    start = 0
    chunk_list = []
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = " ".join(words[start:end])
        chunk_list.append(chunk)
        start = start + chunk_size - overlap
    return chunk_list


# Creates embeddings for the chunks
def createEmbeddings(chunk_list):
    embed_api_url = f"https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    response = requests.post(
        embed_api_url,
        json={"inputs": {"sentences": chunk_list}, "options": {"wait_for_model": True}},
        headers=headers,
    )
    if response.status_code != 200:
        raise Exception(f"HF API Error: {response.status_code} - {response.text}")
    return response.json()


# Creates embedding for the user query
def createQueryEmbedding(query):
    return createEmbeddings(chunk_list=[query])[0]


# Getting SQL connection
def getSqliteConnection(db_path="./rag_pipeline.db", clear_on_start=True):
    conn = sqlite3.connect(db_path)

    if clear_on_start:
        conn.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_text TEXT,
            embedding_json TEXT
        );
    """)
    conn.commit()
    return conn


# Storing the data in SQL, embeddings ke string e convert korte hobe to store as Text
def storeSqlite(chunk_list, raw_embeddings, conn):
    cursor = conn.cursor()
    for i in range(len(chunk_list)):
        embedding_string = json.dumps(raw_embeddings[i])
        cursor.execute(
            "INSERT INTO documents (chunk_text, embedding_json) VALUES (?, ?)",
            (chunk_list[i], embedding_string),
        )
    conn.commit()


# Calls GROQ api
def callLLM(system_prompt, user_prompt):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    response = requests.post(
        api_url,
        json={
            "model": "openai/gpt-oss-120b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        headers=headers,
    )
    if response.status_code != 200:
        raise Exception(f"GROQ API Error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"]


# Checks cosine similarity between query and stored chunks
def getSimilarity(conn, query_embeddings, k):
    cursor = conn.execute("SELECT chunk_text, embedding_json FROM documents")
    rows = cursor.fetchall()

    if not rows:
        return []

    scored_chunks = []
    for row in rows:
        doc = row[0]
        emb = json.loads(row[1])

        score = cosine_similarity(query_embeddings, emb)
        scored_chunks.append({"document": doc, "score": score})

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = [item["document"] for item in scored_chunks[:k]]
    return top_chunks


# Reads, chunks and creates embeddings for a single pdf
def processPDF(file_path):
    print_status("Reading file: " + file_path)
    text = readFile(file_path)
    if not text:
        return file_path, [], []

    print_status("Chunking document: " + file_path)
    chunks = createChunks(text, chunk_size=50, overlap=10)

    try:
        print_status("Creating embeddings for: " + file_path)
        embeddings = createEmbeddings(chunks)
        return file_path, chunks, embeddings
    except Exception as e:
        return file_path, [], []


# Walks the directory and processes every pdf concurrently
def ingestDirectory(target_dir, db_conn):
    if not os.path.exists(target_dir):
        print(f"Directory '{target_dir}' does not exist.")
        return

    pdf_paths = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        print("No PDF files found in the directory.")
        return

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(processPDF, path): path for path in pdf_paths}

        for future in as_completed(futures):
            file_path, chunks, embeddings = future.result()

            if chunks and embeddings:
                storeSqlite(chunks, embeddings, db_conn)


def get_system_prompt_template(context=""):
    with open("prompts/system_prompt.md", "r", encoding="utf-8") as file:
        content = file.read()

    if context:
        content = content.replace("{context}", context)

    return content


def get_user_prompt_template(context="", query=""):
    with open("prompts/user_prompt.md", "r", encoding="utf-8") as file:
        content = file.read()

    if context:
        content = content.replace("{context}", context)

    if query:
        content = content.replace("{query}", query)

    return content


if __name__ == "__main__":
    # User query
    query = "How many types of loops are present in python?"

    # initiating sqlite
    conn = getSqliteConnection()

    print_status("Starting Ingestion...")

    # Ingesting all pdfs from a directory
    ingestDirectory("documents", db_conn=conn)

    print_status("Generating query embeddings...")

    # creating embeddings for the user query
    query_embeddings = createQueryEmbedding(query=query)

    print_status("Started similarity search...")

    # Querying chroma (similarity search)
    retrieved_chunks = getSimilarity(conn=conn, query_embeddings=query_embeddings, k=5)
    conn.close()

    print_status("Similar chunks retrieved...")

    # Creating context string to pass to llm
    context = "\n---\n".join(retrieved_chunks)

    # system_prompt = (
    #     "You are a helpful assistant. Answer the user's question using ONLY the provided text context. "
    #     "If the answer cannot be found in the context, say 'I cannot find the answer in the document.' "
    #     "Do not make up information or use outside knowledge."
    # )

    system_prompt = get_system_prompt_template()

    # user_prompt = f"""Context:
    # {context}
    # Question: {query}
    # Answer:"""

    user_prompt = get_user_prompt_template(context=context, query=query)

    print_status("Prompting LLM...")

    # Calling GROQ Api
    response = callLLM(system_prompt=system_prompt, user_prompt=user_prompt)

    clearMsg()
    print("\rResponse:\n", flush=True)
    print(response)
