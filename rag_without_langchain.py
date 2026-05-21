import os
import re
import requests
import pypdf
import chromadb
from dotenv import load_dotenv

load_dotenv()


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


# Instansiate chroma instance and return the collection
def getChromaCollection(name):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name=name)


# Store the embeddings in Chroma db
def storeChroma(chunk_list, raw_embeddings, collection):
    string_ids = [f"id_{i}" for i in range(len(chunk_list))]
    collection.add(embeddings=raw_embeddings, documents=chunk_list, ids=string_ids)


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


if __name__ == "__main__":
    # User query
    query = "What is the significance of Regulatory guidelines while maintaining strict data integrity"

    # Reading the pdf
    text = readFile(file_path="documents/test.pdf")

    # Creating chunks out of the pdf text
    chunks = createChunks(text=text, chunk_size=50, overlap=10)

    # Creating embeddings for the chunks
    raw_embeddings = createEmbeddings(chunks)

    # instansiating chroma
    collection = getChromaCollection(name="compliance_collection")

    # Storing the chunk embeddings in chroma
    storeChroma(chunk_list=chunks, raw_embeddings=raw_embeddings, collection=collection)

    # creating embeddings for the user query
    query_embeddings = createQueryEmbedding(query=query)

    # Querying chroma (similarity search)
    results = collection.query(query_embeddings=[query_embeddings], n_results=5)

    # getting the texts for associated embeddings
    retrieved_chunks = results["documents"][0] if results["documents"] else []

    # Creating context string to pass to llm
    context = "\n---\n".join(retrieved_chunks)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided text context. "
        "If the answer cannot be found in the context, say 'I cannot find the answer in the document.' "
        "Do not make up information or use outside knowledge."
    )

    user_prompt = f"""Context:
    {context}
    Question: {query}
    Answer:"""

    # Calling GROQ Api
    response = callLLM(system_prompt=system_prompt, user_prompt=user_prompt)

    print(response)
