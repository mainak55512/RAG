import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# Loading pdf
loader = PyPDFLoader("documents/complience.pdf")
docs = loader.load()

# Splitting text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=40)
final_chunks = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
)

# Storing embeddings for the chunks in Chroma (one time)
# vector_store = Chroma.from_documents(
#     documents=final_chunks, embedding=embedding_model, persist_directory="./chroma_db"
# )

vector_store = Chroma(
    persist_directory="./chroma_db", embedding_function=embedding_model
)

# User query
query = "What is the main topic of the PDF?"

# Similarity search for the query in chroma
docs_mapping = vector_store.similarity_search(query, k=5)

# Building context string to pass to llm
context = "\n---\n".join([doc.page_content for doc in docs_mapping])

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
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=None)
messages = [("system", system_prompt), ("user", user_prompt)]
response = llm.invoke(messages).content

print(response)
