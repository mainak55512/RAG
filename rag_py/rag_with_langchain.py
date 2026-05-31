import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = "documents"
DB_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_WORKERS = 4


def process_single_pdf(pdf_path):
    """Worker function to load and split a single PDF file."""
    try:
        print(f"[+] Processing: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)
        print(f"[✓] Successfully split {pdf_path} into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"[X] Error processing {pdf_path}: {e}")
        return []


def main():
    ingest = False
    # ingest = True

    if ingest:
        pdf_files = glob(os.path.join(DOCS_DIR, "*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in '{DOCS_DIR}' directory.")
            return

        all_chunks = []

        print(f"Starting ingestion pool with {MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files
            }

            for future in as_completed(future_to_pdf):
                chunks = future.result()
                if chunks:
                    all_chunks.extend(chunks)

        if not all_chunks:
            print("No text chunks extracted. Exiting.")
            return

        print(f"\nTotal chunks extracted across all files: {len(all_chunks)}")

        embedding_model = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.environ["HF_TOKEN"],
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
        )

        # 4. Store all collected chunks into Chroma in manageable batches
        print("Generating embeddings and saving to Chroma in batches...")

        # Initialize Chroma store with an empty list or the first batch
        BATCH_SIZE = 32
        vector_store = None

        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            print(
                f" -> Processing embedding batch {i // BATCH_SIZE + 1}/{(len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}..."
            )

            if vector_store is None:
                vector_store = Chroma.from_documents(
                    documents=batch, embedding=embedding_model, persist_directory=DB_DIR
                )
            else:
                vector_store.add_documents(documents=batch)

        print("[✓] Vector store built successfully.")
    else:
        embedding_model = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.environ["HF_TOKEN"],
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
        )

        vector_store = Chroma(
            persist_directory="./chroma_db", embedding_function=embedding_model
        )

        # 5. User query & Similarity search
        query = "What is the difference between list and numpy array?"
        docs_mapping = vector_store.similarity_search(query, k=5)

        # 6. Building context string
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

        # 7. Calling GROQ Api with a valid model name
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        messages = [("system", system_prompt), ("user", user_prompt)]

        print("Querying LLM...")
        response = llm.invoke(messages).content
        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    main()
