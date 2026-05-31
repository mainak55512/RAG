import os
import random
import re
import time
import json
import math
import sys
import requests
import pypdf
import sqlite3
import struct
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


def cosine_similarity(vec_a, vec_b, is_normalized=True):
    """
    cosθ = (A . B)/(|A||B|)
    """
    if is_normalized:
        return dot_product(vec_a, vec_b)

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
def getSqliteConnection(db_path="./rag.db", clear_on_start=True):
    conn = sqlite3.connect(db_path)

    if clear_on_start:
        conn.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_text TEXT,
            embedding_blob BLOB
        );
    """)
    conn.commit()
    return conn


# Storing the data in SQL, embeddings ke string e convert korte hobe to store as Text
def storeEmbedding(chunk_list, raw_embeddings, conn, hnsw):
    cursor = conn.cursor()
    for i in range(len(chunk_list)):
        format_string = "<384f"
        embedding_blob = struct.pack(format_string, *raw_embeddings[i])
        cursor.execute(
            "INSERT INTO documents (chunk_text, embedding_blob) VALUES (?, ?)",
            (chunk_list[i], embedding_blob),
        )

        row_id = cursor.lastrowid
        hnsw.insert(row_id, raw_embeddings[i])
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
    cursor = conn.execute("SELECT chunk_text, embedding_blob FROM documents")
    rows = cursor.fetchall()
    format_string = "<384f"

    if not rows:
        return []

    scored_chunks = []
    for row in rows:
        doc = row[0]
        emb = list(struct.unpack(format_string, row[1]))

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
    chunks = createChunks(text, chunk_size=220, overlap=40)

    try:
        print_status("Creating embeddings for: " + file_path)
        embeddings = createEmbeddings(chunks)
        return file_path, chunks, embeddings
    except Exception as e:
        return file_path, [], []


# Walks the directory and processes every pdf concurrently
def ingestDirectory(target_dir, db_conn, graph):
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
                storeEmbedding(chunks, embeddings, db_conn, hnsw)

    hnsw.dump_to_file("hnsw.json")


def getSystemPromptTemplate(context=""):
    with open("prompts/system_prompt.md", "r", encoding="utf-8") as file:
        content = file.read()

    if context:
        content = content.replace("{context}", context)

    return content


def getUserPromptTemplate(context="", query=""):
    with open("prompts/user_prompt.md", "r", encoding="utf-8") as file:
        content = file.read()

    if context:
        content = content.replace("{context}", context)

    if query:
        content = content.replace("{query}", query)

    return content


class ManualHNSW:
    def __init__(self, conn, dim=384, M=16, ef_construction=200, ef_search=128):
        self.dim = dim
        self.conn = conn
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.mL = 1.0 / math.log(M)

        self.nodes = {}
        self.total_nodes = 0

        self.enter_node = None
        self.max_layer = -1

    def _get_random_layer(self):
        r = random.random()
        if r == 0:
            r = 0.0000001

        return int(-math.log(r) * self.mL)

    def get_vector(self, node_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding_blob FROM documents WHERE id = ?", (node_id,))
        row = cursor.fetchone()

        if row and row[0]:
            return list(struct.unpack("<384f", row[0]))
        raise ValueError(f"Node ID {node_id} not found in the database.")

    def get_chunk(self, node_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT chunk_text FROM documents WHERE id = ?", (node_id,))
        row = cursor.fetchone()

        if row and row[0]:
            return row[0]
        raise ValueError(f"Node ID {node_id} not found in the database.")

    def cosine_distance(self, vec_a, node_b_id):
        vec_b = self.get_vector(node_b_id)
        return 1.0 - cosine_similarity(vec_a, vec_b)

    def _search_layer(self, query_vector, enter_node, layer):
        curr_node = enter_node
        curr_dist = self.cosine_distance(query_vector, curr_node)
        while True:
            changed = False
            neighbours = self.nodes.get(curr_node, {}).get(layer, [])
            for neighbour in neighbours:
                neighbour_dist = self.cosine_distance(query_vector, neighbour)
                if neighbour_dist < curr_dist:
                    curr_dist = neighbour_dist
                    curr_node = neighbour
                    changed = True

            if not changed:
                break

        return curr_node

    def _search_layer_ef(self, query_vector, enter_node, layer, ef):
        visited = {enter_node}
        init_dist = self.cosine_distance(query_vector, enter_node)

        v_pool = [(enter_node, init_dist)]
        candidates = [(enter_node, init_dist)]

        while candidates:
            candidates.sort(key=lambda x: x[1])
            curr_node, curr_dist = candidates.pop(0)

            v_pool.sort(key=lambda x: x[1])
            if curr_dist > v_pool[-1][1]:
                break

            neighbors = self.nodes.get(curr_node, {}).get(layer, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)

                    neighbor_dist = self.cosine_distance(query_vector, neighbor)
                    v_pool.sort(key=lambda x: x[1])

                    if neighbor_dist < v_pool[-1][1] or len(v_pool) < ef:
                        candidates.append((neighbor, neighbor_dist))
                        v_pool.append((neighbor, neighbor_dist))

                        if len(v_pool) > ef:
                            v_pool.sort(key=lambda x: x[1])
                            v_pool.pop()

        return [node_id for node_id, _ in v_pool]

    def get_top_k(self, query_vector, neighbor_pool, k):
        scored_neighbors = []
        # base_vector = self.get_vector(node_id)
        for n_id in neighbor_pool:
            dist = self.cosine_distance(query_vector, n_id)
            scored_neighbors.append((n_id, dist))

        scored_neighbors.sort(key=lambda x: x[1])
        return [n_id for n_id, _ in scored_neighbors[:k]]

    def prune_to_max_connection(self, node_id, layer, curr_max_links):
        neighbor_pool = self.nodes[node_id][layer]
        base_vector = self.get_vector(node_id)
        return self.get_top_k(base_vector, neighbor_pool, curr_max_links)

    def insert(self, new_node_id, new_vector):
        self.total_nodes += 1
        # if graph is empty
        if not self.nodes:
            self.enter_node = new_node_id
            self.nodes[new_node_id] = {0: []}
            self.max_layer = 0
            return

        # if graph is not empty
        insert_layer = self._get_random_layer()
        self.nodes[new_node_id] = {}
        for i in range(0, insert_layer + 1):
            self.nodes[new_node_id][i] = []

        curr_obj = self.enter_node

        for l in range(self.max_layer, insert_layer, -1):
            curr_obj = self._search_layer(new_vector, curr_obj, l)

        for l in range(min(self.max_layer, insert_layer), -1, -1):
            candidates = self._search_layer_ef(
                new_vector, curr_obj, l, self.ef_construction
            )
            closest_node = self.get_top_k(new_vector, candidates, 1)[0]
            curr_max_links = self.M0 if l == 0 else self.M
            self.nodes[new_node_id][l].append(closest_node)
            self.nodes[closest_node][l].append(new_node_id)

            if len(self.nodes[closest_node][l]) > curr_max_links:
                self.nodes[closest_node][l] = self.prune_to_max_connection(
                    closest_node, l, curr_max_links
                )
            if len(self.nodes[new_node_id][l]) > curr_max_links:
                self.nodes[new_node_id][l] = self.prune_to_max_connection(
                    new_node_id, l, curr_max_links
                )
            curr_obj = closest_node

        if insert_layer > self.max_layer:
            self.max_layer = insert_layer
            self.enter_node = new_node_id

    def search(self, query, k):
        query_vector = createQueryEmbedding(query)

        if self.total_nodes < 10000:
            print_status(
                "Total nodes are less than, 10000, falling back to bruteforce search"
            )
            return getSimilarity(self.conn, query_vector, k)

        if not self.enter_node or not self.nodes:
            return []

        curr_obj = self.enter_node

        for l in range(self.max_layer, 0, -1):
            curr_obj = self._search_layer(query_vector, curr_obj, l)

        candidates = self._search_layer_ef(query_vector, curr_obj, 0, self.ef_search)

        top_k_ids = self.get_top_k(query_vector, candidates, k)

        return [self.get_chunk(node_id) for node_id in top_k_ids]

    def dump_to_file(self, filepath):
        import json

        export_data = {
            "dim": self.dim,
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "enter_node": self.enter_node,
            "max_layer": self.max_layer,
            "nodes": self.nodes,
            "total_nodes": self.total_nodes,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

    def load_from_file(self, filepath):
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.dim = raw_data["dim"]
        self.M = raw_data["M"]
        self.M0 = raw_data["M0"]
        self.ef_construction = raw_data["ef_construction"]
        self.ef_search = raw_data["ef_search"]
        self.enter_node = raw_data["enter_node"]
        self.max_layer = raw_data["max_layer"]
        self.total_nodes = raw_data["total_nodes"]

        self.nodes = {}
        for str_node_id, layers_dict in raw_data["nodes"].items():
            node_id = int(str_node_id)
            self.nodes[node_id] = {}

            for str_layer_num, neighbors_list in layers_dict.items():
                layer_num = int(str_layer_num)
                self.nodes[node_id][layer_num] = [int(n) for n in neighbors_list]


if __name__ == "__main__":
    # User query
    query = "What is the difference between list and numpy array?"

    ingest = False

    if ingest:
        conn = getSqliteConnection(clear_on_start=True)
        hnsw = ManualHNSW(conn)
        print_status("Starting Ingestion...")
        ingestDirectory("documents", db_conn=conn, graph=hnsw)
        print_status("Ingestion complete...")
    else:
        conn = getSqliteConnection(clear_on_start=False)
        hnsw_graph = ManualHNSW(conn)
        hnsw_graph.load_from_file("hnsw.json")

        print_status("Generating query embeddings...")

        # creating embeddings for the user query
        query_embeddings = createQueryEmbedding(query=query)

        print_status("Started similarity search...")

        # Querying chroma (similarity search)
        # retrieved_chunks = getSimilarity(
        #     conn=conn, query_embeddings=query_embeddings, k=5
        # )

        retrieved_chunks = hnsw_graph.search(query, 5)

        conn.close()

        print_status("Similar chunks retrieved...")

        # Creating context string to pass to llm
        context = "\n---\n".join(retrieved_chunks)

        print("Context:\n\n", context)

        system_prompt = getSystemPromptTemplate()

        user_prompt = getUserPromptTemplate(context=context, query=query)

        print_status("Prompting LLM...")

        # Calling GROQ Api
        response = callLLM(system_prompt=system_prompt, user_prompt=user_prompt)

        clearMsg()
        print("\rResponse:\n", flush=True)
        print(response)
