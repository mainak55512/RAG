package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/joho/godotenv"
	"github.com/ledongthuc/pdf"

	_ "modernc.org/sqlite"
)

type HFRequest struct {
	Inputs  HFInputs  `json:"inputs"`
	Options HFOptions `json:"options"`
}

type HFInputs struct {
	Sentences []string `json:"sentences"`
}

type HFOptions struct {
	WaitForModel bool `json:"wait_for_model"`
}

type ScoredChunk struct {
	Document string
	Score    float32
}

type GroqRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionResponse struct {
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message ChatMessage `json:"message"`
}

type ChatMessage struct {
	Content string `json:"content"`
}

type Result struct {
	FilePath   string
	Chunk      []string
	Embeddings [][]float32
}

func clearMsg() {
	fmt.Fprintf(os.Stdout, "\r%-100s\r", " ")
	os.Stdout.Sync()
}

func printStatus(message string) {
	fmt.Fprintf(os.Stdout, "\r%-100s", message)
	os.Stdout.Sync()
}

func dotProduct(vecA, vecB []float32) float32 {
	var totalSum float32 = 0.0
	for i := 0; i < len(vecA); i++ {
		totalSum += vecA[i] * vecB[i]
	}
	return totalSum
}

func magnitude(vec []float32) float32 {
	var mag float32 = 0.0
	for _, x := range vec {
		mag += x * x
	}
	return float32(math.Sqrt(float64(mag)))
}

func cosineSimilarity(vecA, vecB []float32) float32 {
	magA := magnitude(vecA)
	magB := magnitude(vecB)
	if magA == 0.0 || magB == 0.0 {
		return 0.0
	}
	return dotProduct(vecA, vecB) / (magA * magB)
}

func readPDFFile(filePath string) (string, error) {
	f, r, err := pdf.Open(filePath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var buf bytes.Buffer
	b, err := r.GetPlainText()
	if err != nil {
		return "", err
	}
	buf.ReadFrom(b)
	content := buf.String()
	return content, nil
}

func createChunks(pattern *regexp.Regexp, text string, chunkSize, overlap int) []string {
	words := pattern.FindAllString(text, -1)
	textLen := len(words)
	start := 0
	var chunkList []string
	for start < textLen {
		end := min(start+chunkSize, textLen)
		chunk := strings.Join(words[start:end], " ")
		chunkList = append(chunkList, chunk)
		start = start + chunkSize - overlap
	}
	return chunkList
}

func createEmbeddings(chunkList []string) ([][]float32, error) {
	apiURL := "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

	hfToken := os.Getenv("HF_TOKEN")
	if hfToken == "" {
		return nil, fmt.Errorf("HF_TOKEN environment variable is not set")
	}

	payload := HFRequest{
		Inputs: HFInputs{
			Sentences: chunkList,
		},
		Options: HFOptions{
			WaitForModel: true,
		},
	}

	jsonBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+hfToken)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HF API Error: %d - %s", resp.StatusCode, string(bodyBytes))
	}

	var embeddings [][]float32
	if err := json.NewDecoder(resp.Body).Decode(&embeddings); err != nil {
		return nil, fmt.Errorf("failed to decode response JSON: %w", err)
	}

	return embeddings, nil
}

func createQueryEmbedding(query string) []float32 {
	queryEmbeding, err := createEmbeddings([]string{query})
	if err != nil {
		panic(err)
	}
	return queryEmbeding[0]
}

func getSQLiteConnection(dbPath string, clearOnStart bool) (*sql.DB, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if clearOnStart {
		_, err = db.Exec("DROP TABLE IF EXISTS documents;")
		if err != nil {
			db.Close()
			return nil, fmt.Errorf("failed to drop table: %w", err)
		}
	}

	query := `
	CREATE TABLE IF NOT EXISTS documents (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		chunk_text TEXT,
		embedding_json TEXT
	);`

	_, err = db.Exec(query)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create table: %w", err)
	}

	return db, nil
}

func storeSQLite(db *sql.DB, chunkList []string, rawEmbeddings [][]float32) error {
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("failed to start transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare("INSERT INTO documents (chunk_text, embedding_json) VALUES (?, ?)")
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for i := 0; i < len(chunkList); i++ {
		embeddingBytes, err := json.Marshal(rawEmbeddings[i])
		if err != nil {
			return fmt.Errorf("failed to marshal embedding at index %d: %w", i, err)
		}

		_, err = stmt.Exec(chunkList[i], string(embeddingBytes))
		if err != nil {
			return fmt.Errorf("failed to execute insert at index %d: %w", i, err)
		}
	}

	return tx.Commit()
}

func getSimilarity(db *sql.DB, queryEmbedding []float32, k int) []string {
	rows, err := db.Query("SELECT chunk_text, embedding_json FROM documents")
	if err != nil {
		panic(err)
	}
	defer rows.Close()
	var scoredChunks []ScoredChunk

	for rows.Next() {
		var chunkText, embeddingJson string
		err := rows.Scan(&chunkText, &embeddingJson)
		if err != nil {
			panic(err)
		}

		var emb []float32
		if err := json.Unmarshal([]byte(embeddingJson), &emb); err != nil {
			panic(err)
		}
		score := cosineSimilarity(queryEmbedding, emb)
		scoredChunks = append(scoredChunks, ScoredChunk{
			Document: chunkText,
			Score:    score,
		})
	}

	sort.Slice(scoredChunks, func(i, j int) bool {
		return scoredChunks[i].Score > scoredChunks[j].Score
	})

	if k > len(scoredChunks) {
		k = len(scoredChunks)
	}

	topChunks := make([]string, k)
	for i := 0; i < k; i++ {
		topChunks[i] = scoredChunks[i].Document
	}

	return topChunks

}

func processPDF(filePath string, pattern *regexp.Regexp) (string, []string, [][]float32) {
	printStatus("Reading file: " + filePath)
	text, _ := readPDFFile(filePath)
	if len(text) == 0 {
		return filePath, nil, nil
	}
	printStatus("Chunking document: " + filePath)
	chunks := createChunks(pattern, text, 50, 10)
	printStatus("Creating embeddings for: " + filePath)
	embeddings, _ := createEmbeddings(chunks)
	return filePath, chunks, embeddings
}

func ingestDirectory(targetDir string, conn *sql.DB) {
	files, err := os.ReadDir(targetDir)
	if err != nil {
		log.Printf("Error reading directory: %v\n", err)
		return
	}

	var pdfPaths []string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(strings.ToLower(file.Name()), ".pdf") {
			pdfPaths = append(pdfPaths, filepath.Join(targetDir, file.Name()))
		}
	}

	if len(pdfPaths) == 0 {
		fmt.Println("No PDF files found in the directory.")
		return
	}

	numWorkers := runtime.NumCPU()
	pathsChan := make(chan string, len(pdfPaths))
	resultsChan := make(chan Result, len(pdfPaths))

	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			re := regexp.MustCompile(`\b[\w'-]+\b`)
			for path := range pathsChan {
				filePath, chunks, embeddings := processPDF(path, re)
				resultsChan <- Result{filePath, chunks, embeddings}
			}
		}()
	}

	for _, path := range pdfPaths {
		pathsChan <- path
	}
	close(pathsChan)

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	for result := range resultsChan {
		if len(result.Chunk) > 0 && len(result.Embeddings) > 0 {
			storeSQLite(conn, result.Chunk, result.Embeddings)
		}
	}

}

func apiCall(systemPrompt, userPrompt string) (string, error) {
	apiURL := "https://api.groq.com/openai/v1/chat/completions"
	groqToken := os.Getenv("GROQ_API_KEY")
	if groqToken == "" {
		panic("Groq token unavailable")
	}

	payload := GroqRequest{
		Model: "openai/gpt-oss-120b",
		Messages: []Message{
			{
				Role:    "system",
				Content: systemPrompt,
			},
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
	}
	jsonBytes, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonBytes))
	if err != nil {
		panic(err)
	}
	req.Header.Set("Authorization", "Bearer "+groqToken)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HF API Error: %d - %s", resp.StatusCode, string(bodyBytes))
	}

	var respBody ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&respBody); err != nil {
		return "", fmt.Errorf("failed to decode response JSON: %w", err)
	}

	return respBody.Choices[0].Message.Content, nil
}

func main() {
	godotenv.Load()
	conn, _ := getSQLiteConnection("./rag.db", true)

	printStatus("Starting Ingestion...")
	ingestDirectory("../documents", conn)

	userQuery := "How many types of loops are present in Python?"

	printStatus("Generating query embeddings...")
	queryEmbedding := createQueryEmbedding(userQuery)

	printStatus("Started similarity search...")
	retrievedChunks := getSimilarity(conn, queryEmbedding, 5)

	printStatus("Generating context...")
	context := strings.Join(retrievedChunks, "\n---\n")

	systemPrompt := `You are a helpful assistant. Answer the user's question using ONLY the provided text context. 
If the answer cannot be found in the context, say 'I cannot find the answer in the document.' 
Do not make up information or use outside knowledge.`

	userPrompt := fmt.Sprintf(`Context:
%s
Question: %s
Answer:`, context, userQuery)

	printStatus("Prompting LLM...")

	resp, _ := apiCall(systemPrompt, userPrompt)

	clearMsg()
	fmt.Printf("\rResponse:\n\n")
	fmt.Println(resp)

}
