var rag = new RAGUtil();
var query = "How to set automatic replies in Microsoft Outlook?";

var topChunks = rag.search(query, 3);

var context = topChunks.join("\n---\n");

var systemPrompt = `You are a helpful assistant. Answer the user's question using ONLY the provided text context.
If the answer cannot be found in the context, say 'I cannot find the answer in the document.'
Do not make up information or use outside knowledge.`

var userPrompt = `Context:
${context}
Question: ${query}
Answer:`

gs.print(rag.callLLM(systemPrompt, userPrompt));
