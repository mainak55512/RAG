gs.setProperty("rag.enter_node", "");
gs.setProperty("rag.max_layer", 0);

var embGr = new GlideRecord("u_embeddings");
var relGr = new GlideRecord("u_hnsw_relations");
embGr.deleteMultiple();
relGr.deleteMultiple();


var kbGr = new GlideRecord("kb_knowledge");
kbGr.addEncodedQuery("workflow_state=published");
kbGr.query();

while (kbGr.next()) {
    try {
        var articleNumber = kbGr.number.toString();
        gs.info("HNSW Ingestion: Attempting processing for " + articleNumber);
        
        var text = kbGr.getDisplayValue('text');

        if (!text || text.trim() === "") {
            gs.warn("HNSW Ingestion: Skipping article " + articleNumber + " because the text field is empty.");
            continue; 
        }

        var rag = new RAGUtil(); 
        
        var chunkList = rag.createChunks(text, 200, 20);

        if (!chunkList || chunkList.length === 0) {
            gs.warn("HNSW Ingestion: Chunk generation returned 0 blocks for article " + articleNumber);
            continue;
        }

        var rawEmbeddings = rag.createEmbeddings(chunkList);

        if (!rawEmbeddings || rawEmbeddings.length === 0) {
            gs.warn("HNSW Ingestion: Embedding generation returned 0 vectors for article " + articleNumber);
            continue;
        }

        rag.storeEmbeddings(chunkList, rawEmbeddings);
        
        gs.info("HNSW Ingestion: Successfully processed and committed " + articleNumber);

    } catch (articleError) {
        gs.error("HNSW Ingestion FATAL EXCEPTION on article " + kbGr.number.toString() + ". Error Details: " + articleError.toString());
    }
}

gs.info("HNSW Ingestion: Completed successfully!")
