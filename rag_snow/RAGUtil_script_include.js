var RAGUtil = Class.create();
RAGUtil.prototype = Object.extendsObject(AbstractAjaxProcessor, {

    initialize: function() {
        this.dim = gs.getProperty("rag.dim");
        this.M = gs.getProperty("rag.max_conn");
        this.M0 = 2 * this.M;
        this.ef_construct = gs.getProperty("rag.ef_construct");
        this.ef_search = gs.getProperty("rag.ef_search");
        this.enter_node = gs.getProperty("rag.enter_node");
        this.max_layer = gs.getProperty("rag.max_layer");
        this.mL = 1 / Math.log(this.M);
        this.nodes = {};

    },

    dotProduct: function(vecA, vecB) {
        let totalSum = 0;
        let vecLen = Math.min(vecA.length, vecB.length);
        for (i = 0; i < vecLen; i++) {
            totalSum += vecA[i] * vecB[i];
        }
        return totalSum;
    },

    magnitude: function(vec) {
        return Math.sqrt(vec.reduce((sum, x) => sum + x ** 2, 0));
    },

    cosineSimilarity: function(vecA, vecB, isNormalized = true) {
        if (isNormalized) return this.dotProduct(vecA, vecB);

        let magA = this.magnitude(vecA);
        let magB = this.magnitude(vecB);

        if (magA == 0 || magB == 0) return 0.0;
        return this.dotProduct(vecA, vecB) / (magA * magB);
    },

    createChunks: function(text, chunkSize, overlap) {
        if (!text) return [];

        var words = text.match(/\b[\w'-]+\b/g) || [];
        var textLen = words.length;
        var start = 0;
        var chunkList = [];

        while (start < textLen) {
            var end = Math.min(start + chunkSize, textLen);

            var chunk = words.slice(start, end).join(' ');
            chunkList.push(chunk);

            start = start + chunkSize - overlap;

            if (chunkSize <= overlap) {
                break;
            }
        }

        return chunkList;
    },

    createEmbeddings: function(chunkList) {
        var provider = new sn_cc.StandardCredentialsProvider();
        var credential = provider.getCredentialByAliasID("fa0f862d93950710c987fc532bba1010");

        var token = credential.getAttribute('api_key');

        if (!token) {
            throw new Error("Unable to retrieve Hugging Face token from connection alias 'HF_API_TOKEN'.");
        }

        var endpoint = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction";

        var request = new sn_ws.RESTMessageV2();
        request.setEndpoint(endpoint);
        request.setHttpMethod('POST');

        request.setRequestHeader('Authorization', 'Bearer ' + token);
        request.setRequestHeader('Content-Type', 'application/json');

        var payload = {
            "inputs": {
                "sentences": chunkList
            },
            "options": {
                "wait_for_model": true
            }
        };
        request.setRequestBody(JSON.stringify(payload));

        var response = request.execute();
        var statusCode = response.getStatusCode();
        var responseBody = response.getBody();

        token = null;

        if (statusCode !== 200) {
            throw new Error("HF API Error: " + statusCode + " - " + responseBody);
        }

        return JSON.parse(responseBody);

    },

    createQueryEmbedding: function(query) {
        var embeddings = this.createEmbeddings([query]);

        if (embeddings && embeddings.length > 0) {
            return embeddings[0];
        }

        return null;
    },

    storeEmbeddings: function(chunkList, rawEmbeddings) {
        let arrLen = Math.min(chunkList.length, rawEmbeddings.length);
        var chunkGr = new GlideRecord('u_embeddings');
        for (i = 0; i < arrLen; i++) {
            chunkGr.initialize();
            chunkGr.u_chunk = chunkList[i];
            chunkGr.u_embedding = JSON.stringify(rawEmbeddings[i]);
            chunkGr.insert();
        }
    },

    callLLM: function(systemPrompt, userPrompt) {
        var provider = new sn_cc.StandardCredentialsProvider();
        var credential = provider.getCredentialByAliasID("d90e657193958710c987fc532bba1076");

        var token = credential.getAttribute('api_key');

        if (!token) {
            throw new Error("Unable to retrieve Groq token from connection alias.");
        }

        var apiUrl = "https://api.groq.com/openai/v1/chat/completions";

        var request = new sn_ws.RESTMessageV2();
        request.setEndpoint(apiUrl);
        request.setHttpMethod('POST');

        request.setRequestHeader('Authorization', 'Bearer ' + token);
        request.setRequestHeader('Content-Type', 'application/json');

        var payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [{
                    "role": "system",
                    "content": systemPrompt
                },
                {
                    "role": "user",
                    "content": userPrompt
                }
            ]
        };
        request.setRequestBody(JSON.stringify(payload));

        var response = request.execute();
        var statusCode = response.getStatusCode();
        var responseBody = response.getBody();

        token = null;

        if (statusCode !== 200) {
            throw new Error("GROQ API Error: " + statusCode + " - " + responseBody);
        }

        var jsonResponse = JSON.parse(responseBody);
        if (jsonResponse && jsonResponse.choices && jsonResponse.choices.length > 0) {
            return jsonResponse.choices[0].message.content;
        }

        throw new Error("Unexpected empty choices layout returned from Groq API response.");

    },

    getSimilarity: function(queryEmbedding, k) {
        var scoredChunks = [];

        var grDoc = new GlideRecord("u_embeddings");

        grDoc.query();

        while (grDoc.next()) {
            var docText = grDoc.getValue('u_chunk');
            var embBlob = grDoc.getValue('u_embedding');

            if (!docText || !embBlob) {
                continue;
            }

            var embArray;
            try {
                embArray = JSON.parse(embBlob);
            } catch (e) {
                embArray = embBlob.split(',').map(Number);
            }

            var score = this.cosineSimilarity(queryEmbeddings, embArray);

            scoredChunks.push({
                document: docText,
                score: score
            });
        }

        if (scoredChunks.length === 0) {
            return [];
        }

        scoredChunks.sort(function(a, b) {
            return b.score - a.score;
        });

        var topChunks = [];
        var limit = Math.min(k, scoredChunks.length);
        for (var i = 0; i < limit; i++) {
            topChunks.push(scoredChunks[i].document);
        }

        return topChunks;
    },


    type: 'RAGUtil'
});
