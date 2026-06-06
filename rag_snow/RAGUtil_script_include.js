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

    getRandomLayer: function() {
        let r = Math.random();
        if (r == 0) r = 0.0000001;
        return Math.floor(-Math.log(r) * this.mL);
    },

    cosineSimilarity: function(vecA, vecB, isNormalized = true) {
        if (isNormalized) return this.dotProduct(vecA, vecB);

        let magA = this.magnitude(vecA);
        let magB = this.magnitude(vecB);

        if (magA == 0 || magB == 0) return 0.0;
        return this.dotProduct(vecA, vecB) / (magA * magB);
    },

    cosineDistance: function(vec_a, node_b_id) {
        var nodeGr = new GlideRecord("u_embeddings");
        nodeGr.get(node_b_id);
        var vec_b = JSON.parse(nodeGr.u_embedding);
        return 1.0 - this.cosineSimilarity(vec_a, vec_b);
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
        var arrLen = Math.min(chunkList.length, rawEmbeddings.length);
        var savedNodeIds = [];

        // Phase 1: Write all embeddings safely to the database
        for (var i = 0; i < arrLen; i++) {
            var chunkGr = new GlideRecord('u_embeddings');
            chunkGr.initialize();
            chunkGr.u_chunk = chunkList[i];
            chunkGr.u_embedding = JSON.stringify(rawEmbeddings[i]);

            var new_id = chunkGr.insert();
            savedNodeIds.push({
                id: new_id,
                vector: rawEmbeddings[i]
            });
        }

        // Phase 2: Now build the HNSW graph safely with existing DB records
        for (var j = 0; j < savedNodeIds.length; j++) {
            this.insert(savedNodeIds[j].id, savedNodeIds[j].vector);
        }
        gs.setProperty("rag.enter_node", this.enter_node);
        gs.setProperty("rag.max_layer", this.max_layer);
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

    searchLayer: function(query_vector, enter_node, layer) {
        var curr_node = enter_node;
        var curr_dist = this.cosineDistance(query_vector, curr_node);

        while (true) {
            var changed = false;

            var neighbourGr = new GlideRecord("u_hnsw_relations");
            neighbourGr.addEncodedQuery("u_base_node=" + curr_node.sys_id.toString() + "^u_layer=" + layer);
            neighbourGr.query();

            while (neighbourGr.next()) {
                // let neighbour_dist = this.cosineDistance(query_vector, neighbourGr.sys_id.toString());
                let neighbour_dist = this.cosineDistance(query_vector, neighbourGr.getValue("u_neighbour_node"));

                if (neighbour_dist < curr_dist) {
                    curr_dist = neighbour_dist;
                    // curr_node = neighbourGr.sys_id.toString();
                    curr_node = neighbourGr.getValue("u_neighbour_node");
                    changed = true;
                }
            }

            if (!changed) {
                break;
            }
        }

        return curr_node;
    },

    searchLayerEf: function(query_vector, enter_node, layer, ef) {
        var enter_node_id = enter_node.toString();

        var visited = {};
        visited[enter_node_id] = true;

        var init_dist = this.cosineDistance(query_vector, enter_node_id);

        var v_pool = [{
            id: enter_node_id,
            dist: init_dist
        }];
        var candidates = [{
            id: enter_node_id,
            dist: init_dist
        }];

        while (candidates.length > 0) {
            candidates.sort(function(a, b) {
                return a.dist - b.dist;
            });

            var curr = candidates.shift();
            var curr_node_id = curr.id;
            var curr_dist = curr.dist;

            if (curr_dist > v_pool[v_pool.length - 1].dist) {
                break;
            }

            var neighbourGr = new GlideRecord("u_hnsw_relations");
            neighbourGr.addQuery("u_base_node", curr_node_id);
            neighbourGr.addQuery("u_layer", layer);
            neighbourGr.query();

            while (neighbourGr.next()) {
                // var neighbour_id = neighbourGr.u_neighbour.toString();
                var neighbour_id = neighbourGr.getValue("u_neighbour_node");

                if (!neighbour_id) {
                    gs.warn("Skipping null neighbour for base_node=" + curr_node_id + " layer=" + layer);
                    continue;
                }

                if (!visited[neighbour_id]) {
                    visited[neighbour_id] = true;

                    var neighbour_dist = this.cosineDistance(query_vector, neighbour_id);
                    var far_dist = v_pool[v_pool.length - 1].dist;

                    if (neighbour_dist < far_dist || v_pool.length < ef) {
                        var newCandidate = {
                            id: neighbour_id,
                            dist: neighbour_dist
                        };

                        candidates.push(newCandidate);
                        v_pool.push(newCandidate);

                        v_pool.sort(function(a, b) {
                            return a.dist - b.dist;
                        });

                        if (v_pool.length > ef) {
                            v_pool.pop();
                        }
                    }
                }
            }
        }

        var resultNodes = [];
        for (var i = 0; i < v_pool.length; i++) {
            resultNodes.push(v_pool[i].id);
        }

        return resultNodes;
    },

    insert: function(new_node_id, new_vector) {
        new_node_id = new_node_id.toString();

        if (!this.enter_node) {
            this.enter_node = new_node_id;
            this.max_layer = 0;

            this._createNodePlaceholder(new_node_id);
            return;
        }

        var insert_layer = this.getRandomLayer();

        var curr_obj = this.enter_node.toString();

        for (var l = this.max_layer; l > insert_layer; l--) {
            curr_obj = this.searchLayer(new_vector, curr_obj, l);
        }

        var start_layer = Math.min(this.max_layer, insert_layer);
        for (var l = start_layer; l >= 0; l--) {

            var candidates = this.searchLayerEf(new_vector, curr_obj, l, this.ef_construct);
            var curr_max_links = (l === 0) ? this.M0 : this.M;

            var top_neighbors = this.getTopK(new_vector, candidates, curr_max_links);
            if (top_neighbors.length > 0) {
                curr_obj = top_neighbors[0];
            }

            for (var i = 0; i < top_neighbors.length; i++) {
                var closest_node = top_neighbors[i].toString();

                this._addRelation(new_node_id, closest_node, l);
                this._addRelation(closest_node, new_node_id, l);

                if (this._getLinkCount(closest_node, l) > curr_max_links) {
                    this.pruneToMaxConnection(closest_node, l, curr_max_links);
                }
            }

            if (this._getLinkCount(new_node_id, l) > curr_max_links) {
                this.pruneToMaxConnection(new_node_id, l, curr_max_links);
            }
        }

        // If the new node climbed higher than the previous maximum layer, promote it
        if (insert_layer > this.max_layer) {
            this.max_layer = insert_layer;
            this.enter_node = new_node_id;
        }
    },

    getTopK: function(query_vector, neighbor_pool, k) {
        var scored_neighbors = [];

        for (var i = 0; i < neighbor_pool.length; i++) {
            var n_id = neighbor_pool[i].toString();

            var dist = this.cosineDistance(query_vector, n_id);

            scored_neighbors.push({
                id: n_id,
                dist: dist
            });
        }

        scored_neighbors.sort(function(a, b) {
            return a.dist - b.dist;
        });

        var topKNodes = [];
        var limit = Math.min(scored_neighbors.length, k);

        for (var j = 0; j < limit; j++) {
            topKNodes.push(scored_neighbors[j].id.toString());
        }

        return topKNodes;
    },

    pruneToMaxConnection: function(node_id, layer, curr_max_links) {
        node_id = node_id.toString();

        var base_vector = null;
        var embedGr = new GlideRecord("u_embeddings");
        if (embedGr.get(node_id)) {
            base_vector = JSON.parse(embedGr.u_embedding);
        } else {
            return;
        }

        var neighbor_pool = [];
        var neighborGr = new GlideRecord("u_hnsw_relations");
        neighborGr.addQuery("u_base_node", node_id);
        neighborGr.addQuery("u_layer", layer);
        neighborGr.query();

        while (neighborGr.next()) {
            neighbor_pool.push(neighborGr.getValue("u_neighbour_node"));
        }

        var keep_neighbors = this.getTopK(base_vector, neighbor_pool, curr_max_links);

        var pruneGr = new GlideRecord("u_hnsw_relations");
        pruneGr.addQuery("u_base_node", node_id);
        pruneGr.addQuery("u_layer", layer);
        pruneGr.addQuery("u_neighbour_node", "NOT IN", keep_neighbors.join(','));
        pruneGr.query();

        while (pruneGr.next()) {
            pruneGr.deleteRecord();
        }
    },

    // --- Private Database Helper Utilities ---

    _addRelation: function(base_node, neighbour_node, layer) {
        var relGr = new GlideRecord("u_hnsw_relations");
        relGr.initialize();
        relGr.setValue("u_base_node", base_node);
        relGr.setValue("u_neighbour_node", neighbour_node);
        relGr.setValue("u_layer", layer);
        relGr.insert();
    },

    _getLinkCount: function(node_id, layer) {
        var countGa = new GlideAggregate("u_hnsw_relations");
        countGa.addQuery("u_base_node", node_id); // addQuery handles reference fields correctly
        countGa.addQuery("u_layer", layer);
        countGa.addAggregate("COUNT");
        countGa.query();
        if (countGa.next()) {
            return parseInt(countGa.getAggregate("COUNT"), 10);
        }
        return 0;
    },

    _createNodePlaceholder: function(node_id) {
        var embedGr = new GlideRecord("u_embeddings");
        if (!embedGr.get(node_id)) {
            embedGr.initialize();
            embedGr.setNewGuidValue(node_id);
            embedGr.insert();
        }
    },

    search: function(query, k) {
        var query_vector = this.createQueryEmbedding(query);

        // var totalNodes = this._getTotalNodeCount();
        // if (totalNodes < 1000) {
        //     gs.info("[!] Total nodes are less than 1000, falling back to bruteforce search");
        //     return this.getSimilarity(query_vector, k);
        // }

        if (!this.enter_node) {
            return [];
        }

        var curr_obj = this.enter_node.toString();
        var current_max_layer = parseInt(this.max_layer, 10) || 0;

        for (var l = current_max_layer; l > 0; l--) {
            curr_obj = this.searchLayer(query_vector, curr_obj, l);
        }

        var efSearchLimit = parseInt(this.ef_search, 10) || 16;
        var candidates = this.searchLayerEf(query_vector, curr_obj, 0, efSearchLimit);

        var top_k_ids = this.getTopK(query_vector, candidates, k);

        var finalChunks = [];
        for (var i = 0; i < top_k_ids.length; i++) {
            var chunkText = this.getChunk(top_k_ids[i]);
            if (chunkText) {
                finalChunks.push(chunkText);
            }
        }

        return finalChunks;
    },

    _getTotalNodeCount: function() {
        var countGa = new GlideAggregate("u_embeddings");
        countGa.addAggregate("COUNT");
        countGa.query();
        if (countGa.next()) {
            return parseInt(countGa.getAggregate("COUNT"), 10);
        }
        return 0;
    },

    getChunk: function(node_id) {
        var embedGr = new GlideRecord("u_embeddings");
        if (embedGr.get(node_id.toString())) {
            return embedGr.getValue("u_chunk");
        }
        return null;
    },


    type: 'RAGUtil'
});
