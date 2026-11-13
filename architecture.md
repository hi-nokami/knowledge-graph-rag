# Project 1: Knowledge Graph RAG — System Architecture Specification

## 1. System Overview

Knowledge Graph RAG (KG-RAG) is a hybrid retrieval-augmented generation system engineered to resolve the core failure mode of standard vector RAG: **multi-hop relational reasoning across dispersed documents**. 

Standard vector search encodes semantic similarity, performing reliably on single-hop factual lookups but degrading significantly on questions requiring 2-hop, 3-hop, or aggregative reasoning across multiple entities. KG-RAG pairs a dense vector database (PostgreSQL with `pgvector`) with a native property graph database (`Neo4j`), bridging both stores via a shared chunk identifier (`chunk_id`). A lightweight semantic query router classifies questions into Vector, Graph, or Hybrid retrieval paths, followed by Reciprocal Rank Fusion (RRF), context formatting with labeled evidence, local LLM generation, and automated citation validation.

All inference runs locally via Ollama (`http://localhost:11434/v1`) optimized for an 8GB VRAM constraint (NVIDIA GeForce RTX 5060 Laptop GPU).

```
                             +------------------------+
                             |     User Question      |
                             +-----------+------------+
                                         |
                                         v
                             +------------------------+
                             |   Fast Query Router    |
                             |     (qwen2.5:1.5b)     |
                             +-----------+------------+
                                         |
                   +---------------------+---------------------+
                   | (VECTOR / BOTH)                           | (GRAPH / BOTH)
                   v                                           v
       +-----------------------+                   +-----------------------+
       |   Vector Retriever    |                   |   Graph Retriever     |
       |  (pgvector + HNSW)    |                   |   (Neo4j + Cypher)    |
       |  `nomic-embed-text`   |                   |  Parameterized Query  |
       +-----------+-----------+                   +-----------+-----------+
                   |                                           |
                   | Top-K Chunks                              | Paths + Backing Chunks
                   | (chunk_id, text, score)                   | (statements, chunk_ids)
                   +---------------------+---------------------+
                                         |
                                         v
                             +------------------------+
                             | Hybrid Fusion & Rerank |
                             | (Reciprocal Rank Fusion|
                             |       k = 60)          |
                             +-----------+------------+
                                         |
                                         v
                             +------------------------+
                             | Context Assembly       |
                             | - Labeled Graph Facts  |
                             | - Labeled Text Chunks  |
                             +-----------+------------+
                                         |
                                         v
                             +------------------------+
                             | Answer Generation      |
                             |      (qwen3:8b)        |
                             | With Claim Citations   |
                             +-----------+------------+
                                         |
                                         v
                             +------------------------+
                             | Citation Validator     |
                             | (Post-Generation Check)|
                             +-----------+------------+
                                         |
                                         v
                             +------------------------+
                             | Grounded Answer Output |
                             +------------------------+
```

---

## 2. Schema & Ontology Specification

### 2.1 Neo4j Property Graph Ontology

An open-ended extraction creates an unqueryable graph. We enforce a constrained corporate & financial domain ontology tailored for multi-hop relationship reasoning and benchmark queries (e.g., corporate structures, supply chains, investments, officers, and financial metrics).

#### Node Labels & Properties

1. **`Company`**
   - `id`: `STRING` (UUID or slug, e.g. `comp_acme_corp`)
   - `name`: `STRING` (raw observed name)
   - `canonical_name`: `STRING` (normalized primary key, e.g. `acme corp`)
   - `aliases`: `LIST<STRING>` (e.g. `["ACME", "Acme Corporation", "Acme Inc."]`)
   - `sector`: `STRING` (e.g. `Energy`, `Technology`, `Healthcare`)
   - `created_at`: `DATETIME`

2. **`Person`**
   - `id`: `STRING` (UUID or slug, e.g. `person_jane_doe`)
   - `name`: `STRING` (raw observed name)
   - `canonical_name`: `STRING` (normalized primary key)
   - `aliases`: `LIST<STRING>`
   - `role`: `STRING` (e.g. `CEO`, `Director`, `Founder`)

3. **`Product`**
   - `id`: `STRING`
   - `name`: `STRING`
   - `canonical_name`: `STRING`
   - `category`: `STRING`

4. **`Location`**
   - `id`: `STRING`
   - `name`: `STRING`
   - `canonical_name`: `STRING`
   - `type`: `STRING` (e.g. `City`, `Country`, `Headquarters`)

5. **`FinancialMetric`**
   - `id`: `STRING`
   - `metric_type`: `STRING` (e.g. `Revenue`, `NetIncome`, `FundingRound`)
   - `amount`: `FLOAT`
   - `currency`: `STRING` (e.g. `USD`)
   - `period`: `STRING` (e.g. `Q3`, `Annual`)
   - `year`: `INTEGER`

6. **`DocumentChunk`**
   - `chunk_id`: `STRING` (Primary key, e.g. `doc_sec_2023_chunk_0042`)
   - `doc_id`: `STRING`
   - `section`: `STRING`
   - `content_hash`: `STRING`

#### Relationship Types (Directed Edges)

Every relationship edge carries `source_chunk_id` and `confidence` to allow bi-directional navigation between structural edges and backing textual evidence.

| Relationship | Source Label | Target Label | Edge Properties |
|---|---|---|---|
| `SUBSIDIARY_OF` | `Company` | `Company` | `since: INT, source_chunk_id: STRING, confidence: FLOAT` |
| `SUPPLIES` | `Company` | `Company` | `product: STRING, contract_value: FLOAT, source_chunk_id: STRING, confidence: FLOAT` |
| `ACQUIRED` | `Company` | `Company` | `year: INT, amount: FLOAT, source_chunk_id: STRING, confidence: FLOAT` |
| `INVESTED_IN` | `Company` \| `Person` | `Company` | `round: STRING, amount: FLOAT, source_chunk_id: STRING, confidence: FLOAT` |
| `EMPLOYS` | `Company` | `Person` | `title: STRING, since: INT, source_chunk_id: STRING, confidence: FLOAT` |
| `BOARD_MEMBER_OF` | `Person` | `Company` | `since: INT, source_chunk_id: STRING, confidence: FLOAT` |
| `PRODUCES` | `Company` | `Product` | `source_chunk_id: STRING, confidence: FLOAT` |
| `LOCATED_IN` | `Company` \| `Person` | `Location` | `type: STRING, source_chunk_id: STRING, confidence: FLOAT` |
| `REPORTED` | `Company` | `FinancialMetric` | `source_chunk_id: STRING, confidence: FLOAT` |
| `MENTIONED_IN` | `Company` \| `Person` \| `Product` | `DocumentChunk` | `count: INT, source_chunk_id: STRING` |

#### Constraints and Indexes in Neo4j

```cypher
// Uniqueness constraints
CREATE CONSTRAINT unique_company_canonical IF NOT EXISTS FOR (c:Company) REQUIRE c.canonical_name IS UNIQUE;
CREATE CONSTRAINT unique_person_canonical IF NOT EXISTS FOR (p:Person) REQUIRE p.canonical_name IS UNIQUE;
CREATE CONSTRAINT unique_product_canonical IF NOT EXISTS FOR (pr:Product) REQUIRE pr.canonical_name IS UNIQUE;
CREATE CONSTRAINT unique_location_canonical IF NOT EXISTS FOR (l:Location) REQUIRE l.canonical_name IS UNIQUE;
CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (dc:DocumentChunk) REQUIRE dc.chunk_id IS UNIQUE;

// Search indexes
CREATE INDEX idx_company_name IF NOT EXISTS FOR (c:Company) ON (c.name);
CREATE INDEX idx_person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX idx_edge_chunk_id IF NOT EXISTS FOR ()-[r:SUBSIDIARY_OF]-() ON (r.source_chunk_id);
```

---

### 2.2 PostgreSQL + pgvector Schema

PostgreSQL 16 with the `pgvector` extension hosts chunk text, metadata, and 768-dimensional dense vector embeddings generated by `nomic-embed-text`.

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: document_chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(128) UNIQUE NOT NULL,
    doc_id VARCHAR(128) NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    embedding vector(768) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Relational entity index for fast cross-referencing
CREATE TABLE IF NOT EXISTS entity_mentions (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(128) NOT NULL REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
    entity_canonical_name VARCHAR(256) NOT NULL,
    entity_type VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_metadata ON document_chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_canonical ON entity_mentions(entity_canonical_name);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk_id ON entity_mentions(chunk_id);

-- Approximate Nearest Neighbor (ANN) Index
-- HNSW with cosine distance operator
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
```

#### Vector Search Configuration
- **Distance Metric**: Cosine Distance (`<=>`). Cosine similarity is computed as `1 - (embedding <=> query_vector)`.
- **Index Type**: Hierarchical Navigable Small World (`HNSW`). Chosen over `IVFFlat` for superior recall without requiring periodic index retraining on dynamic document additions.
- **Query Parameter Tuning**: `SET hnsw.ef_search = 40;` (balanced latency/recall tradeoff).

---

## 3. Ingestion Pipeline & Shared Chunk Architecture

```
  Raw Documents (PDF / MD / TXT)
               │
               ▼
  ┌─────────────────────────┐
  │ Recursive Text Splitter │ (Chunk size: 1000 chars, Overlap: 200 chars)
  └────────────┬────────────┘
               │ Assign unique chunk_id (e.g. doc01_chunk0001)
               ▼
  ┌─────────────────────────┐
  │ Chunk Record Created    │
  └───────┬─────────┬───────┘
          │         │
          ▼         ▼
┌───────────────┐ ┌──────────────────────────────────────┐
│  pgvector     │ │  LLM Structured Entity Extraction    │
│  Embed with   │ │  (llama3.1:8b or qwen3:8b via Ollama)│
│nomic-embed-txt│ └──────────────────┬───────────────────┘
│  (768 dim)    │                    │
└───────┬───────┘                    ▼
        │         ┌──────────────────────────────────────┐
        │         │  Entity Resolution & Normalization   │
        │         │  (Strip suffixes, alias resolution)  │
        │         └──────────────────┬───────────────────┘
        │                            │
        │                            ▼
        │         ┌──────────────────────────────────────┐
        │         │  Idempotent Neo4j MERGE Ingestion    │
        │         │  (Nodes + Edges with source_chunk_id)│
        │         └──────────────────┬───────────────────┘
        │                            │
        └─────────────┬──────────────┘
                      ▼
         Synchronized Dual-Store Ready
```

### 3.1 The Shared Chunk ID Mechanism
Every chunk created during document parsing is assigned a deterministic `chunk_id` formatted as `{doc_id}_ch{index:05d}`.
- In **pgvector**, `chunk_id` is the primary reference key.
- In **Neo4j**, every edge created from that chunk stores `source_chunk_id: "{chunk_id}"`. In addition, a `DocumentChunk` node is created and connected via `[:MENTIONED_IN]`.

**Bi-directional Navigation**:
1. **Graph-to-Text**: When Cypher traversal uncovers paths (e.g. `Acme -> SUPPLIES -> Baker`), the retriever reads `edge.source_chunk_id` and performs an index lookup in PostgreSQL (`SELECT content FROM document_chunks WHERE chunk_id = ...`) to provide verbatim supporting context.
2. **Text-to-Graph**: When vector similarity retrieves a chunk, the system queries `entity_mentions` in PostgreSQL or `(:DocumentChunk {chunk_id: $id})<-[:MENTIONED_IN]-(e)` in Neo4j to expand a 1-to-2 hop graph neighborhood around the retrieved chunk.

### 3.2 Entity Extraction & Resolution
- **Extraction Model**: `llama3.1:8b` via Ollama (lowest JSON schema hallucination rate per `AGENTS.md`) using strict JSON schema output.
- **Extraction Schema**:
  ```json
  {
    "entities": [
      {"name": "Acme Corp", "type": "Company", "aliases": ["ACME", "Acme Corporation"], "sector": "Energy"}
    ],
    "relationships": [
      {"source": "Acme Corp", "target": "Baker Industries", "type": "SUPPLIES", "properties": {"product": "Turbines"}, "confidence": 0.95}
    ]
  }
  ```
- **Entity Resolution Algorithm**:
  1. **Syntactic Normalization**: Lowercase, strip legal entity suffixes (`Inc.`, `Corp.`, `LLC`, `Ltd.`), remove punctuation and excessive whitespace.
  2. **Canonical Mapping**: If normalized name matches an existing canonical name or known alias, map to that canonical entity.
  3. **Embedding Clustered Matching**: For ambiguous candidates, compare name embeddings with `nomic-embed-text`. If cosine similarity $> 0.88$, merge candidate into canonical entity alias list.
  4. **Idempotency**: All Neo4j writes use `MERGE` statements with `ON CREATE SET` and `ON MATCH SET` to guarantee ingestion idempotency.

---

## 4. Retrieval Architecture & Hybrid Fusion

```
                           Incoming Query
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Fast Classifier Router│
                     │    (qwen2.5:1.5b)     │
                     └───────────┬───────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
     [VECTOR ONLY]             [BOTH]            [GRAPH ONLY]
            │                    │                    │
            │            ┌───────┴───────┐            │
            ▼            ▼               ▼            ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  Vector RAG  │ │  Vector RAG  │ │  Graph RAG   │
     │  (pgvector)  │ │  (pgvector)  │ │   (Neo4j)    │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                └───────┬────────┘       │
            │                        ▼                │
            │            ┌───────────────────────┐    │
            │            │ Reciprocal Rank Fusion│    │
            │            │  (RRF: k=60, top-K=8) │    │
            │            └───────────┬───────────┘    │
            │                        │                │
            └────────────────► ◄─────┴────────────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │   Context Formatter   │
                   │ Graph Facts + Passages│
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │ Local Answer Generator│
                   │      (qwen3:8b)       │
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  Citation Validation  │
                   └───────────┬───────────┘
                               │
                               ▼
                          Final Output
```

### 4.1 Semantic Query Router
To prevent executing expensive graph traversals on simple factoid queries, questions are routed using a few-shot prompt executed on `qwen2.5:1.5b` (~1GB VRAM, sub-second latency):
- **`VECTOR`**: Single-fact lookups, definitions, policy text, direct chunk quotes.
- **`GRAPH`**: Multi-hop connections ("Which suppliers of X also supply Y?"), cross-entity relationships, hierarchy traversal ("All subsidiaries of ParentCo"), relational aggregations.
- **`BOTH`**: Complex, open-ended, or ambiguous questions where both textual passages and relational topology are required.
- **Low-Confidence Fallback**: Any uncertain classification defaults to `BOTH`.

### 4.2 Knowledge Graph Retrieval & Cypher Template Library
To ensure security and deterministic execution, **the LLM is never permitted to emit raw Cypher against the database**. The system uses a parameterized Cypher template registry:

1. **Entity & Intent Extraction**: The query is parsed to identify target entities and traversal intent.
2. **Template Selection**:
   - `TEMPLATE_SHARED_SUPPLIERS`:
     ```cypher
     MATCH (s:Company)-[:SUPPLIES]->(a:Company {canonical_name: $company_a}),
           (s:Company)-[:SUPPLIES]->(b:Company {canonical_name: $company_b})
     RETURN s.name AS supplier, s.sector AS sector, collect(r1.source_chunk_id) AS sources;
     ```
   - `TEMPLATE_MULTI_HOP_PATHS`:
     ```cypher
     MATCH p = (start:Company {canonical_name: $start_name})-[*1..3]-(target:Company {canonical_name: $target_name})
     RETURN p, [r in relationships(p) | r.source_chunk_id] AS source_chunks
     LIMIT 5;
     ```
   - `TEMPLATE_SUBSIDIARY_NETWORK`:
     ```cypher
     MATCH (parent:Company {canonical_name: $parent_name})<-[:SUBSIDIARY_OF*1..2]-(sub:Company)
     OPTIONAL MATCH (sub)-[:REPORTED]->(m:FinancialMetric)
     RETURN sub.name AS subsidiary, sub.sector AS sector, m.amount AS revenue, m.year AS year;
     ```
   - `TEMPLATE_NEIGHBORHOOD_EXPANSION`:
     ```cypher
     MATCH (e {canonical_name: $entity_name})-[r]-(neighbor)
     RETURN type(r) AS rel, neighbor.name AS neighbor_name, labels(neighbor)[0] AS neighbor_type, r.source_chunk_id AS source
     LIMIT 15;
     ```
3. **Graph Path to Natural Language Verbalization**:
   Raw graph triples are serialized into natural statements before entering context:
   - `(Acme)-[:SUBSIDIARY_OF]->(ParentCo)` -> `"Acme Corp is a subsidiary of ParentCo [source: doc01_ch00012]"`
   - `(GlobalParts)-[:SUPPLIES]->(Acme)` -> `"GlobalParts supplies Acme Corp [source: doc01_ch00045]"`

### 4.3 Hybrid Fusion: Reciprocal Rank Fusion (RRF)
When both vector and graph paths run, results are merged using Reciprocal Rank Fusion:

$$RRF\_Score(d) = \sum_{m \in \{vector, graph\}} \frac{1}{k + rank_m(d)}$$

Where:
- $k = 60$ (standard smoothing constant preventing high-ranking outliers from dominating).
- $rank_m(d)$ is the 1-based rank position of chunk $d$ in the retrieved list from store $m$. If chunk $d$ was not retrieved by store $m$, its term is 0.
- Chunks supported by **both** vector similarity and graph edge traversal receive high composite scores, pushing the most corroborated evidence to the top of the context window.
- The fused list is truncated to the top $K$ items ($K=8$).

---

## 5. Answer Generation & Citation Validation

### 5.1 Context Assembly Format
The context fed into `qwen3:8b` is partitioned and labeled:

```markdown
You are a factual research assistant. Answer the user question strictly using the provided context.
For every factual claim you make, you MUST cite the source chunk using the format [source: <chunk_id>].
If the provided context does not contain sufficient information, state that the question cannot be answered.

=== Graph-Derived Relationships ===
- Acme Corp is a subsidiary of ParentCo. [source: doc01_ch00012]
- GlobalParts supplies Acme Corp and Baker Industries. [source: doc01_ch00045]

=== Retrieved Text Passages ===
--- Chunk doc01_ch00012 ---
ParentCo acquired Acme Corp in 2021 to expand its renewable infrastructure portfolio...

--- Chunk doc01_ch00045 ---
GlobalParts entered into a multi-year supplier agreement with both Acme Corp and Baker Industries...

=== User Question ===
Which company supplies both Acme Corp and Baker Industries, and who owns Acme Corp?
```

### 5.2 Citation Validation Engine
Post-generation, the answer undergoes verification:
1. Regex extraction: `r"\[source:\s*([a-zA-Z0-9_\-]+)\]"`
2. Set comparison: Extracted citation IDs $\subseteq$ Valid retrieved chunk IDs $\{d_1, \dots, d_K\}$.
3. If an unrecognized or fabricated citation is found:
   - Log citation violation alert.
   - Strip fabricated citation or trigger a fallback regeneration pass with temperature 0.0 and explicit penalty instructions.

---

## 6. Infrastructure & Deployment Topology

### 6.1 Docker Services (`knowledge-graph-rag/docker-compose.yml`)
- **Neo4j**:
  - Image: `neo4j:5.26-community`
  - Ports: `7474:7474` (HTTP Browser/API), `7687:7687` (Bolt protocol)
  - Auth: `neo4j/password123`
  - Volumes: Persistent volume `neo4j_data` mapped to `/data`
- **PostgreSQL with pgvector**:
  - Image: `pgvector/pgvector:pg16`
  - Ports: `5432:5432`
  - Database: `rag_db`, User: `postgres`, Password: `postgres`
  - Volumes: Persistent volume `postgres_data` mapped to `/var/lib/postgresql/data`

### 6.2 Python Runtime & Local Inference
- Python 3.11+ managed with `uv`.
- Virtual environment at `knowledge-graph-rag/.venv`.
- Integration with root `_shared/ollama_helper.py` for Ollama model pulling, health checks, and OpenAI client initialization.
- Local LLM allocations on 8GB RTX 5060:
  - Routing: `qwen2.5:1.5b` (~1.0GB VRAM)
  - Extraction: `llama3.1:8b` (~4.9GB VRAM)
  - Embedding: `nomic-embed-text` (~0.3GB VRAM, 768 dimensions)
  - Generation: `qwen3:8b` (~5.1GB VRAM, chain-of-thought enabled)
  - *Co-residence rule*: Never run `llama3.1:8b` and `qwen3:8b` concurrently. Ollama handles LRU unloading with ~2-3s switch overhead.

---

## 7. MultiHopRAG Benchmark Methodology

The definitive validation artifact is an automated benchmark comparing Hybrid KG-RAG against a Vector-Only baseline on a 100-item evaluation slice from Hugging Face's `yixuantt/MultiHopRAG`.

### Metric Dimensions
1. **Hop-Stratified Accuracy**:
   - Single-hop questions (Expectation: Parity between Vector and Hybrid, ~80-90%).
   - 2-hop relational questions (Expectation: Hybrid outperforms Vector by $\ge 25\%$).
   - 3-hop / multi-hop relational questions (Expectation: Hybrid outperforms Vector by $\ge 40\%$).
   - Aggregations across entity sets (Expectation: Hybrid achieves substantial accuracy advantage).
2. **Citation Precision**:
   - Percentage of citations in generated answers that accurately map to ground-truth backing documents.
3. **Query Latency**:
   - p50, p90, and p95 end-to-end response times recorded for both Vector-only and Hybrid RAG.
4. **Ingestion Cost & Footprint**:
   - Number of chunks, number of graph nodes, number of edges, and offline build duration.

The benchmark script exports results to both JSON (`benchmark_results.json`) and a formatted Markdown comparison table suitable for display at the head of `README.md`.
