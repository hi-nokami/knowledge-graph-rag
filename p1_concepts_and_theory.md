# P1: Knowledge Graph RAG — Concepts & Theory

> **Goal of this document**: Build your mental model of *every* piece of this system before we write a line of code or draw an architecture diagram. After reading this, you should be able to explain to an interviewer — from first principles — why plain vector RAG breaks, what a knowledge graph adds, how they combine, and what makes the result credible.

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [Plain Vector RAG — What It Is and Where It Breaks](#2-plain-vector-rag--what-it-is-and-where-it-breaks)
3. [Knowledge Graphs — The Missing Structure](#3-knowledge-graphs--the-missing-structure)
4. [Hybrid Retrieval — Combining Both Worlds](#4-hybrid-retrieval--combining-both-worlds)
5. [The Query Router — Picking the Right Path](#5-the-query-router--picking-the-right-path)
6. [Answer Synthesis — Merging Sources into Grounded Responses](#6-answer-synthesis--merging-sources-into-grounded-responses)
7. [The Benchmark — Proving It Works](#7-the-benchmark--proving-it-works)
8. [Technology Choices and Why](#8-technology-choices-and-why)
9. [End-to-End Data Flow](#9-end-to-end-data-flow)
10. [Key Vocabulary](#10-key-vocabulary)

---

## 1. The Problem We're Solving

### The interview question this answers

> *"You've built RAG before. What happens when the answer requires connecting information from multiple documents?"*

Almost every AI engineering candidate has built vector RAG — chunk documents, embed them, retrieve the top-k most similar chunks to a question, feed them to an LLM. It works for **single-hop** factual questions: *"What is Acme Corp's refund policy?"*

It breaks — often silently — on **multi-hop** questions:

- *"Which companies that supply Acme Corp also have contracts with Baker Industries?"* (2-hop: Acme → suppliers → Baker)
- *"What is the total revenue of all subsidiaries of ParentCo that operate in the energy sector?"* (3-hop: ParentCo → subsidiaries → sector filter → revenue aggregation)
- *"How did the board composition change between the 2022 and 2023 annual reports?"* (comparison across entities and time)

These questions require **traversing relationships** between entities. Embeddings encode *meaning*, not *structure*. That's the fundamental gap.

### Why this matters in production

In enterprise settings — legal discovery, financial analysis, compliance, supply-chain intelligence — the most valuable questions are almost always relational. A system that can only answer "what does this paragraph say?" is a search engine with extra steps. A system that can answer "how are these things connected?" is genuinely useful.

This project builds the second kind and **proves the difference with numbers**.

---

## 2. Plain Vector RAG — What It Is and Where It Breaks

### How vector RAG works

```
Documents → Chunk → Embed → Store in vector DB → Query time:
  User question → Embed → Find top-k similar chunks → Feed to LLM → Answer
```

Each step in detail:

| Step | What happens | Key decision |
|------|-------------|--------------|
| **Chunk** | Split documents into overlapping segments (~500-1000 tokens) | Chunk size, overlap amount |
| **Embed** | Convert each chunk into a dense vector (e.g., 768 dimensions) | Embedding model choice |
| **Index** | Store vectors with an approximate nearest-neighbor index | HNSW vs IVF, distance metric |
| **Retrieve** | Find the k chunks whose vectors are closest to the query vector | k value, similarity threshold |
| **Generate** | LLM reads the retrieved chunks and answers the question | Context window management |

### Why it works for single-hop questions

When a user asks *"What is Acme Corp's refund policy?"*, the embedding of that question is semantically close to the embedding of the paragraph that contains the refund policy. Cosine similarity does the right thing. The relevant chunk lands in the top-k, the LLM reads it, done.

### Where and why it fails

#### Failure mode 1: Information is spread across chunks

> Q: "Which companies supply both Acme Corp and Baker Industries?"

The answer requires:
- Chunk A (doc 1, page 3): "Acme Corp's primary suppliers include Steelworks Inc and GlobalParts..."
- Chunk B (doc 4, page 12): "Baker Industries sources materials from GlobalParts and MetalCo..."

The question's embedding is somewhat similar to both chunks, but in a corpus of 10,000 chunks, neither may rank in the top-10. Even if both are retrieved, the LLM has to **infer the intersection** — something it can do if both chunks happen to appear, but that's luck, not a guarantee.

#### Failure mode 2: Reasoning requires traversal, not similarity

> Q: "What is the total revenue of all subsidiaries of ParentCo in the energy sector?"

This requires:
1. Know that ParentCo has subsidiaries X, Y, Z (relationship)
2. Filter to those in the energy sector (attribute)
3. Look up each one's revenue (data retrieval)
4. Sum them (aggregation)

No single chunk contains this answer. The *embeddings* of the subsidiaries' revenue paragraphs aren't semantically similar to the question about ParentCo. Vector search finds passages that **sound like** the question, not passages that **logically connect** to it.

#### Failure mode 3: Entity ambiguity

"Acme Corp", "Acme Corporation", "ACME", and "the Company" (in an Acme filing) all refer to the same entity. Embeddings treat them as *similar but distinct* strings. A knowledge graph collapses them into **one node**.

### The core insight

> **Embeddings encode semantic similarity. Graphs encode structural relationships. Multi-hop questions need structure.**

Vector RAG is not wrong — it's *incomplete*. We're not replacing it. We're adding what it's missing.

---

## 3. Knowledge Graphs — The Missing Structure

### What is a knowledge graph?

A knowledge graph is a data structure that stores **entities** (things) and **relationships** (connections between things) as a network of nodes and edges.

```
[Acme Corp] --SUPPLIES--> [Baker Industries]
    |                          |
    |--SUBSIDIARY_OF-->  [ParentCo]
    |                          |
    |--HAS_REVENUE-->    [$2.4B]
    |
    |--HEADQUARTERED_IN--> [Houston, TX]
    |
    |--SECTOR--> [Energy]
```

Each element:
- **Node** (entity): A thing with a type and properties. `(Company {name: "Acme Corp", sector: "Energy", revenue: 2.4e9})`
- **Edge** (relationship): A typed, directed connection between two nodes. `(Acme)-[:SUPPLIES]->(Baker)`
- **Property**: Key-value data attached to a node or edge. `{since: 2019, contract_value: 5e6}`

### Why graphs answer multi-hop questions

The question *"Which companies supply both Acme Corp and Baker Industries?"* becomes a **graph traversal**:

```cypher
MATCH (supplier)-[:SUPPLIES]->(acme:Company {name: "Acme Corp"}),
      (supplier)-[:SUPPLIES]->(baker:Company {name: "Baker Industries"})
RETURN supplier.name
```

This is a **structural query** — it follows edges. It doesn't care about semantic similarity. It asks: "give me nodes that have outgoing SUPPLIES edges to both of these specific nodes." The answer is exact, complete, and reproducible.

Compare that to asking an embedding model to find chunks that are "similar to" the question. One is a database query. The other is a fuzzy search.

### The ontology — why constraints matter

An **ontology** is a predefined schema for your graph: what types of entities exist and what types of relationships connect them.

**Without an ontology** (open-ended extraction):
```
"Extract all entities and relationships from this text."
→ Produces: Person, Organization, Location, Date, Amount, Concept, Event, ...
→ 200 entity types, 500 relationship types
→ No two documents extract the same way
→ You can't write queries against this
```

**With a constrained ontology**:
```
Entity types (5-10):   Company, Person, Filing, Regulation, Location
Relationship types (8-15): SUPPLIES, SUBSIDIARY_OF, FILED, REGULATES, LOCATED_IN,
                          EMPLOYS, CONTRACTED_BY, REPORTED_REVENUE
```

Now every extraction pass produces nodes and edges that fit the same schema. You can write Cypher queries because you know what types exist. The graph is **queryable** rather than a random hairball.

> [!IMPORTANT]
> The ontology is the single most consequential design decision in this project. Too broad → unqueryable. Too narrow → misses real relationships. We'll define ours before writing any extraction code.

### Entity extraction — how the graph gets built

Raw documents don't arrive as graphs. We use an LLM to **extract structured triples** from text:

```
Input chunk: "Acme Corp, a subsidiary of ParentCo, reported Q3 revenue of $2.4 billion.
              The company's Houston facility..."

LLM extraction output (JSON):
{
  "entities": [
    {"name": "Acme Corp", "type": "Company", "aliases": ["ACME", "the company"]},
    {"name": "ParentCo", "type": "Company"},
    {"name": "Houston", "type": "Location"}
  ],
  "relationships": [
    {"source": "Acme Corp", "target": "ParentCo", "type": "SUBSIDIARY_OF"},
    {"source": "Acme Corp", "target": "Houston", "type": "LOCATED_IN"}
  ],
  "properties": [
    {"entity": "Acme Corp", "key": "q3_revenue", "value": 2.4e9}
  ]
}
```

The LLM reads text and outputs structured data according to the ontology. Each extraction carries:
- The **source chunk ID** (so we can trace any graph fact back to the sentence that produced it)
- A **confidence score** (so we can filter low-quality extractions)

### Entity resolution — the hard part everyone skips

"Acme Corp", "Acme Corporation", "ACME", and "the Company" must all map to the **same node**. If they don't, your graph thinks there are four separate companies and none of them have enough edges to be useful.

Entity resolution involves:

1. **Normalization**: Lowercase, strip suffixes (Corp/Corporation/Inc/LLC), standardize whitespace
2. **Embedding similarity**: If two entity names embed within a threshold distance, they're candidates for merging
3. **Alias storage**: The canonical node stores all known aliases: `{name: "Acme Corp", aliases: ["Acme Corporation", "ACME"]}`
4. **Merge strategy**: Use `MERGE` in Cypher (not `CREATE`) so re-running ingestion is idempotent — it updates existing nodes rather than creating duplicates

> [!WARNING]
> Skipping entity resolution is cited in the spec as one of the top failure modes. If "Acme Corp" exists as four disconnected nodes, traversal queries that should return results will return nothing. The graph looks populated but is functionally broken.

---

## 4. Hybrid Retrieval — Combining Both Worlds

### The architecture in one sentence

**Both the graph and the vector index are built from the same chunks, share the same chunk IDs, and are queried by a router that picks the right one (or both) per question.**

### The shared chunk ID — the whole trick

This is the critical design insight. When we process a document:

1. We **chunk** it → each chunk gets a unique ID (`chunk_0042`)
2. We **embed** the chunk → stored in pgvector with `chunk_id = "chunk_0042"`
3. We **extract entities/relationships** from the chunk → every graph edge carries `source_chunk_id = "chunk_0042"`

This shared key enables two crucial operations:

- **Graph → Text**: A graph traversal returns a path like `Acme --SUPPLIES--> Baker`. We follow the `source_chunk_id` on that edge back to the original sentence: *"Acme Corp has been a primary supplier to Baker Industries since 2019."* Now we have both the structural fact and the supporting text.

- **Text → Graph**: A vector search returns a chunk about Acme Corp. We look up which entity IDs that chunk mentions, then expand the **graph neighborhood** around those entities. Now we have both the relevant passage and the broader context of how Acme is connected to other entities.

```
┌─────────────────────────────────────────────┐
│                 Document                     │
│  ┌─────────────────────────────────────┐    │
│  │           Chunking                   │    │
│  │  chunk_001, chunk_002, chunk_003 ... │    │
│  └──────────┬──────────────┬───────────┘    │
│             │              │                 │
│      ┌──────▼──────┐  ┌───▼────────────┐   │
│      │  Embed      │  │  Extract       │   │
│      │  (vectors)  │  │  (entities +   │   │
│      │             │  │   relations)   │   │
│      └──────┬──────┘  └───┬────────────┘   │
│             │              │                 │
│      ┌──────▼──────┐  ┌───▼────────────┐   │
│      │  pgvector   │  │  Neo4j         │   │
│      │  chunk_id ◄─┼──┼─► chunk_id     │   │
│      │  + metadata │  │  on every edge │   │
│      └─────────────┘  └────────────────┘   │
└─────────────────────────────────────────────┘
```

### Why two stores instead of one?

| Capability | Vector (pgvector) | Graph (Neo4j) |
|-----------|-------------------|---------------|
| "Find passages about X" | ✅ Excellent | ❌ Not its job |
| "How is X related to Y?" | ❌ Can't traverse | ✅ Exactly this |
| "What does the policy say?" | ✅ Semantic match | ❌ No text search |
| "Which entities are connected to X within 3 hops?" | ❌ No structure | ✅ Cypher traversal |
| "Define term Z" | ✅ Definition lives in a chunk | ❌ Terms aren't entities |
| "Compare X and Y's financials" | ⚠️ Might find both | ✅ Structured comparison |

Neither store is sufficient alone. Together, they cover the full question space.

---

## 5. The Query Router — Picking the Right Path

### Why we need a router

Every incoming question should go to the retrieval method best equipped to answer it. Sending a simple definition question through graph traversal is wasteful. Sending a multi-hop relationship question through vector search produces garbage.

### How the router works

The router is a **cheap, fast classifier** — not a deep reasoning step. It takes the user's question and outputs one of three decisions:

| Decision | When | Example question |
|----------|------|-----------------|
| **Vector** | Definitions, policy lookups, single-fact questions | "What is the company's return policy?" |
| **Graph** | Connection questions, multi-hop chains, comparisons, aggregations | "Which suppliers serve both Acme and Baker?" |
| **Both** | Ambiguous questions, or when confidence is low | "Tell me about Acme Corp's business" |

### Implementation: a few-shot classification prompt

```
System: Classify the user's question into one of: VECTOR, GRAPH, BOTH.

VECTOR — the answer is in a single passage (definitions, policies, facts)
GRAPH  — the answer requires connecting entities across relationships
BOTH   — ambiguous, or the answer benefits from both structure and text

Examples:
Q: "What is the company's refund policy?" → VECTOR
Q: "Which companies supply both Acme and Baker?" → GRAPH
Q: "What did the CEO say about the merger?" → VECTOR
Q: "How are Acme's subsidiaries connected to the energy sector?" → GRAPH
Q: "Tell me about Acme Corp" → BOTH
```

This runs on the **fast/routing model** (`qwen2.5:1.5b` — ~1GB VRAM) so it doesn't compete with the main reasoning model for GPU memory. It returns an enum, not free text.

### The low-confidence fallback

When the router isn't sure, it outputs `BOTH`. In this mode:
1. Run **both** retrieval paths in parallel
2. Merge and deduplicate the results
3. Feed the combined context to the LLM

This is more expensive (two retrievals instead of one) but safer. Over time, the routing log data lets you tune the router's accuracy.

### Security: no raw Cypher from the LLM

> [!CAUTION]
> The spec is explicit: **never let the model emit raw Cypher against your database.** This is both a security control and a reproducibility control.

Instead, we maintain a **template library** — parameterized Cypher queries keyed by query type:

```python
CYPHER_TEMPLATES = {
    "find_connections": """
        MATCH (a:{entity_type} {{name: $entity_a}})-[r]->(b:{entity_type} {{name: $entity_b}})
        RETURN type(r) as relationship, r.source_chunk_id as source
    """,
    "find_shared_suppliers": """
        MATCH (s)-[:SUPPLIES]->(a:Company {name: $company_a}),
              (s)-[:SUPPLIES]->(b:Company {name: $company_b})
        RETURN s.name as supplier, s.sector as sector
    """,
    # ... more templates
}
```

The LLM's job is to:
1. Extract entity names from the question
2. Resolve them to node IDs (using entity resolution)
3. Pick the right template
4. Fill in the parameters

It never writes Cypher. It fills slots.

---

## 6. Answer Synthesis — Merging Sources into Grounded Responses

### The problem with raw graph output

A graph traversal returns **paths** — sequences of nodes and edges:

```
(Acme Corp)-[:SUBSIDIARY_OF]->(ParentCo)-[:OPERATES_IN]->(Energy)
```

If you dump this raw into an LLM prompt, it generates awkward, stilted text. The fix: **convert paths into readable statements** before they enter the prompt:

```
"Acme Corp is a subsidiary of ParentCo. ParentCo operates in the Energy sector."
```

### Context assembly

The final LLM prompt is assembled from two labeled sections:

```
=== Graph-derived facts ===
- Acme Corp is a subsidiary of ParentCo. [source: chunk_0042]
- ParentCo operates in the Energy sector. [source: chunk_0187]
- Baker Industries is also supplied by Acme Corp. [source: chunk_0091]

=== Retrieved passages ===
- "In Q3 2023, Acme Corp reported revenue of $2.4 billion, driven primarily
   by increased demand in the energy infrastructure segment..." [source: chunk_0042]
- "Baker Industries announced a new five-year supply agreement with..."
   [source: chunk_0155]
```

Why labeled sections? So the LLM (and any downstream evaluator) can distinguish between **structural facts** (from the graph) and **textual evidence** (from vector retrieval). This matters for citation validation.

### Citation validation — the zero-cost quality check

Every claim in the LLM's answer must carry a citation (`[source: chunk_XXXX]`). After generation, we run a **post-processing step**:

1. Extract all cited chunk IDs from the response
2. Check each one against the set of chunk IDs that were *actually retrieved*
3. If a citation references a chunk that wasn't retrieved → the LLM fabricated a source → **reject and regenerate**

This is cheap (string matching, no LLM call) and eliminates invented sources entirely. It doesn't guarantee the answer is *correct*, but it guarantees every claim points to a real document.

---

## 7. The Benchmark — Proving It Works

### Why the benchmark is the project

The spec is clear: the benchmark table is the portfolio artifact. It goes at the top of the README, above the architecture diagram. Without it, this is a tutorial you followed.

### What we measure

We build two systems and run them against the **same** question set:

| System | Description |
|--------|-------------|
| **Baseline** | Plain vector RAG (pgvector only, no graph) |
| **Hybrid** | Our full system (vector + graph + router) |

### The question set (50-100 items)

Stratified by difficulty:

| Category | Count | Example | Expected winner |
|----------|-------|---------|----------------|
| **Single-hop** | ~20 | "What is Acme's refund policy?" | Tied (both should get this) |
| **Two-hop** | ~20 | "Which suppliers serve both Acme and Baker?" | Hybrid wins |
| **Three-hop** | ~10 | "Total revenue of ParentCo's energy-sector subsidiaries?" | Hybrid wins big |
| **Aggregation** | ~10 | "How many companies operate in the energy sector?" | Hybrid wins |
| **Out-of-scope** | ~10 | "What's the weather in Houston?" | Both should refuse |

### The expected shape

```
Accuracy by hop count:

100% ┤
 90% ┤  ██  ██          ██
 80% ┤  ██  ██          ██  ██
 70% ┤  ██  ██    ██    ██  ██
 60% ┤  ██  ██    ██    ██  ██
 50% ┤  ██  ██    ██    ██  ██
 40% ┤  ██  ██    ██ ██ ██  ██
 30% ┤  ██  ██    ██ ██ ██  ██
     └──────────────────────────
      1-hop     2-hop    3-hop

      ██ Hybrid    ██ Vector-only
```

**Parity on single-hop, widening gap as hops increase.** That curve is the entire story.

### What else we report (honestly)

| Metric | Vector-only | Hybrid | Note |
|--------|------------|--------|------|
| Accuracy (1-hop) | ~85% | ~85% | Expected parity |
| Accuracy (2-hop) | ~40% | ~75% | The gap |
| Accuracy (3-hop) | ~15% | ~60% | The big gap |
| Avg latency/query | ~2s | ~4s | Hybrid is slower |
| Ingestion cost | Low | Higher | Graph extraction costs LLM calls |
| Infrastructure | pgvector only | pgvector + Neo4j | More moving parts |

> [!TIP]
> Being honest that graph RAG is slower and more expensive to build is what makes the accuracy claim credible. A benchmark that shows your system winning on every dimension is suspicious. Real engineering involves trade-offs.

---

## 8. Technology Choices and Why

### Our stack (adapted for local-only inference)

| Component | Technology | Why |
|-----------|-----------|-----|
| **Graph database** | Neo4j (Docker) | Industry standard for property graphs. Cypher query language is readable and powerful. Native graph traversal — not a relational DB pretending. |
| **Vector database** | pgvector (PostgreSQL extension) | We need metadata filtering alongside vector search. PostgreSQL is battle-tested. One fewer service vs. a standalone vector DB. |
| **Entity extraction LLM** | `qwen3:8b` via Ollama | Needs reasoning ability for extraction. Built-in chain-of-thought helps with complex passages. |
| **Structured output LLM** | `llama3.1:8b` via Ollama | Lowest JSON hallucination rate — critical for schema-validated extraction output. |
| **Embedding model** | `nomic-embed-text` via Ollama | 0.3GB VRAM, 8K context, co-resident with any large model. |
| **Router model** | `qwen2.5:1.5b` via Ollama | 1GB VRAM, fast enum classification. Can co-reside with the embedding model. |
| **API framework** | FastAPI | Async, type-safe, automatic OpenAPI docs. Standard for Python ML services. |
| **Orchestration** | LangChain | Provides abstractions for the retrieval pipeline. Ollama integration is mature. |

### VRAM management for this project

We can't load multiple 5GB models simultaneously. Our strategy:

```
Ingestion pipeline:
  Phase 1: Load llama3.1:8b (extraction, ~4.9GB) + nomic-embed-text (~0.3GB) = ~5.2GB ✅

Query pipeline:
  Phase 1: Load qwen2.5:1.5b (routing, ~1.0GB) + nomic-embed-text (~0.3GB) = ~1.3GB ✅
  Phase 2: Swap to qwen3:8b (answer generation, ~5.1GB) + nomic-embed-text (~0.3GB) = ~5.4GB ✅
```

Ollama auto-swaps idle models with a ~2-5 second delay. The query pipeline has two model-swap points, which adds latency but keeps us within 8GB.

---

## 9. End-to-End Data Flow

### Ingestion (offline, run once per corpus update)

```
Raw Documents
     │
     ▼
┌──────────┐
│  Chunker  │  Split into overlapping segments (~512 tokens)
└────┬──┬───┘  Each chunk gets a unique ID
     │  │
     │  └──────────────────────────────────┐
     ▼                                     ▼
┌─────────────┐                    ┌──────────────────┐
│  Embed       │                    │  LLM Extraction   │
│  (nomic)     │                    │  (llama3.1:8b)     │
│              │                    │                    │
│  chunk →     │                    │  chunk → entities  │
│  vector      │                    │  + relationships   │
│              │                    │  + confidence       │
└──────┬──────┘                    └──────┬─────────────┘
       │                                  │
       ▼                                  ▼
┌─────────────┐                    ┌──────────────────┐
│  pgvector    │                    │  Entity Resolution │
│              │                    │  (deduplicate,     │
│  Store:      │                    │   merge aliases)   │
│  - vector    │                    └──────┬─────────────┘
│  - chunk_id  │                           │
│  - metadata  │                           ▼
│  (doc_id,    │                    ┌──────────────────┐
│   section,   │                    │  Neo4j             │
│   date,      │                    │                    │
│   entity_ids)│                    │  Store:            │
│              │◄───── shared ──────│  - nodes (entities)│
│              │     chunk_id       │  - edges (rels)    │
└─────────────┘                    │  - source_chunk_id │
                                   │    on every edge   │
                                   └──────────────────┘
```

### Query (online, per user question)

```
User Question: "Which companies supply both Acme Corp and Baker Industries?"
     │
     ▼
┌──────────────┐
│  Router       │  qwen2.5:1.5b classifies → GRAPH
│  (few-shot)   │
└──────┬───────┘
       │
       ▼ (GRAPH path selected)
┌──────────────────┐
│  Entity Extractor │  Extract "Acme Corp" and "Baker Industries" from question
│  + Resolver       │  Resolve to node IDs in Neo4j
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Template Picker  │  Select "find_shared_suppliers" template
│  + Param Filler   │  Fill: company_a="Acme Corp", company_b="Baker Industries"
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Neo4j Query      │  Execute parameterized Cypher
│                    │  Returns: paths + source_chunk_ids
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Chunk Retrieval  │  Fetch original text for each source_chunk_id from pgvector
│  (for citations)  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Context Assembly │  Convert graph paths → readable statements
│                    │  Label: graph facts vs. text passages
│                    │  Deduplicate across sources
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  LLM Generation   │  qwen3:8b reads assembled context
│                    │  Generates answer with per-claim citations
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Citation Check   │  Verify every cited chunk_id was actually retrieved
│                    │  If fabricated → reject and regenerate
└──────┬───────────┘
       │
       ▼
  Final Answer (with validated citations)
```

---

## 10. Key Vocabulary

| Term | Definition | Why it matters here |
|------|-----------|-------------------|
| **RAG** | Retrieval-Augmented Generation — retrieve relevant info, then generate an answer grounded in it | The foundation of the entire project |
| **Multi-hop question** | A question whose answer requires connecting 2+ pieces of information via relationships | The problem vector RAG can't solve |
| **Knowledge graph** | A database of entities (nodes) and relationships (edges) | Provides the structure for multi-hop traversal |
| **Ontology** | A predefined schema of entity types and relationship types | Constrains extraction so the graph is queryable |
| **Entity resolution** | Collapsing different names for the same thing into one node | Without it, the graph is fragmented and useless |
| **Cypher** | Neo4j's query language for graph traversal | How we ask structural questions |
| **pgvector** | PostgreSQL extension for vector similarity search | Our vector store, co-located with metadata |
| **HNSW** | Hierarchical Navigable Small World — an ANN index algorithm | Fast approximate nearest-neighbor search in pgvector |
| **Chunk ID** | Unique identifier for a text segment, shared across both stores | The bridge between graph facts and source text |
| **Router** | A classifier that decides which retrieval path to use per question | Avoids wasting graph queries on simple questions |
| **MERGE (Cypher)** | Create-if-not-exists, update-if-exists | Makes re-ingestion idempotent |
| **Citation validation** | Post-generation check that cited sources were actually retrieved | Eliminates fabricated references at zero cost |
| **Grounding** | Ensuring LLM output is based on retrieved evidence, not invented | The entire point of RAG |

---

## What's Next

Once you're comfortable with these concepts, the next step is **system/architecture design** — where we'll define:

1. The exact ontology (entity types, relationship types) for our chosen corpus
2. The corpus selection (what documents we'll use)
3. The service architecture (what runs where, how services communicate)
4. The data models (Pydantic schemas, database schemas)
5. The API contract (FastAPI endpoints)
6. The deployment topology (Docker Compose services)

We won't write code until the architecture is settled and you're confident in every decision.
