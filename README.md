# Knowledge Graph RAG (KG-RAG)

Hybrid Retrieval-Augmented Generation pairing native Property Graphs (**Neo4j**) with dense Vector Search (**PostgreSQL + pgvector**), fused via Reciprocal Rank Fusion (RRF) and validated with post-generation citation checking.

## MultiHopRAG Benchmark Results

*Evaluation on 100-item stratified slice from MultiHopRAG (1-hop, 2-hop, 3-hop, and entity aggregation).*

| Question Category | Hop Count | Vector-Only Accuracy | Hybrid KG-RAG Accuracy | Accuracy Gain (Δ) | Vector p50 (ms) | Hybrid p50 (ms) | Citation Precision |
|:-------------------|:---------:|:--------------------:|:----------------------:|:-----------------:|:---------------:|:---------------:|:------------------:|
| **1-Hop Factoid**  | 1         | 0.0% | 0.0% | +0.0% | 16.08ms | 16.38ms | 100.0% |
| **2-Hop Relational**| 2        | 0.0% | 0.0% | +0.0% | 16.10ms | 16.37ms | 100.0% |
| **3-Hop Relational**| 3        | 8.0% | 8.0% | +0.0% | 16.17ms | 16.41ms | 100.0% |
| **Aggregation**    | Agg       | 12.0% | 12.0% | +0.0% | 16.15ms | 16.36ms | 100.0% |
| **Overall**        | —         | **5.0%** | **5.0%** | **+0.0%** | **16.11ms** | **16.38ms** | **100.0%** |

## Architecture

- **Vector Store**: PostgreSQL 16 with `pgvector`, HNSW index (`vector_cosine_ops`), 768-dim embeddings (`nomic-embed-text`).
- **Graph Store**: Neo4j Community 5, strict corporate and financial ontology (`Company`, `Person`, `Product`, `Location`, `FinancialMetric`).
- **Dual-Store Bridge**: Chunks assigned deterministic IDs (`{doc_id}_ch{index:05d}`). Graph edges store `source_chunk_id` for bidirectional navigation.
- **Routing**: `qwen2.5:1.5b` classifies queries into `VECTOR`, `GRAPH`, or `BOTH`.
- **Traversal**: Parameterized Cypher template registry (prevents raw LLM Cypher generation).
- **Fusion**: Reciprocal Rank Fusion (RRF, $k=60$) combining vector and graph retrieval streams.
- **Answer Generation**: `qwen3:8b` via Ollama with structured evidence sections and claim citations `[source: <chunk_id>]`.
- **Validation**: Regex citation extraction and set comparison against retrieved context.

## Local Infrastructure

Start the databases via Docker Compose:
```bash
docker compose up -d
```

## Setup & Installation

Managed with `uv`:
```bash
uv venv
uv pip install -e .
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

```bash
python benchmarks/multihop_benchmark.py
```
