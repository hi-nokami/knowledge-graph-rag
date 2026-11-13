# Knowledge Graph RAG (KG-RAG)

Hybrid Retrieval-Augmented Generation pairing native Property Graphs (**Neo4j**) with dense Vector Search (**PostgreSQL + pgvector**), fused via Reciprocal Rank Fusion (RRF) and validated with post-generation citation checking.

## MultiHopRAG Benchmark Results

*Evaluation on 100-item stratified slice from MultiHopRAG (1-hop, 2-hop, 3-hop, and entity aggregation).*

| Question Category | Hop Count | Vector-Only Accuracy | Hybrid KG-RAG Accuracy | Accuracy Gain (Δ) | Vector p50 (ms) | Hybrid p50 (ms) | Citation Precision |
|:-------------------|:---------:|:--------------------:|:----------------------:|:-----------------:|:---------------:|:---------------:|:------------------:|
| **1-Hop Factoid**  | 1         | 52.0% | 16.0% | -36.0% | 5484.52ms | 6031.70ms | 76.0% |
| **2-Hop Relational**| 2        | 36.0% | 40.0% | +4.0% | 8183.94ms | 11041.16ms | 92.0% |
| **3-Hop Relational**| 3        | 68.0% | 32.0% | -36.0% | 9779.76ms | 10794.23ms | 76.0% |
| **Aggregation**    | Agg       | 96.0% | 72.0% | -24.0% | 7917.07ms | 7222.25ms | 84.0% |
| **Overall**        | —         | **63.0%** | **40.0%** | **-23.0%** | **8154.87ms** | **7987.39ms** | **82.0%** |

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
