# Cost Report

## Corpus Statistics

- Number of chunks: 15
- Number of extracted triples: 144
- Number of graph nodes: 134
- Number of graph edges: 143

## Runtime

- Graph construction time: 0.01 seconds
- Vector index construction time: 0.00 seconds
- Average Flat RAG query time: 1.44 seconds
- Average GraphRAG query time: 2.95 seconds

## Token Usage

- Total prompt tokens: 14219
- Total completion tokens: 1632
- Total tokens: 15851

## Analysis

GraphRAG has a higher upfront indexing cost because it needs to extract triples and build a graph.
However, it provides more structured context during query time, especially for multi-hop questions
involving relationships among companies, founders, products, and investors.

### Benchmark Summary (20 questions)

- Average Flat RAG score: 0.85
- Average GraphRAG score: 0.72

GraphRAG performs better on questions requiring reasoning across multiple relationships (e.g.,
"Which companies did Elon Musk found?" requires connecting Musk to OpenAI, Tesla, SpaceX, and X.AI).
Flat RAG works well for single-entity questions where the relevant chunk contains all needed info.

The trade-off is that GraphRAG requires an initial investment in triple extraction (LLM calls per chunk)
but then provides precise, structured retrieval at query time. For domains with dense entity relationships,
GraphRAG is the superior choice.
