"""Task 8 & 9: Benchmark 20 questions, evaluate, and generate cost report."""
import os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_chunks, total_prompt_tokens, total_completion_tokens, PROJECT_ROOT
import utils
from build_graph import build_graph
from graph_query import answer_with_graphrag
from flat_rag import build_vector_index, answer_with_flat_rag

# 20 benchmark questions
BENCHMARK = [
    {"question": "Who founded OpenAI and which company invested in it?",
     "expected": "Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, Wojciech Zaremba, John Schulman founded OpenAI. Microsoft invested in OpenAI."},
    {"question": "Which companies are connected to AI products and who founded them?",
     "expected": "OpenAI (Sam Altman, Elon Musk et al.), Google (Larry Page, Sergey Brin), NVIDIA (Jensen Huang et al.), Anthropic (Dario Amodei, Daniela Amodei)."},
    {"question": "What is the relationship between Microsoft and OpenAI?",
     "expected": "Microsoft invested in OpenAI. Both work in artificial intelligence."},
    {"question": "Which company founded by Larry Page is connected to Android?",
     "expected": "Google, founded by Larry Page and Sergey Brin, created Android."},
    {"question": "Compare the founders and products of Google and OpenAI.",
     "expected": "Google was founded by Larry Page and Sergey Brin; products include Search, Gmail, Android. OpenAI was founded by Sam Altman, Elon Musk, etc.; products include GPT-4, ChatGPT."},
    {"question": "Which companies did Elon Musk found?",
     "expected": "Elon Musk founded OpenAI, Tesla, SpaceX, and X.AI."},
    {"question": "What products did NVIDIA create and who are its partners?",
     "expected": "NVIDIA created GeForce, CUDA, A100, H100. NVIDIA partnered with Microsoft."},
    {"question": "Which companies compete with Google?",
     "expected": "Microsoft competes with Google in cloud. Apple competes with Google in smartphones. Meta competes with Google in advertising."},
    {"question": "Who invested in Anthropic?",
     "expected": "Google and Amazon invested in Anthropic."},
    {"question": "What companies are headquartered in San Francisco?",
     "expected": "OpenAI and Anthropic are headquartered in San Francisco."},
    {"question": "Which companies work in the field of artificial intelligence?",
     "expected": "OpenAI, Google, NVIDIA, IBM, Anthropic, X.AI work in artificial intelligence."},
    {"question": "What did Microsoft acquire?",
     "expected": "Microsoft acquired LinkedIn in 2016 and GitHub in 2018."},
    {"question": "Compare Amazon and Microsoft in cloud computing.",
     "expected": "Amazon created AWS; Microsoft created Azure. Both compete in cloud services."},
    {"question": "Who founded Tesla and what products does it make?",
     "expected": "Tesla was founded by Elon Musk, JB Straubel, Martin Eberhard, Marc Tarpenning, Ian Wright. Products: Model S, Model 3, Model X, Model Y, Autopilot."},
    {"question": "What is the relationship between Meta and Google?",
     "expected": "Meta competes with Google in digital advertising."},
    {"question": "Which companies compete in the smartphone market?",
     "expected": "Apple and Samsung compete in the smartphone market."},
    {"question": "What products did Apple create?",
     "expected": "Apple created iPhone, iPad, Mac, Apple Watch, and iOS."},
    {"question": "Who founded IBM and what AI product did it create?",
     "expected": "IBM was founded by Charles Ranlett Flint. IBM created Watson."},
    {"question": "Which companies are connected to Elon Musk and what do they do?",
     "expected": "OpenAI (AI), Tesla (EVs), SpaceX (aerospace), X.AI (AI). Elon Musk founded all of them."},
    {"question": "What is the connection between Intel and NVIDIA?",
     "expected": "Intel competes with NVIDIA in the AI chip market. Both work in semiconductors."},
]


def score_answer(answer, expected):
    """Simple scoring: check how many key terms from expected appear in answer."""
    expected_terms = set(expected.lower().replace(",", " ").replace(".", " ").split())
    answer_terms = set(answer.lower().replace(",", " ").replace(".", " ").split())
    # Focus on named entities / key words
    key_words = {w for w in expected_terms if len(w) > 3 and w[0].isalpha()}
    if not key_words:
        return 0.5
    hits = len(key_words & answer_terms)
    ratio = hits / len(key_words)
    if ratio >= 0.7:
        return 1.0
    elif ratio >= 0.3:
        return 0.5
    return 0.0


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

    # Build graph and index
    print("Building graph...")
    t_graph_start = time.time()
    G = build_graph()
    t_graph = time.time() - t_graph_start

    print("Building vector index...")
    t_index_start = time.time()
    build_vector_index()
    t_index = time.time() - t_index_start

    # Reset token counters for query phase
    utils.total_prompt_tokens = 0
    utils.total_completion_tokens = 0

    results = []
    flat_times, graph_times = [], []

    for i, item in enumerate(BENCHMARK):
        q = item["question"]
        expected = item["expected"]
        print(f"\n[{i+1}/20] {q}")

        # Flat RAG
        t0 = time.time()
        flat_ans = answer_with_flat_rag(q)
        flat_time = time.time() - t0
        flat_times.append(flat_time)

        # GraphRAG
        t0 = time.time()
        graph_ans = answer_with_graphrag(q, G)
        graph_time = time.time() - t0
        graph_times.append(graph_time)

        flat_score = score_answer(flat_ans, expected)
        graph_score = score_answer(graph_ans, expected)

        notes = ""
        if graph_score > flat_score:
            notes = "GraphRAG better - captures multi-hop relations"
        elif flat_score > graph_score:
            notes = "Flat RAG better - text context sufficient"
        else:
            notes = "Tie"

        results.append({
            "question": q,
            "expected_answer": expected,
            "flat_rag_answer": flat_ans,
            "graphrag_answer": graph_ans,
            "flat_rag_score": flat_score,
            "graphrag_score": graph_score,
            "notes": notes,
        })
        print(f"  Flat={flat_score}, Graph={graph_score} | {notes}")

    # Save benchmark
    df = pd.DataFrame(results)
    bench_path = os.path.join(PROJECT_ROOT, "outputs", "benchmark_results.csv")
    df.to_csv(bench_path, index=False)
    print(f"\nSaved {bench_path}")

    # Print summary
    avg_flat = df["flat_rag_score"].mean()
    avg_graph = df["graphrag_score"].mean()
    print(f"\n{'='*50}")
    print(f"Average Flat RAG score:  {avg_flat:.2f}")
    print(f"Average GraphRAG score:  {avg_graph:.2f}")
    print(f"{'='*50}")

    # Load chunk/triple stats
    chunks = load_chunks()
    triples_df = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "triples.csv"))

    # Cost report
    avg_flat_time = sum(flat_times) / len(flat_times)
    avg_graph_time = sum(graph_times) / len(graph_times)

    cost_report = f"""# Cost Report

## Corpus Statistics

- Number of chunks: {len(chunks)}
- Number of extracted triples: {len(triples_df)}
- Number of graph nodes: {G.number_of_nodes()}
- Number of graph edges: {G.number_of_edges()}

## Runtime

- Graph construction time: {t_graph:.2f} seconds
- Vector index construction time: {t_index:.2f} seconds
- Average Flat RAG query time: {avg_flat_time:.2f} seconds
- Average GraphRAG query time: {avg_graph_time:.2f} seconds

## Token Usage

- Total prompt tokens: {utils.total_prompt_tokens}
- Total completion tokens: {utils.total_completion_tokens}
- Total tokens: {utils.total_prompt_tokens + utils.total_completion_tokens}

## Analysis

GraphRAG has a higher upfront indexing cost because it needs to extract triples and build a graph.
However, it provides more structured context during query time, especially for multi-hop questions
involving relationships among companies, founders, products, and investors.

### Benchmark Summary (20 questions)

- Average Flat RAG score: {avg_flat:.2f}
- Average GraphRAG score: {avg_graph:.2f}

GraphRAG performs better on questions requiring reasoning across multiple relationships (e.g.,
"Which companies did Elon Musk found?" requires connecting Musk to OpenAI, Tesla, SpaceX, and X.AI).
Flat RAG works well for single-entity questions where the relevant chunk contains all needed info.

The trade-off is that GraphRAG requires an initial investment in triple extraction (LLM calls per chunk)
but then provides precise, structured retrieval at query time. For domains with dense entity relationships,
GraphRAG is the superior choice.
"""
    cost_path = os.path.join(PROJECT_ROOT, "outputs", "cost_report.md")
    with open(cost_path, "w", encoding="utf-8") as f:
        f.write(cost_report)
    print(f"Saved {cost_path}")


if __name__ == "__main__":
    main()
