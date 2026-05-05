"""Task 6: GraphRAG query with 2-hop traversal."""
import os, sys
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from utils import call_llm
from build_graph import build_graph

ANSWER_PROMPT = """You are answering a question using only the provided knowledge graph context.
If the answer is not in the context, say that the information is not available.

Question: {question}

Knowledge graph context:
{context}

Answer clearly and concisely."""


def extract_query_entity(question):
    """Use LLM to extract main entities from question."""
    prompt = f"Extract the main entity names from this question. Return only a JSON list of strings.\nQuestion: {question}"
    import json
    raw = call_llm(prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except:
        # fallback: split by commas
        return [e.strip().strip('"') for e in raw.strip("[]").split(",")]


def get_subgraph_context(G, entity, max_hops=2):
    """Get context from graph within max_hops of entity."""
    context_lines = []
    # Find matching node (case-insensitive)
    matched = None
    for n in G.nodes():
        if n.lower() == entity.lower():
            matched = n
            break
    if not matched:
        return ""

    visited = {matched}
    frontier = [matched]
    for hop in range(max_hops):
        next_frontier = []
        for node in frontier:
            # outgoing edges
            for _, target, data in G.out_edges(node, data=True):
                context_lines.append(f"{node} {data['relation']} {target}")
                if target not in visited:
                    visited.add(target)
                    next_frontier.append(target)
            # incoming edges
            for source, _, data in G.in_edges(node, data=True):
                context_lines.append(f"{source} {data['relation']} {node}")
                if source not in visited:
                    visited.add(source)
                    next_frontier.append(source)
        frontier = next_frontier

    return "\n".join(list(dict.fromkeys(context_lines)))  # deduplicate preserving order


def answer_with_graphrag(question, G=None):
    """Answer a question using GraphRAG."""
    if G is None:
        G = build_graph()

    entities = extract_query_entity(question)
    all_context = []
    for entity in entities:
        ctx = get_subgraph_context(G, entity)
        if ctx:
            all_context.append(ctx)

    context = "\n".join(all_context) if all_context else "No relevant information found in knowledge graph."
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    return call_llm(prompt)


def main():
    G = build_graph()
    questions = [
        "Who founded OpenAI and what products is it known for?",
        "What is the relationship between Microsoft and OpenAI?",
        "Which companies did Elon Musk found?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        ans = answer_with_graphrag(q, G)
        print(f"A: {ans}")


if __name__ == "__main__":
    main()
