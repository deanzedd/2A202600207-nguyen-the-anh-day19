"""Task 5: Build knowledge graph from triples.csv and save visualization."""
import os, sys
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import PROJECT_ROOT


def build_graph(triples_path=None):
    """Build a directed graph from triples CSV."""
    if triples_path is None:
        triples_path = os.path.join(PROJECT_ROOT, "outputs", "triples.csv")
    df = pd.read_csv(triples_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row["subject"])
        G.add_node(row["object"])
        G.add_edge(row["subject"], row["object"],
                    relation=row["relation"],
                    source_chunk_id=row.get("source_chunk_id", ""))
    return G


def visualize_graph(G, output_path=None):
    """Save graph visualization to PNG."""
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "outputs", "graph.png")
    plt.figure(figsize=(20, 14))
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold")

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True,
                            arrowsize=12, connectionstyle="arc3,rad=0.1")

    # Edge labels
    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=5, font_color="red")

    plt.title("Tech Company Knowledge Graph", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {output_path}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    G = build_graph()
    visualize_graph(G)
    # Also save as GraphML
    graphml_path = os.path.join(PROJECT_ROOT, "outputs", "knowledge_graph.graphml")
    nx.write_graphml(G, graphml_path)
    print(f"GraphML saved to {graphml_path}")


if __name__ == "__main__":
    main()
