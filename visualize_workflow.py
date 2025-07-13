import networkx as nx, matplotlib.pyplot as plt

def _edges(node, parent=None, out=None):
    if out is None:
        out = []
    name = node.__class__.__name__
    if name == "Predict":
        return out                         # skip internal nodes
    if parent:
        out.append((parent, name))
    for child in vars(node).values():
        if isinstance(child, dspy.Module):
            _edges(child, name, out)
    return out


# ── prettier palette & styling ──────────────────────────────────────────
def visualize(agent):
    G   = nx.DiGraph()
    G.add_edges_from(_edges(agent))

    # layout ‒ spring looks cleaner with k tweak
    pos = nx.spring_layout(G, k=0.7, seed=42)

    # node styling
    node_colors = "#009E73"   # a pleasant teal
    edge_color  = "#444444"
    label_color = "#222222"

    plt.figure(figsize=(9, 7), facecolor="#fafafa")
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2000,
        node_color=node_colors,
        edgecolors="#555555",
        linewidths=1.5,
        alpha=0.9
    )
    nx.draw_networkx_edges(
        G, pos,
        width=2.0,
        edge_color=edge_color,
        arrowsize=20,
        arrowstyle="-|>"
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family="DejaVu Sans",
        font_color=label_color
    )

    plt.title("DSPy Workflow Graph", fontsize=14, pad=15, color="#333333")
    plt.axis("off")
    plt.tight_layout()
    plt.show()