import networkx as nx
from causallearn.search.ScoreBased.GES import ges           # GES core[6]
from causallearn.graph.Endpoint import Endpoint

def _label(node, col_names):
    raw = getattr(node, "get_name", lambda: str(node))()
    try:
        idx = int(raw)
        return col_names[idx]
    except (ValueError, IndexError):
        return raw

def run(
    df,
    score_func="local_score_BIC"
):
    cpdag = ges(df, score_func=score_func)

    # Collect directed vs. undirected edges
    cols = list(df.columns)
    directed, ambiguous = [], []
    for e in cpdag.get_edges():
        a = _label(e.get_node1(), cols)
        b = _label(e.get_node2(), cols)
        ea, eb = e.get_endpoint1(), e.get_endpoint2()

        # a  ─►  b
        if ea == Endpoint.TAIL and eb == Endpoint.ARROW:
            directed.append((a, b))
        # b  ─►  a
        elif ea == Endpoint.ARROW and eb == Endpoint.TAIL:
            directed.append((b, a))
        # anything else (TAIL-TAIL or CIRCLE endpoints) is still undirected
        else:
            ambiguous.append((a, b))

    # 3 — build a DAG: add directed edges first
    dag = nx.DiGraph()
    dag.add_nodes_from(cols)
    dag.add_edges_from(directed)

    # 4 — greedily orient remaining undirected edges without creating cycles
    for a, b in ambiguous:
        # pick the alphabetically first direction, flip if that makes a cycle
        if not dag.has_edge(a, b) and not dag.has_edge(b, a):
            candidate = (a, b) if str(a) < str(b) else (b, a)
            dag.add_edge(*candidate)
            if not nx.is_directed_acyclic_graph(dag):
                dag.remove_edge(*candidate)
                dag.add_edge(candidate[1], candidate[0])
                # Skip that edge if it still creates a cycle
                if not nx.is_directed_acyclic_graph(dag):
                    dag.remove_edge(candidate[1], candidate[0])

    return dag
