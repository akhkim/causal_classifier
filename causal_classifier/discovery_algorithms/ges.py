import networkx as nx
from causallearn.search.ScoreBased.GES import ges as ges_algorithm
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
    result = ges_algorithm(df, score_func=score_func)
    
    # Extract the graph from the result dictionary
    if isinstance(result, dict) and 'G' in result:
        cpdag = result['G']
    else:
        cpdag = result
    
    # Collect directed vs. undirected edges
    cols = list(df.columns)
    directed, ambiguous = [], []
    
    # Access edges through the graph's adjacency matrix
    if hasattr(cpdag, 'graph'):
        # cpdag.graph is the adjacency matrix
        adj_matrix = cpdag.graph
        n_vars = adj_matrix.shape[0]
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adj_matrix[i, j] != 0:  # There's an edge
                    a = cols[i]
                    b = cols[j]
                    
                    # Check the type of edge based on adjacency matrix values
                    # In causallearn: 1 = tail, 2 = arrowhead, 3 = circle
                    edge_ij = adj_matrix[i, j]
                    edge_ji = adj_matrix[j, i]
                    
                    if edge_ij == 2 and edge_ji == 1:  # i -> j (arrowhead at j, tail at i)
                        directed.append((a, b))
                    elif edge_ij == 1 and edge_ji == 2:  # j -> i (arrowhead at i, tail at j)
                        directed.append((b, a))
                    elif edge_ij == 1 and edge_ji == 1:  # i - j (undirected)
                        if (b, a) not in ambiguous:  # Avoid duplicates
                            ambiguous.append((a, b))
                    # Other combinations (circles, etc.) treated as ambiguous
                    elif edge_ij != 0 and edge_ji != 0 and (b, a) not in ambiguous and (a, b) not in directed:
                        ambiguous.append((a, b))
    else:
        # Fallback: if no graph attribute, create empty graph
        print("Warning: Could not extract edges from GES result")
        directed, ambiguous = [], []

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
