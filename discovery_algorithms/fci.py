import networkx as nx
from causallearn.search.ConstraintBased.FCI import fci
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
    alpha=0.05,
    indep_test="kci"
):
    pag, _ = fci(
        df,
        alpha=alpha,
        independence_test_method=indep_test
    )
    dag = nx.DiGraph()
    dag.add_nodes_from(df.columns)

    # PAG -> DAG
    col_names = list(df.columns)
    for edge in pag.get_edges():
        a_obs = _label(edge.get_node1(), col_names)
        b_obs = _label(edge.get_node2(), col_names)
        ep_a = edge.get_endpoint1()
        ep_b = edge.get_endpoint2()

        # Oriented edge  A -> B
        if ep_a == Endpoint.TAIL and ep_b == Endpoint.ARROW:
            dag.add_edge(a_obs, b_obs)

        # Opposite orientation  B <- A
        elif ep_a == Endpoint.ARROW and ep_b == Endpoint.TAIL:
            dag.add_edge(b_obs, a_obs)

        # Bidirected edge  A <-> B  ==> latent confounder
        elif ep_a == Endpoint.ARROW and ep_b == Endpoint.ARROW:
            latent = f"U_{min(a_obs,b_obs)}_{max(a_obs,b_obs)}"
            if latent not in dag:
                dag.add_node(latent, latent=True)
            dag.add_edge(latent, a_obs)
            dag.add_edge(latent, b_obs)
        else:
            continue

    # Remove cycle-creating edges by removing the last edge
    if not nx.is_directed_acyclic_graph(dag):
        for cycle in list(nx.simple_cycles(dag)):
            dag.remove_edge(cycle[-1], cycle[0])

    return dag
