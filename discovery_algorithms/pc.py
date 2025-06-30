import networkx as nx
from causallearn.search.ConstraintBased.PC import pc

def run(
    data,
    alpha=0.05,
    test="chisq"
):
    cg = pc(
        data,
        alpha,
        test
    )

    nx_g = cg.to_nx_graph()
    mapping = {i: label for i, label in enumerate(cg.labels)}
    nx_g = nx.relabel_nodes(nx_g, mapping)

    return nx_g
