import pandas as pd
import networkx as nx
from lingam import DirectLiNGAM

def run(
    df: pd.DataFrame,
    max_iter: int = 1000
):
    model = DirectLiNGAM(max_iter=max_iter)
    model.fit(df.values)

    # adjacency_matrix_[i,j] = causal effect i → j
    adj = model.adjacency_matrix_

    dag = nx.DiGraph()
    dag.add_nodes_from(df.columns)

    for i, src in enumerate(df.columns):
        for j, tgt in enumerate(df.columns):
            w = adj[i, j]   # weight of the edge src → tgt
            if w != 0:
                dag.add_edge(src, tgt, weight=float(w))

    return dag
