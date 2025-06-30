from notears.linear import notears_linear
import networkx as nx

def run(
    df,
    lambda1=0.01,
    max_iter=100,
    w_threshold=0.3
):
    X = df.to_numpy(dtype=float)

    # run linear NOTEARS
    W = notears_linear(
        X,
        lambda1=lambda1,
        max_iter=max_iter
    )  # W_{i,j} = effect i → j  [1]

    dag = nx.DiGraph()
    dag.add_nodes_from(df.columns)

    for i, src in enumerate(df.columns):
        for j, tgt in enumerate(df.columns):
            w = W[i, j]
            if abs(w) > w_threshold:
                dag.add_edge(src, tgt, weight=float(w))

    return dag
