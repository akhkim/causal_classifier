import networkx as nx

def is_d_separated(G, X, Y, Z):
    return nx.d_separated(G, X, Y, Z)

def remove_incoming_edges(G, nodes):
    G_mod = G.copy()
    for node in nodes:
        G_mod.remove_edges_from(list(G.in_edges(node)))
    return G_mod

def do_calculus_rule1(G, Y, X, Z, W):
    """
    Rule 1: If Y ⫫ Z | X, W in G_X (where incoming edges to X are removed),
    then P(Y | do(X), Z, W) = P(Y | do(X), W)
    """
    G_X = remove_incoming_edges(G, X)
    if is_d_separated(G_X, set(Y), set(Z), set(X) | set(W)):
        return True
    return False

def do_calculus_rule2(G, Y, X, Z, W):
    """
    Rule 2: If Y ⫫ Z | X, W in G_{X,Z}, then
    P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)
    """
    G_XZ = remove_incoming_edges(G, X + Z)
    if is_d_separated(G_XZ, set(Y), set(Z), set(X) | set(W)):
        return True
    return False

def do_calculus_rule3(G, Y, X, Z, W):
    """
    Rule 3: If Y ⫫ Z | X, W in G_{X,Z}^{remove(Z->Y)},
    then P(Y | do(X), do(Z), W) = P(Y | do(X), W)
    """
    G_mod = remove_incoming_edges(G, X)
    for z in Z:
        for y in Y:
            if G_mod.has_edge(z, y):
                G_mod.remove_edge(z, y)

    if is_d_separated(G_mod, set(Y), set(Z), set(X) | set(W)):
        return True
    return False