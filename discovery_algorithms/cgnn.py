from itertools import combinations
import numpy as np
import networkx as nx
from cdt.causality.graph import CGNN
from cdt.utils.loss import MMDloss
import torch

# Fit CGNN once and return its bootstrap-mean MMD score
def _cgnn_score(
    df,
    skeleton,
    nh,
    nruns,
    gpus
):
    model = CGNN(nh=nh, nruns=nruns, gpus=gpus)
    _ = model.orient_undirected_graph(df, skeleton)

    # 2. use the score stored by CDT
    if hasattr(model, "score_"):
        return float(model.score_)

    # 3. fallback with CDT’s MMDloss
    Xr = torch.tensor(df.values, dtype=torch.float32)
    Xg = torch.tensor(model.generate(df.shape[0]), dtype=torch.float32)
    if gpus:
        Xr = Xr.cuda()
        Xg = Xg.cuda()

    mmd = MMDloss(input_size=Xr.size(0)).to(Xr.device)
    return float(mmd(Xr, Xg).item()) 

def run(
    df,
    nh="auto",
    nruns="auto"
):
    gpus = 1 if torch.cuda.is_available() else 0
    nh_candidates=(5, 10, 20, 30, 40, 60),
    cv_threshold=0.05,
    max_nruns=128

    skeleton = nx.Graph()
    skeleton.add_nodes_from(df.columns)
    skeleton.add_edges_from(combinations(df.columns, 2))

    # Automatic nh search
    if nh == "auto":
        best_nh, best_mmd = None, np.inf
        for h in nh_candidates:
            score = _cgnn_score(
                df, skeleton, nh=h,
                nruns=max(8, nruns if isinstance(nruns, int) else 8),
                gpus=gpus
            )
            if score < best_mmd - 1e-3:
                best_mmd, best_nh = score, h
            else:
                break
        nh = best_nh

    # Adaptive nruns search
    if nruns=="auto":
        nr = 8
        while True:
            scores = [
                _cgnn_score(df, skeleton, nh=nh, nruns=nr,
                            gpus=gpus)
                for i in range(3)
            ]
            mu, sigma = np.mean(scores), np.std(scores, ddof=1)
            cv = 0 if mu == 0 else sigma / mu
            if cv <= cv_threshold or nr >= max_nruns:
                break
            nr *= 2
        nruns = nr

    model = CGNN(nh=nh, nruns=nruns, gpus=gpus)
    dag = model.orient_undirected_graph(df, skeleton)

    return dag
