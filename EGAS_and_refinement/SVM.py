import multiprocessing as mp
import os
from collections import defaultdict
from functools import partial

import numpy as np
import hashlib
import math
import pennylane as qml
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import data_load_and_process
from utils import setup_logger, apply_structure, count_param_gates, set_single_thread_env

logger = setup_logger()
BIAS_GAIN = 10.0


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_projection_matrix(path: str, out_dim: int, in_dim: int) -> np.ndarray:
    A = np.load(path).astype(np.float32, copy=False)
    assert A.shape == (out_dim, in_dim), f"Projection matrix shape mismatch: {A.shape} != {(out_dim, in_dim)}"
    logger.info(f"[proj] A loaded: {path}, sha256={_sha256_of_file(path)[:12]}, shape={A.shape}")
    return A

def project_feat_np(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    z = A @ np.asarray(x, dtype=np.float32).ravel()
    two_pi = 2.0 * math.pi
    return np.remainder(z, two_pi)


def make_overlap_kernel(embedding_fn, num_qubit: int):
    dev = qml.device("lightning.qubit", wires=num_qubit, shots=None)

    @qml.qnode(dev)
    def overlap_circuit(x1, x2):
        wires = list(range(num_qubit))
        embedding_fn(x1, wires=wires)
        qml.adjoint(embedding_fn)(x2, wires=wires)
        return qml.probs(wires=wires)

    def kernel_fn(x1, x2):
        return overlap_circuit(x1, x2)[0]

    return kernel_fn


def fit_eval_svm(X_train, y_train, X_test, y_test, kernel_callable, C: float = 0.1):
    K_train = qml.kernels.kernel_matrix(X_train, X_train, kernel_callable)
    logger.info(f"Mean: {K_train.mean()}, Std: {K_train.std()}")
    clf = SVC(C=C, kernel="precomputed").fit(K_train, y_train)
    train_acc = clf.score(K_train, y_train)
    K_test = qml.kernels.kernel_matrix(X_test, X_train, kernel_callable)
    logger.info(f"Mean: {K_test.mean()}, Std: {K_test.std()}")
    test_acc = clf.score(K_test, y_test)
    return train_acc, test_acc


def load_ckpt(ckpt_path):
    model = torch.load(ckpt_path, map_location="cpu")
    return model["state_dict"], model["arch_info"]


def zz_apply_default(x, wires):
    z = project_feat_np(x, PROJ_A)

    n = len(wires)
    for i in range(n):
        qml.Hadamard(wires=wires[i])
        qml.RZ(z[i], wires=wires[i])
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(2.0 * z[i] * z[j], wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


class BiasNet(nn.Module):
    def __init__(self, num_bias, num_feat):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(num_feat, num_feat * 4), nn.ReLU(),
            nn.Linear(num_feat * 4, num_feat * 2), nn.ReLU(),
        )
        self.head = nn.Linear(num_feat * 2, num_bias, bias=True)

    def forward(self, x):
        z = self.feat(x)
        return self.head(z)


def make_user_embedding_from_ckpt(ckpt_path: str, num_feat: int):
    state, arch = load_ckpt(ckpt_path)
    assert arch["type"] == "structured"
    gate_seq = arch["gate_seq"]
    num_bias = count_param_gates(gate_seq)
    net = build_model(BiasNet, state, num_bias=num_bias, num_feat=num_feat)

    def user_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            raw_b = net(xt)[0].cpu().numpy()
            b = BIAS_GAIN * raw_b
        apply_structure(gate_seq, x, b)

    return user_embedding


def make_user_embedding_from_ckpt_no_bias(ckpt_path: str, num_feat: int):
    _, arch = load_ckpt(ckpt_path)
    assert arch["type"] == "structured"
    gate_seq = arch["gate_seq"]
    num_bias = count_param_gates(gate_seq)
    zero_bias = np.zeros((num_bias,), dtype=float)

    def user_embedding_no_bias(x, wires):
        apply_structure(gate_seq, x, zero_bias)

    return user_embedding_no_bias


class NQE(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(num_feat, num_feat * 2), nn.ReLU(),
            nn.Linear(num_feat * 2, num_feat * 2), nn.ReLU(),
            nn.Linear(num_feat * 2, num_feat),
        )

    def forward(self, x):
        return self.enc(x)


def build_model(net_cls, state_dict: dict, **kwargs):
    net = net_cls(**kwargs)
    with torch.no_grad():
        for k, v in net.state_dict().items():
            v.copy_(state_dict[k])
    net.eval()
    return net


def zz_apply_nqe(n_layers: int, z, wires):
    z2 = project_feat_np(z, PROJ_A)
    
    n = len(wires)
    for _ in range(n_layers):
        for i in range(n):
            qml.Hadamard(wires=wires[i])
            qml.RZ(-2.0 * z2[i], wires=wires[i])
        for i in range(n):
            j = (i + 1) % n
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(-2.0 * ((np.pi - z2[i]) * (np.pi - z2[j])), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


def make_zz_nqe_embedding_from_ckpt(ckpt_path: str, num_feat: int):
    state, arch = load_ckpt(ckpt_path)
    assert arch["type"] == "zz_nqe"
    n_layers = int(arch.get("n_layer", 1))
    net = build_model(NQE, state, num_feat=num_feat)

    def zz_nqe_embedding(x, wires):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).view(1, -1)
            z = net(xt)[0].cpu().numpy()
        zz_apply_nqe(n_layers, z, wires)

    return zz_nqe_embedding


def worker_eval_ckpt(ckpt_file, ckpt_dir, num_qubit, C, X_train, X_test, y_train, y_test):
    set_single_thread_env()

    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    outs = []

    _, arch = load_ckpt(ckpt_path)
    arch_type = arch.get("type", "structured")

    if arch_type == "structured":
        base_label = os.path.splitext(os.path.basename(ckpt_path))[0]

        user_emb_bias = make_user_embedding_from_ckpt(ckpt_path, num_feat=num_feat)
        kernel_bias = make_overlap_kernel(user_emb_bias, num_qubit=num_qubit)
        tr_b, te_b = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_bias, C=C)
        outs.append((f"{base_label}_Bias", float(tr_b), float(te_b)))

        user_emb_plain = make_user_embedding_from_ckpt_no_bias(ckpt_path, num_feat=num_feat)
        kernel_plain = make_overlap_kernel(user_emb_plain, num_qubit=num_qubit)
        tr_p, te_p = fit_eval_svm(X_train, y_train, X_test, y_test, kernel_plain, C=C)
        outs.append((f"{base_label}", float(tr_p), float(te_p)))

    elif arch_type == "zz_nqe":
        zznqe_emb = make_zz_nqe_embedding_from_ckpt(ckpt_path, num_feat=num_feat)
        kernel = make_overlap_kernel(zznqe_emb, num_qubit=num_qubit)
        tr, te = fit_eval_svm(X_train, y_train, X_test, y_test, kernel, C=C)
        outs.append(("zz_nqe", float(tr), float(te)))

    logger.info(f"ckpt: {ckpt_path}, results={outs}")
    return outs


def eval_ckpt_dir(ckpt_dir, num_worker, num_qubit, C, include, X_train, X_test, y_train, y_test):
    files_all = {f for f in os.listdir(ckpt_dir) if f.endswith(".pt")}
    model_types = include['model_type']
    upto = include['upto']
    wanted = []
    for mt in model_types:
        if mt == "zz_nqe":
            wanted.append("zz_nqe.pt")
        else:
            wanted.extend(f"{mt}{i}.pt" for i in range(1, upto + 1))
    files = sorted([f for f in wanted if f in files_all])

    worker = partial(worker_eval_ckpt, ckpt_dir=ckpt_dir, num_qubit=num_qubit, C=C,
                     X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    results = []
    with mp.Pool(processes=num_worker, maxtasksperchild=1) as pool:
        for outs in pool.imap(worker, files, chunksize=1):
            results += outs
    return results


def run_pairwise(models_root, num_worker, num_qubit, num_feat, C, data_name, include, train_len, test_len, slice_start, rbf_gamma):
    subdirs = sorted(os.path.join(models_root, d) for d in os.listdir(models_root))
    n_pairs = len(subdirs)

    logger.info(f"models_root={models_root}, subdirs={n_pairs}, C={C}, workers={num_worker}")

    agg_train = defaultdict(list)
    agg_test = defaultdict(list)

    for i, ckpt_dir in enumerate(subdirs):
        logger.info(f"====== Pair {i + 1}/{n_pairs} ====== dir={ckpt_dir}")

        start_point = slice_start + i * (train_len + test_len)
        X_tr, X_te, y_tr, y_te = data_load_and_process(dataset=data_name, reduction_sz=num_feat,
                                                       train_len=train_len, test_len=test_len, slice_start=start_point)

        kernel_zz = make_overlap_kernel(zz_apply_default, num_qubit=num_qubit)
        zz_train, zz_test = fit_eval_svm(X_tr, y_tr, X_te, y_te, kernel_zz, C=C)
        agg_train["zz"].append(float(zz_train))
        agg_test["zz"].append(float(zz_test))
        logger.info(f"baseline zz: train={zz_train:.4f}, test={zz_test:.4f}")

        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        clf = SVC(C=C, kernel="linear").fit(X_tr_scaled, y_tr)
        tr_cls = clf.score(X_tr_scaled, y_tr)
        te_cls = clf.score(X_te_scaled, y_te)
        agg_train["classical"].append(float(tr_cls))
        agg_test["classical"].append(float(te_cls))
        logger.info(f"classical svm: train={tr_cls:.4f}, test={te_cls:.4f}")

        clf = SVC(C=C, kernel="rbf", gamma=rbf_gamma).fit(X_tr_scaled, y_tr)
        tr_rbf = clf.score(X_tr_scaled, y_tr)
        te_rbf = clf.score(X_te_scaled, y_te)
        agg_train["classical_rbf"].append(float(tr_rbf))
        agg_test["classical_rbf"].append(float(te_rbf))
        logger.info(f"classical rbf svm: train={tr_rbf:.4f}, test={te_rbf:.4f}, gamma={rbf_gamma}")

        res = eval_ckpt_dir(ckpt_dir=ckpt_dir, num_worker=num_worker, num_qubit=num_qubit, C=C, include=include,
                            X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
        for label, tr, te in res:
            agg_train[label].append(tr)
            agg_test[label].append(te)

        logger.info(f"====== End Pair {i}/{n_pairs} ======")

    return agg_train, agg_test


def plot_errorbars(agg_train, agg_test, C, any_train_len, any_test_len, save_path="svm_errorbars.html", rbf_gamma=None):
    labels = sorted(
        agg_test.keys(),
        key=lambda k: (np.mean(agg_test[k]) if len(agg_test[k]) > 0 else 0.0),
        reverse=True
    )

    train_means = [float(np.mean(agg_train[label])) for label in labels]
    train_stds = [float(np.std(agg_train[label], ddof=0)) for label in labels]
    test_means = [float(np.mean(agg_test[label])) for label in labels]
    test_stds = [float(np.std(agg_test[label], ddof=0)) for label in labels]

    xs = np.arange(len(labels), dtype=float)
    offset = 0.18

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs - offset,
        y=train_means,
        mode="markers",
        name="Train (mean ± std)",
        marker=dict(color="blue", size=9),
        error_y=dict(type="data", array=train_stds, visible=True, thickness=1.2),
        hovertemplate="Model: %{customdata}<br>Train mean: %{y:.6f}<br>Train std: %{meta:.6f}<extra></extra>",
        customdata=labels,
        meta=np.array(train_stds),
    ))

    fig.add_trace(go.Scatter(
        x=xs + offset,
        y=test_means,
        mode="markers",
        name="Test (mean ± std)",
        marker=dict(color="red", size=9),
        error_y=dict(type="data", array=test_stds, visible=True, thickness=1.2),
        hovertemplate="Model: %{customdata}<br>Test mean: %{y:.6f}<br>Test std: %{meta:.6f}<extra></extra>",
        customdata=labels,
        meta=np.array(test_stds),
    ))

    title = f"SVM (mean ± std over runs) — sample_train={any_train_len}, sample_test={any_test_len}, C={C}"
    if rbf_gamma is not None:
        title = f"{title}, RBF gamma={rbf_gamma}"

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Model",
            tickmode="array",
            tickvals=xs,
            ticktext=labels,
            tickangle=-45
        ),
        yaxis=dict(title="Accuracy", range=[0.0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=60, b=90),
        template="plotly_white",
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    logger.info(f"[plot] saved: {save_path}")


if __name__ == "__main__":
    mp.set_start_method("fork")
    logger.info("======== START RUN ========")

    data_name = "wine"
    num_qubit = 8
    num_feat = 8
    proj_seed = 42
    dirname = "EGAS_and_refinement"
    proj_path = f"{dirname}/proj_A_{num_qubit}x{num_feat}_seed{proj_seed}_{data_name}.npy"
    PROJ_A = load_projection_matrix(proj_path, out_dim=num_qubit, in_dim=num_feat)

    model_root = f"{dirname}/GQE_{data_name}_qubit{num_qubit}_tune_models"
    output_path = f"{dirname}/{data_name}_qubit{num_qubit}_svm_errorbars.html"

    C = 0.05
    rbf_gamma = 0.125
    num_worker = 10
    include = {"upto": 10, "model_type": ["zz_nqe", "G", "B"]}

    train_len = 400
    test_len = 50
    slice_start = 0

    agg_train, agg_test = run_pairwise(models_root=model_root, num_worker=num_worker, num_qubit=num_qubit, num_feat=num_feat, C=C,
                                       data_name=data_name, include=include,
                                       train_len=train_len, test_len=test_len, slice_start=slice_start,
                                       rbf_gamma=rbf_gamma)

    plot_errorbars(agg_train, agg_test, C=C, any_train_len=train_len, any_test_len=test_len, save_path=output_path,
                   rbf_gamma=rbf_gamma)

    logger.info("======== END RUN ========")
