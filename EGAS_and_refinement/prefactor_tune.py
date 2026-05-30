import json
import os
import pickle
import random
import time
from datetime import datetime
from multiprocessing import get_context

import pandas as pd
import pennylane as qml
import plotly.graph_objects as go
import torch
from numpy import pi
from pennylane import numpy as pnp
import numpy as np
import hashlib
import math

from torch import nn, optim

from utils import setup_logger, apply_structure, count_param_gates, set_single_thread_env

logger = setup_logger()

BIAS_GAIN = 10.0
P_EPS = 1e-6
GRAD_CLIP_NORM = 2.0
BIAS_L2_LAMBDA = 1e-6


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_projection_matrix(path: str, out_dim: int, in_dim: int, seed: int = 42, dtype=np.float32):
    if os.path.exists(path):
        A = np.load(path).astype(dtype, copy=False)
    else:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((out_dim, in_dim), dtype=dtype)
        A = A / np.sqrt(in_dim)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, A)
    assert A.shape == (out_dim, in_dim), f"Projection matrix shape mismatch: {A.shape} != {(out_dim, in_dim)}"
    logger.info(f"[proj] A loaded: {path}, sha256={_sha256_of_file(path)[:12]}, shape={A.shape}, seed={seed}")
    return A


def project_feat_torch(x: torch.Tensor, A_torch: torch.Tensor) -> torch.Tensor:
    z = torch.matmul(A_torch, x)  # (out_dim,)
    two_pi = 2.0 * math.pi
    return torch.remainder(z, two_pi)


def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for _ in range(batch_size):
        n, m = pnp.random.randint(len(X)), pnp.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)

    X1_new = torch.tensor(pnp.array(X1_new), dtype=torch.float32)
    X2_new = torch.tensor(pnp.array(X2_new), dtype=torch.float32)
    Y_new = torch.tensor(pnp.array(Y_new), dtype=torch.float32)

    return X1_new, X2_new, Y_new


def get_circuit(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df['aveTrueE'] = df.groupby('GeneratedIteration')['energy'].transform('mean')
    return df


def get_data(filename):
    with open(filename, "rb") as f:
        df = pickle.load(f)
    raw_X, raw_Y, processed_data = df['raw_X'], df['raw_Y'], df['processed']
    return raw_X, raw_Y, processed_data


def get_circuit_by_energy(data, top_or_bottom, n_circuit):
    logger.info(f'extracting {top_or_bottom}...')
    top_or_bottom = top_or_bottom == 'top'
    selected_df = data.sort_values(by='energy', ascending=top_or_bottom)[:1000]
    rm_duple = selected_df.drop_duplicates(subset=["gen_op_seq"])
    return rm_duple[:n_circuit]


def make_random_circuit(gate_type, max_gate, num_qubit, num_feat, scales):
    circuit = []
    for _ in range(max_gate):
        gate = random.choice(gate_type)

        if gate in ['H', 'I']:
            target = random.randint(0, num_qubit - 1)
            circuit.append([gate, None, [target, None]])

        elif gate == 'CNOT':
            control, target = random.sample(range(num_qubit), 2)
            circuit.append([gate, None, [control, target]])

        elif gate in ['RX', 'RY', 'RZ', 'MultiRZ']:
            param_idx = random.randint(0, num_feat - 1)
            control, target = random.sample(range(num_qubit), 2)
            scale = random.choice(scales)
            param_idx = [param_idx, scale]
            if gate == 'MultiRZ':
                circuit.append([gate, param_idx, [control, target]])
            else:
                circuit.append([gate, param_idx, [target, None]])

    return circuit


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_final_model(save_path, name, model, arch_info, losses, metric, epoch, ave_len):
    payload = {"model_name": name,
               "state_dict": model.state_dict(),
               "arch_info": arch_info,
               "loss_trace": losses,
               "final_metric": float(metric),
               "epoch": int(epoch),
               "averaging_length": int(ave_len),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
    torch.save(payload, save_path)


def get_color(key):
    if key.startswith("G"):
        return "#0D28BF"
    elif key.startswith("B"):
        return "red"
    elif key.startswith("R"):
        return "#1487B5"
    elif key == "zz":
        return "black"
    elif key == "zz_nqe":
        return "#2E7D32"
    else:
        return "gray"


def plot_energy_errorbars(energy_list, html_path="energy_errorbar.html", filter_keys=None):
    df = pd.DataFrame(energy_list)

    if filter_keys is not None:
        df = df[filter_keys]

    mean_vals = df.mean()
    std_vals = df.std()

    fig = go.Figure()

    for key in df.columns:
        color = get_color(key)
        fig.add_trace(go.Scatter(
            x=[key],
            y=[mean_vals[key]],
            mode='markers',
            marker=dict(color=color, size=6),
            error_y=dict(type='data', array=[std_vals[key]], color=color, thickness=1.5),
            name=key,
            showlegend=True
        ))

    fig.update_layout(
        title="Prefactored Energy per Circuit (Mean ± Std)",
        xaxis_title="Circuit",
        yaxis_title="Energy",
        xaxis_tickangle=-90,
        template="plotly_white",
        showlegend=True,
    )

    fig.write_html(html_path)
    logger.info(f"graph save: {html_path}")


def plot_epoch_trajectories(trace_dict, html_path="epoch_trajectories.html",
                            title="Energy vs Epoch (per model)", filter_keys=None):
    fig = go.Figure()
    keys = list(trace_dict.keys()) if filter_keys is None else filter_keys
    for name in keys:
        ys = trace_dict[name]
        xs = list(range(1, len(ys) + 1))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Energy",
        template="plotly_white",
        showlegend=True,
    )
    fig.write_html(html_path)
    logger.info(f"trajectory graph save: {html_path}")


def plot_initial_final_arrows(trace_dict, html_path="init_final_arrows.html",
                              ave_len=1, filter_keys=None,
                              title="Init → Final Energy (per model)"):
    fig = go.Figure()
    keys = list(trace_dict.keys()) if filter_keys is None else filter_keys
    xs = list(range(len(keys)))

    fig.update_xaxes(tickmode="array", tickvals=xs, ticktext=keys)

    for i, name in enumerate(keys):
        ys = trace_dict[name]
        if len(ys) == 0:
            continue

        y0 = float(ys[0])
        L = max(1, min(ave_len, len(ys)))
        y1 = float(pnp.mean(ys[-L:]))

        base = get_color(name)

        fig.add_trace(go.Scatter(
            x=[xs[i]], y=[y0], mode="markers",
            marker=dict(symbol="x", size=7, line=dict(width=1, color=base), color=base),
            name=f"{name} (init)"
        ))
        fig.add_trace(go.Scatter(
            x=[xs[i]], y=[y1], mode="markers",
            marker=dict(symbol="circle", size=5, color=base),
            name=f"{name} (final)"
        ))
        fig.add_annotation(
            x=xs[i], y=y1, ax=xs[i], ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowwidth=1.5,
            arrowcolor=base, opacity=0.6
        )

    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Energy",
        template="plotly_white",
        showlegend=True,
    )
    fig.write_html(html_path)
    logger.info(f"graph save: {html_path}")



def run_multiple_compare(n_repeat, num_workers, **kwargs):
    task_args = [kwargs.copy() for _ in range(n_repeat)]

    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        results = list(pool.imap(run_compare_wrapper, task_args, chunksize=1))
    return results


def run_compare_wrapper(args):
    logger = setup_logger()
    torch.set_num_threads(1)
    seed = int(time.time() * 1e6) % (2 ** 32 - 1) + os.getpid()
    random.seed(seed)
    pnp.random.seed(seed)
    torch.manual_seed(seed)
    proj_path = args.pop("proj_path")
    proj_seed = int(args.pop("proj_seed", 42))
    num_qubit = int(args["num_qubit"])
    num_feat = int(args["num_feat"])
    A_np = ensure_projection_matrix(proj_path, out_dim=num_qubit, in_dim=num_feat, seed=proj_seed)
    A_torch = torch.tensor(A_np, dtype=torch.float32)  # CPU 고정
    args["A_torch"] = A_torch

    logger.info(f"[Worker PID {os.getpid()}] Using seed: {seed}")
    
    save_root = args.pop("save_root", None)
    worker_dir = None
    if save_root is not None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        worker_dir = os.path.join(save_root, f"worker_pid{os.getpid()}_{run_tag}")
        ensure_dir(worker_dir)
        logger.info(f"[Worker PID {os.getpid()}] save dir: {worker_dir}")
    args["save_dir"] = worker_dir
    return run_compare(**args)


def run_compare(data_x, data_y, num_qubit, num_feat, A_torch, n_layer, batch_size, epoch, 
                good_circuits, bad_circuits, rand_circuits, ave_len, save_dir=None):
    logger.info("Running comparison...")
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    opts = {}
    nets = {}
    qnodes = {}
    arch_infos = {}
    
    for i in range(len(good_circuits)):
        gate_seq = good_circuits.iloc[i]['gen_op_seq']
        num_of_bias = count_param_gates(gate_seq)
        name = f"G{i + 1}"

        nets[name] = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_feat)
        opts[name] = optim.RMSprop(nets[name].parameters(), lr=0.0005, alpha=0.99, eps=1e-6, weight_decay=0.0)
        qnodes[name] = build_qnode_with_bias(num_qubit, gate_seq, num_feat=num_feat)
        arch_infos[name] = {"type": "structured", "num_qubit": num_qubit, "num_feat": num_feat, "gate_seq": gate_seq}

    for i in range(len(bad_circuits)):
        gate_seq = bad_circuits.iloc[i]['gen_op_seq']
        num_of_bias = count_param_gates(gate_seq)
        name = f"B{i + 1}"

        nets[name] = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_feat)
        opts[name] = optim.RMSprop(nets[name].parameters(), lr=0.0005, alpha=0.99, eps=1e-6, weight_decay=0.0)
        qnodes[name] = build_qnode_with_bias(num_qubit, gate_seq, num_feat=num_feat)
        arch_infos[name] = {"type": "structured", "num_qubit": num_qubit, "num_feat": num_feat, "gate_seq": gate_seq}

    if rand_circuits is not None:
        for i in range(len(rand_circuits)):
            gate_seq = rand_circuits[i]
            num_of_bias = count_param_gates(gate_seq)
            name = f"R{i + 1}"

            nets[name] = BiasNet(num_of_bias=num_of_bias, num_of_feature=num_feat)
            opts[name] = optim.RMSprop(nets[name].parameters(), lr=0.0005, alpha=0.99, eps=1e-6, weight_decay=0.0)
            qnodes[name] = build_qnode_with_bias(num_qubit, gate_seq, num_feat=num_feat)
            arch_infos[name] = {"type": "structured", "num_qubit": num_qubit, "num_feat": num_feat, "gate_seq": gate_seq}

    zz_bias_num = count_param_gates_zz(num_qubit, n_layer)
    nets["zz"] = BiasNet(num_of_bias=zz_bias_num, num_of_feature=num_feat)
    opts["zz"] = optim.RMSprop(nets["zz"].parameters(), lr=0.0005, alpha=0.99, eps=1e-6, weight_decay=0.0)
    qnodes["zz"] = build_zz_qnode_with_bias(num_qubit, n_layer, num_feat=num_feat, A_torch=A_torch)
    arch_infos["zz"] = {"type": "zz_bias", "num_qubit": num_qubit, "num_feat": num_feat, "n_layer": n_layer}

    nets["zz_nqe"] = EncoderNet(num_of_feature=num_feat)
    opts["zz_nqe"] = optim.RMSprop(nets["zz_nqe"].parameters(), lr=0.0005, alpha=0.99, eps=1e-6, weight_decay=0.0)
    qnodes["zz_nqe"] = build_zz_qnode_nqe(num_qubit, n_layer, num_feat=num_feat, A_torch=A_torch)
    arch_infos["zz_nqe"] = {"type": "zz_nqe", "num_qubit": num_qubit, "num_feat": num_feat, "n_layer": n_layer}

    loss_lists = {name: [] for name in nets.keys()}

    loss_fn = torch.nn.BCELoss()

    for it in range(epoch):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, data_x, data_y)
        logger.info(f"Epoch {it + 1}/{epoch}...")
        for name, model in nets.items():
            opts[name].zero_grad()
            if name == "zz_nqe":
                packed = nets[name](X1_batch, X2_batch)
                preds = []
                for i in range(batch_size):
                    probs = qnodes[name](packed[i])
                    preds.append(probs[0])
                preds = torch.stack(preds).float()

                preds = preds.clamp(P_EPS, 1.0 - P_EPS)
                bce_loss = loss_fn(preds, Y_batch)
                mse_metric = ((preds - Y_batch) ** 2).mean()

                bce_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

                opts[name].step()
                loss_lists[name].append(float(mse_metric.item()))
            else:
                bias = nets[name](X1_batch, X2_batch)
                bias = BIAS_GAIN * bias

                preds = []
                for i in range(batch_size):
                    packed = torch.cat([X1_batch[i], X2_batch[i], bias[i]], dim=0)
                    probs = qnodes[name](packed)
                    preds.append(probs[0])
                preds = torch.stack(preds).float()
                preds = preds.clamp(P_EPS, 1.0 - P_EPS)

                bias_l2 = (bias ** 2).sum(dim=1).mean()
                bce_loss = loss_fn(preds, Y_batch) + BIAS_L2_LAMBDA * bias_l2

                mse_metric = ((preds - Y_batch) ** 2).mean()

                bce_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

                opts[name].step()
                loss_lists[name].append(float(mse_metric.item()))

    final_energy = {name: pnp.mean(energy[-ave_len:]) for name, energy in loss_lists.items()}
    
    if save_dir is not None:
        ensure_dir(save_dir)
        for name, model in nets.items():
            metric = final_energy[name]
            fpath = os.path.join(save_dir, f"{name}.pt")
            save_final_model(save_path=fpath, 
                             name=name, 
                             model=model, 
                             arch_info=arch_infos[name], 
                             losses=loss_lists[name],
                             metric=metric,
                             epoch=epoch,
                             ave_len=ave_len)
            logger.info(f"[{name}] saved at {fpath} (metric={metric:.6f})")

    return {"final": final_energy, "trace": loss_lists}


class BiasNet(nn.Module):
    def __init__(self, num_of_bias, num_of_feature):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(num_of_feature, num_of_feature * 4), nn.ReLU(),
            nn.Linear(num_of_feature * 4, num_of_feature * 2), nn.ReLU(),
        )
        self.head = nn.Linear(num_of_feature * 2, num_of_bias, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x1, x2):
        x1 = self.feat(x1)
        x2 = self.feat(x2)
        b1 = self.head(x1)
        b2 = self.head(x2)
        return torch.cat([b1, b2], dim=1)


def build_qnode_with_bias(num_qubit, gate_seq, num_feat):
    dev = qml.device("lightning.qubit", wires=num_qubit)

    @qml.qnode(dev, interface="torch")
    def qnode(packed):
        x1 = packed[:num_feat]
        x2 = packed[num_feat:num_feat * 2]
        bcat = packed[num_feat * 2:]
        L = bcat.shape[0] // 2
        b1 = bcat[:L]
        b2 = bcat[L:]

        apply_structure(gate_seq, x1, b1)
        qml.adjoint(lambda: apply_structure(gate_seq, x2, b2))()
        return qml.probs(wires=range(num_qubit))

    return qnode


def count_param_gates_zz(num_qubit, n_layers):
    return 2 * num_qubit * n_layers


def apply_zz_with_bias(num_qubit, n_layers, x, bias, A_torch):
    z = project_feat_torch(x, A_torch)

    bias_count = 0
    for _ in range(n_layers):
        for j in range(num_qubit):
            qml.Hadamard(wires=j)
            theta = -2.0 * z[j] + bias[bias_count]
            bias_count += 1
            qml.RZ(theta, wires=j)

        for k in range(num_qubit):
            t = (k + 1) % num_qubit
            qml.CNOT(wires=[k, t])
            theta = -2.0 * ((pi - z[k]) * (pi - z[t])) + bias[bias_count]
            bias_count += 1
            qml.RZ(theta, wires=t)
            qml.CNOT(wires=[k, t])

    assert bias_count == len(bias), "zz bias length mismatch"


def build_zz_qnode_with_bias(num_qubit, n_layers, num_feat, A_torch):
    dev = qml.device("lightning.qubit", wires=num_qubit)

    @qml.qnode(dev, interface="torch")
    def qnode(packed):
        x1 = packed[:num_feat]
        x2 = packed[num_feat:num_feat * 2]
        bcat = packed[num_feat * 2:]
        L = bcat.shape[0] // 2
        b1 = bcat[:L]
        b2 = bcat[L:]

        apply_zz_with_bias(num_qubit, n_layers, x1, b1, A_torch)
        qml.adjoint(lambda: apply_zz_with_bias(num_qubit, n_layers, x2, b2, A_torch))()
        return qml.probs(wires=range(num_qubit))

    return qnode


class EncoderNet(nn.Module):
    def __init__(self, num_of_feature):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(num_of_feature, num_of_feature * 2), nn.ReLU(),
            nn.Linear(num_of_feature * 2, num_of_feature * 2), nn.ReLU(),
            nn.Linear(num_of_feature * 2, num_of_feature),
        )

    def forward(self, x1, x2):
        z1 = self.enc(x1)
        z2 = self.enc(x2)
        return torch.cat([z1, z2], dim=1)


def apply_zz_no_bias(num_qubit, n_layers, x, A_torch):
    z = project_feat_torch(x, A_torch)

    for _ in range(n_layers):
        for j in range(num_qubit):
            qml.Hadamard(wires=j)
            qml.RZ(-2.0 * z[j], wires=j)
        for k in range(num_qubit):
            t = (k + 1) % num_qubit
            qml.CNOT(wires=[k, t])
            qml.RZ(-2.0 * ((pi - z[k]) * (pi - z[t])), wires=t)
            qml.CNOT(wires=[k, t])


def build_zz_qnode_nqe(num_qubit, n_layers, num_feat, A_torch):
    dev = qml.device("lightning.qubit", wires=num_qubit)

    @qml.qnode(dev, interface="torch")
    def qnode(packed):
        z1 = packed[:num_feat]
        z2 = packed[num_feat:num_feat * 2]
        apply_zz_no_bias(num_qubit, n_layers, z1, A_torch)
        qml.adjoint(lambda: apply_zz_no_bias(num_qubit, n_layers, z2, A_torch))()
        return qml.probs(wires=range(num_qubit))

    return qnode


if __name__ == "__main__":
    set_single_thread_env()
    logger.info("Starting prefactor tune...")

    data_name = "twonorm"
    num_qubit = 8
    num_feat = 8
    proj_seed = 42

    dirname = "EGAS_and_refinement"
    circuit_filename = f'{dirname}/GQE_{data_name}_qubit{num_qubit}_generated_circuit.json'
    data_filename = f'{dirname}/GQE_{data_name}_qubit{num_qubit}_data_store.pkl'
    html_filename = f"{dirname}/GQE_{data_name}_qubit{num_qubit}_tune"
    save_root = ensure_dir(f"{html_filename}_models")
    proj_path = f"{dirname}/proj_A_{num_qubit}x{num_feat}_seed{proj_seed}_{data_name}.npy"
    _ = ensure_projection_matrix(proj_path, out_dim=num_qubit, in_dim=num_feat, seed=proj_seed)
    

    batch_size = 25
    n_layer = 1
    n_circuit = 10  # 30
    epoch = 400  # 100
    averaging_length = 10  # 10
    num_cpus = 10  # 16
    repeat = num_cpus

    gate_type = ['RX', 'RY', 'RZ', 'CNOT', 'MultiRZ', 'H', 'I']
    max_gate = 28
    scales = [0.1, 0.3, 0.5, 0.7, 1]

    circuits = get_circuit(circuit_filename)
    data_x, data_y, _ = get_data(data_filename)

    good_circuits = get_circuit_by_energy(circuits, 'top', n_circuit=n_circuit)
    bad_circuits = get_circuit_by_energy(circuits, 'bottom', n_circuit=n_circuit)

    logger.info('Tune begin...')

    results = run_multiple_compare(n_repeat=repeat,
                                   num_workers=num_cpus,
                                   data_x=data_x,
                                   data_y=data_y,
                                   num_qubit=num_qubit,
                                   num_feat=num_feat,
                                   proj_path=proj_path,
                                   proj_seed=proj_seed,
                                   n_layer=n_layer,
                                   batch_size=batch_size,
                                   epoch=epoch,
                                   good_circuits=good_circuits,
                                   bad_circuits=bad_circuits,
                                   rand_circuits=None,
                                   ave_len=averaging_length,
                                   save_root=save_root)

    logger.info('Tune finished...')

    energy_list = [res["final"] for res in results]
    plot_energy_errorbars(energy_list, html_path=f"{html_filename}_errorbar.html")

    trace_repeat_idx = 0
    plot_epoch_trajectories(results[trace_repeat_idx]["trace"],
                            html_path=f"{html_filename}_trajectory.html", title=f"Energy vs Epoch")
    plot_initial_final_arrows(results[trace_repeat_idx]["trace"], html_path=f"{html_filename}_arrow.html",
                              ave_len=averaging_length, title=f"Init(X) → Final(O) Energy")
