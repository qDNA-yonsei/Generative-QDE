import os
import re
import json
import math
from typing import Dict
import pandas as pd


MAX_PAIR = 10


def _log_has_ckpt_results(filename: str) -> bool:
    with open(filename, "r") as f:
        for line in f:
            if "results=" in line:
                return True
    return False


def _parse_meta_from_log(filename: str):
    meta_re = re.compile(r"models_root=([^,]+),\s*subdirs=([0-9]+),\s*C=([0-9.]+)")
    models_root = None
    n_pairs = None
    C = None

    with open(filename, "r") as f:
        for line in f:
            m = meta_re.search(line)
            if m:
                models_root = m.group(1)
                n_pairs = int(m.group(2))
                C = float(m.group(3))
                break

    if models_root is None:
        raise ValueError(f"Cannot find models_root/subdirs in log: {filename}")

    base = os.path.basename(models_root.rstrip("/"))
    m2 = re.match(r"GQE_(.+)_qubit(\d+)_tune_models", base)
    if not m2:
        raise ValueError(f"Cannot parse data_name/num_qubit from models_root='{models_root}'")

    data_name = m2.group(1)
    num_qubit = int(m2.group(2))

    return {
        "data_name": data_name,
        "num_qubit": num_qubit,
        "n_pairs": n_pairs,
        "models_root": models_root,
        "C": C,
    }


def _extract_test_acc_from_log_old(filename: str) -> Dict[str, Dict[int, float]]:
    import ast

    pair_re = re.compile(r"Pair\s+(\d+)\s*/\s*(\d+)")
    baseline_re = re.compile(r"baseline zz: train=([0-9.]+), test=([0-9.]+)")
    classical_re = re.compile(r"classical svm: train=([0-9.]+), test=([0-9.]+)")
    classical_rbf_re = re.compile(r"classical rbf svm: train=([0-9.]+), test=([0-9.]+)")
    results_prefix = "results="

    acc_map: Dict[str, Dict[int, float]] = {}
    current_pair = None

    def _set(label: str, p: int, test_acc: float):
        acc_map.setdefault(label, {})[p] = test_acc

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            m = pair_re.search(line)
            if m:
                current_pair = int(m.group(1))
                continue

            if current_pair is None:
                continue

            m = baseline_re.search(line)
            if m:
                _, te = m.groups()
                _set("zz", current_pair, float(te))
                continue

            m = classical_rbf_re.search(line)
            if m:
                _, te = m.groups()
                _set("classical_rbf", current_pair, float(te))
                continue

            m = classical_re.search(line)
            if m:
                _, te = m.groups()
                _set("classical", current_pair, float(te))
                continue

            if results_prefix in line:
                try:
                    _, after = line.split(results_prefix, 1)
                    res_list = ast.literal_eval(after.strip())
                except Exception:
                    continue
                for label, tr, te in res_list:
                    _set(str(label), current_pair, float(te))

    return acc_map


def _extract_classical_from_log_new(filename: str) -> Dict[int, float]:
    pair_re = re.compile(r"Pair\s+(\d+)\s*/\s*(\d+)")
    classical_re = re.compile(r"classical svm: train=([0-9.]+), test=([0-9.]+)")

    acc_by_pair: Dict[int, float] = {}
    current_pair = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            m = pair_re.search(line)
            if m:
                current_pair = int(m.group(1))
                continue
            m = classical_re.search(line)
            if m and current_pair is not None:
                _, te = m.groups()
                acc_by_pair[current_pair] = float(te)

    return acc_by_pair


def _extract_classical_rbf_from_log_new(filename: str) -> Dict[int, float]:
    pair_re = re.compile(r"Pair\s+(\d+)\s*/\s*(\d+)")
    classical_rbf_re = re.compile(r"classical rbf svm: train=([0-9.]+), test=([0-9.]+)")

    acc_by_pair: Dict[int, float] = {}
    current_pair = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            m = pair_re.search(line)
            if m:
                current_pair = int(m.group(1))
                continue
            m = classical_rbf_re.search(line)
            if m and current_pair is not None:
                _, te = m.groups()
                acc_by_pair[current_pair] = float(te)

    return acc_by_pair


def _scan_svm_analysis_under_logs(logs_root: str):
    rows = []

    svm_root = os.path.join(logs_root, "SVM_analysis")
    if not os.path.isdir(svm_root):
        return rows

    for ds_name in os.listdir(svm_root):
        ds_dir = os.path.join(svm_root, ds_name)
        if not os.path.isdir(ds_dir):
            continue

        m = re.match(r"(.+)_qubit(\d+)", ds_name)
        if not m:
            continue
        data_name = m.group(1)
        num_qubit = int(m.group(2))

        for pair_name in os.listdir(ds_dir):
            pair_dir = os.path.join(ds_dir, pair_name)
            if not os.path.isdir(pair_dir):
                continue

            m2 = re.match(r"pair_(\d+)_of_(\d+)", pair_name)
            if not m2:
                continue
            pair_idx = int(m2.group(1))
            if pair_idx > MAX_PAIR:
                continue

            for label in os.listdir(pair_dir):
                stats_path = os.path.join(pair_dir, label, "stats.json")
                if not os.path.isfile(stats_path):
                    continue
                try:
                    with open(stats_path, "r") as f:
                        obj = json.load(f)
                    test_acc = float(obj["test"]["acc"])
                except Exception:
                    continue

                rows.append(
                    {
                        "dataset": data_name,
                        "num_qubit": num_qubit,
                        "pair": pair_idx,
                        "model": label,
                        "test_acc": test_acc,
                        "source": "svm_analysis",
                    }
                )
    return rows


def build_svm_dataframe_unified(logs_root: str = "SVM/logs") -> pd.DataFrame:
    rows = []

    rows.extend(_scan_svm_analysis_under_logs(logs_root))

    for fname in os.listdir(logs_root):
        if not fname.endswith(".log"):
            continue
        log_path = os.path.join(logs_root, fname)

        try:
            meta = _parse_meta_from_log(log_path)
        except Exception:
            continue

        data_name = meta["data_name"]
        num_qubit = meta["num_qubit"]

        if _log_has_ckpt_results(log_path):
            acc_map = _extract_test_acc_from_log_old(log_path)
            for label, pair_dict in acc_map.items():
                for pair_idx, acc in pair_dict.items():
                    if pair_idx > MAX_PAIR:
                        continue
                    rows.append(
                        {
                            "dataset": data_name,
                            "num_qubit": num_qubit,
                            "pair": pair_idx,
                            "model": label,
                            "test_acc": acc,
                            "source": "old_log",
                        }
                    )
        else:
            classical_map = _extract_classical_from_log_new(log_path)
            for pair_idx, acc in classical_map.items():
                if pair_idx > MAX_PAIR:
                    continue
                rows.append(
                    {
                        "dataset": data_name,
                        "num_qubit": num_qubit,
                        "pair": pair_idx,
                        "model": "classical",
                        "test_acc": acc,
                        "source": "new_log_classical",
                    }
                )

            classical_rbf_map = _extract_classical_rbf_from_log_new(log_path)
            for pair_idx, acc in classical_rbf_map.items():
                if pair_idx > MAX_PAIR:
                    continue
                rows.append(
                    {
                        "dataset": data_name,
                        "num_qubit": num_qubit,
                        "pair": pair_idx,
                        "model": "classical_rbf",
                        "test_acc": acc,
                        "source": "new_log_classical_rbf",
                    }
                )

    if not rows:
        raise RuntimeError(f"No SVM results found under logs_root='{logs_root}'")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["dataset", "num_qubit", "pair", "model", "test_acc", "source"]
    )
    df = df[df["pair"] <= MAX_PAIR].copy()
    return df


def save_svm_dataframe_unified(logs_root: str = "SVM/logs",
                               out_csv: str = "svm_df_retrieve_2.csv"):
    df = build_svm_dataframe_unified(logs_root=logs_root)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    save_svm_dataframe_unified(logs_root="SVM/logs",
                           out_csv="svm_log_retrieve.csv")
