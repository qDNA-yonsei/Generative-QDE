import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


HTML_FILES = [
    "phishing_websites_qubit8_svm_errorbars.html",
    "waveform_database_generator_version_1_qubit8_svm_errorbars.html",
    "pol_qubit8_svm_errorbars.html",
    "drybean_qubit8_svm_errorbars.html",
    "wine_color_qubit8_svm_errorbars.html",
    "wine_qubit8_svm_errorbars.html",
    "magic_gamma_telescope_qubit8_svm_errorbars.html",
    "electrical_grid_stability_simulated_data_qubit8_svm_errorbars.html",
]

DATASET_TOKENS = ["phishing_websites", "waveform_database_generator_version_1", "pol", "drybean", 
                  "wine_color", "wine", "magic_gamma_telescope", "electrical_grid_stability_simulated_data"]

OUTPUT_CSV = "svm_errorbar_combined.csv"


def detect_dataset(path: str) -> str:
    lower = os.path.basename(path).lower()
    for tok in DATASET_TOKENS:
        if tok in lower:
            return tok
    return "unknown"


def detect_qubits(path: str) -> Optional[int]:
    m = re.search(r"qubit(\d+)", os.path.basename(path).lower())
    return int(m.group(1)) if m else None


def _extract_balanced(text: str, start: int, open_ch: str, close_ch: str) -> Tuple[str, int]:
    if start >= len(text) or text[start] != open_ch:
        raise ValueError(f"Expected '{open_ch}' at position {start}")
    depth = 0
    i = start
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1], i + 1
        i += 1
    raise ValueError("Balanced block not found")


def extract_last_plotly_newplot(html_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pos = html_text.rfind("Plotly.newPlot")
    if pos == -1:
        raise ValueError("Plotly.newPlot(...) not found")
    i = html_text.find("(", pos)
    if i == -1:
        raise ValueError("Malformed Plotly.newPlot call")
    i += 1

    # 첫 인자 스킵 → 콤마까지
    in_str = False
    esc = False
    while i < len(html_text):
        ch = html_text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == ",":
                i += 1
                break
        i += 1

    while i < len(html_text) and html_text[i].isspace():
        i += 1
    data_str, j = _extract_balanced(html_text, i, "[", "]")
    i = j

    while i < len(html_text) and (html_text[i].isspace() or html_text[i] == ","):
        i += 1
    layout_str, _ = _extract_balanced(html_text, i, "{", "}")

    data = json.loads(data_str)
    layout = json.loads(layout_str)
    return data, layout


def parse_svm_title_params(layout: Dict[str, Any]) -> Dict[str, Any]:
    out = {"sample_train": None, "sample_test": None, "C": None, "title": None}
    title = None
    try:
        title = layout.get("title", {}).get("text")
    except Exception:
        title = None
    out["title"] = title

    if isinstance(title, str):
        m1 = re.search(r"sample_train\s*=\s*([0-9]+)", title)
        m2 = re.search(r"sample_test\s*=\s*([0-9]+)", title)
        m3 = re.search(r"\bC\s*=\s*([0-9]*\.?[0-9]+)", title)
        out["sample_train"] = int(m1.group(1)) if m1 else None
        out["sample_test"] = int(m2.group(1)) if m2 else None
        out["C"] = float(m3.group(1)) if m3 else None
    return out


def rows_from_svm_errorbar(html_path: str) -> List[Dict[str, Any]]:
    html = Path(f"SVM/{html_path}").read_text(encoding="utf-8", errors="ignore")
    data, layout = extract_last_plotly_newplot(html)

    dataset = detect_dataset(html_path)
    qubits = detect_qubits(html_path)
    title_params = parse_svm_title_params(layout)

    train = None
    test = None
    for tr in data:
        nm = (tr.get("name") or "").lower()
        if "train" in nm:
            train = tr
        elif "test" in nm:
            test = tr

    if train is None or test is None:
        raise ValueError(f"Could not find Train/Test traces in: {html_path}")

    models = train.get("customdata") or []
    y_tr = train.get("y") or []
    s_tr = (train.get("error_y") or {}).get("array") or []

    models2 = test.get("customdata") or []
    y_te = test.get("y") or []
    s_te = (test.get("error_y") or {}).get("array") or []

    if models2 and models and models2 != models:
        te_map = {m: (y, s) for m, y, s in zip(models2, y_te, s_te)}
        rows = []
        for m, y, s in zip(models, y_tr, s_tr):
            y2, s2 = te_map.get(m, (None, None))
            rows.append({
                "dataset": dataset,
                "qubits": qubits,
                "model": m,
                "train_mean": y,
                "train_std": s,
                "test_mean": y2,
                "test_std": s2,
                "sample_train": title_params["sample_train"],
                "sample_test": title_params["sample_test"],
                "C": title_params["C"],
                "title": title_params["title"],
                "html_file": os.path.basename(html_path),
            })
        return rows

    rows: List[Dict[str, Any]] = []
    for m, y1, s1, y2, s2 in zip(models, y_tr, s_tr, y_te, s_te):
        rows.append({
            "dataset": dataset,
            "qubits": qubits,
            "model": m,
            "train_mean": y1,
            "train_std": s1,
            "test_mean": y2,
            "test_std": s2,
            "sample_train": title_params["sample_train"],
            "sample_test": title_params["sample_test"],
            "C": title_params["C"],
            "title": title_params["title"],
            "html_file": os.path.basename(html_path),
        })
    return rows


def main():
    all_rows: List[Dict[str, Any]] = []
    for hp in HTML_FILES:
        all_rows.extend(rows_from_svm_errorbar(hp))

    fieldnames = [
        "dataset","qubits","model",
        "train_mean","train_std","test_mean","test_std",
        "sample_train","sample_test","C","title","html_file"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"[OK] wrote: {OUTPUT_CSV}  (rows={len(all_rows)})")


if __name__ == "__main__":
    main()
