import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

HTML_FILES = [
    "GQE_phishing_websites_qubit8_tune_arrow.html",
    "GQE_waveform_database_generator_version_1_qubit8_tune_arrow.html",
    "GQE_pol_qubit8_tune_arrow.html",
    "GQE_drybean_qubit8_tune_arrow.html",
    "GQE_wine_color_qubit8_tune_arrow.html",
    "GQE_wine_qubit8_tune_arrow.html",
    "GQE_magic_gamma_telescope_qubit8_tune_arrow.html",
    "GQE_electrical_grid_stability_simulated_data_qubit8_tune_arrow.html",
]

DATASET_TOKENS = ["phishing_websites", "waveform_database_generator_version_1", "pol", "drybean", 
                  "wine_color", "wine", "magic_gamma_telescope", "electrical_grid_stability_simulated_data"]

OUTPUT_CSV = "tune_arrow_combined.csv"


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


def rows_from_arrow(html_path: str) -> List[Dict[str, Any]]:
    html = Path(f"Tune/{html_path}").read_text(encoding="utf-8", errors="ignore")
    data, layout = extract_last_plotly_newplot(html)

    dataset = detect_dataset(html_path)
    qubits = detect_qubits(html_path)
    title = None
    try:
        title = layout.get("title", {}).get("text")
    except Exception:
        title = None

    pat = re.compile(r"^(.*)\s+\((init|final)\)\s*$")

    merged: Dict[str, Dict[str, Any]] = {}
    for tr in data:
        name = tr.get("name", "")
        m = pat.match(name)
        if not m:
            continue
        base = m.group(1).strip()
        stage = m.group(2)

        x = tr.get("x") or []
        y = tr.get("y") or []
        xv = x[0] if len(x) >= 1 else None
        yv = y[0] if len(y) >= 1 else None

        merged.setdefault(base, {})
        merged[base][f"{stage}_x"] = xv
        merged[base][f"{stage}_y"] = yv

    rows: List[Dict[str, Any]] = []
    for base, d in merged.items():
        init_y = d.get("init_y")
        final_y = d.get("final_y")
        delta = (final_y - init_y) if (isinstance(final_y, (int, float)) and isinstance(init_y, (int, float))) else None
        rows.append({
            "dataset": dataset,
            "qubits": qubits,
            "model": base,
            "init_x": d.get("init_x"),
            "init_y": init_y,
            "final_x": d.get("final_x"),
            "final_y": final_y,
            "delta": delta,
            "title": title,
            "html_file": os.path.basename(html_path),
        })

    def sort_key(r):
        m = re.search(r"(\d+)", r["model"])
        return (int(m.group(1)) if m else 10 ** 9, r["model"])

    rows.sort(key=sort_key)

    return rows


def main():
    all_rows: List[Dict[str, Any]] = []
    for hp in HTML_FILES:
        all_rows.extend(rows_from_arrow(hp))

    fieldnames = ["dataset", "qubits", "model", "init_x", "init_y", "final_x", "final_y", "delta", "title", "html_file"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"[OK] wrote: {OUTPUT_CSV}  (rows={len(all_rows)})")


if __name__ == "__main__":
    main()
