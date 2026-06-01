import csv
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import data_load_and_process


datasets = [
    "phishing_websites",
    "waveform_database_generator_version_1",
    "pol",
    "drybean",
    "wine_color",
    "wine",
    "magic_gamma_telescope",
    "electrical_grid_stability",
]

gammas = [2.0 ** k for k in range(-7, 3)]

num_qubit = 8
num_feat = 8
n_pairs = 10
train_len = 400
test_len = 50
slice_start = 0
C = 0.05
cv = 5
random_state = 42

rows = []
scores_by_gamma = defaultdict(list)

for data_name in datasets:
    for pair in range(1, n_pairs + 1):
        start_point = slice_start + (pair - 1) * (train_len + test_len)
        X_train, _, y_train, _ = data_load_and_process(
            dataset=data_name,
            reduction_sz=num_feat,
            train_len=train_len,
            test_len=test_len,
            slice_start=start_point,
        )

        y_train = np.asarray(y_train)
        n_splits = min(cv, min(Counter(y_train).values()))
        if n_splits < 2:
            raise ValueError(f"{data_name} pair {pair}: not enough samples for CV")

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        for gamma in gammas:
            model = make_pipeline(
                StandardScaler(),
                SVC(C=C, kernel="rbf", gamma=gamma),
            )
            scores = cross_val_score(model, X_train, y_train, cv=splitter, scoring="accuracy")
            cv_mean = float(np.mean(scores))

            rows.append(
                {
                    "dataset": data_name,
                    "num_qubit": num_qubit,
                    "pair": pair,
                    "gamma": gamma,
                    "C": C,
                    "cv_mean": cv_mean,
                    "cv_folds": n_splits,
                }
            )
            scores_by_gamma[gamma].append(cv_mean)

with open("rbf_gamma_gridsearch.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["dataset", "num_qubit", "pair", "gamma", "C", "cv_mean", "cv_folds"],
    )
    writer.writeheader()
    writer.writerows(rows)

for gamma in gammas:
    global_cv_mean = float(np.mean(scores_by_gamma[gamma]))
    print(f"gamma={gamma:g}, global_cv_mean={global_cv_mean:.6f}")
