import openml as oml
import torch
from pennylane import numpy as pnp
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo


def _data_getter(data_name):
    uci_id = {
        "magic_gamma_telescope": 159,
        "phishing_websites": 327,
        "electrical_grid_stability": 471,
        "drybean": 602,
        "wine": 186,
        "wine_color": 186,
        "waveform_database_generator_version_1": 107,
    }

    oml_id = {
        "pol": 44122,
    }

    data_registry = {**{k: ("uci", v) for k, v in uci_id.items()}, **{k: ("oml", v) for k, v in oml_id.items()}}

    key = str(data_name).strip()
    try:
        source, did = data_registry[key]
    except KeyError:
        raise ValueError(f"Unsupported data_name: {data_name}")

    if source == "uci":
        data = fetch_ucirepo(id=did)
        if key == "wine_color":
            data.data.targets = data.data.original[["color"]]

    if source == "oml":
        data = oml.datasets.get_dataset(did)

    return data, source


def data_load_and_process(dataset, reduction_sz, train_len, test_len, slice_start=0):
    data, source = _data_getter(dataset)

    if source == "oml":
        X, y, _, _ = data.get_data(dataset_format="dataframe", target=data.default_target_attribute, )

    if source == "uci":
        X, y = data.data.features, data.data.targets

    if dataset == "magic_gamma_telescope":
        shuffled = X.join(y).sample(frac=1, random_state=42).reset_index(drop=True)
        X = shuffled[X.columns]
        y = shuffled[y.columns]
    if dataset == "electrical_grid_stability":
        y = y['stabf']
    if dataset == "drybean":
        y.reset_index(drop=True, inplace=True)
        X.reset_index(drop=True, inplace=True)
        select = y.index[y['Class'].isin(['DERMASON', 'SIRA'])]
        X = X.loc[select]
        y = y.loc[select]

        shuffled = X.join(y).sample(frac=1, random_state=42).reset_index(drop=True)
        X = shuffled[X.columns]
        y = shuffled[y.columns]
    if dataset == "pol":
        y = y.to_frame()
        shuffled = X.join(y).sample(frac=1, random_state=42).reset_index(drop=True)
        X = shuffled[X.columns]
        y = shuffled[y.columns]
    if dataset == "wine":
        y.reset_index(drop=True, inplace=True)
        X.reset_index(drop=True, inplace=True)
        select = y.index[y['quality'].isin([6, 5])]
        X = X.loc[select]
        y = y.loc[select]
    if dataset == "wine_color":
        X = X[:5001]
        y = y[:5001]
        shuffled = X.join(y).sample(frac=1, random_state=42).reset_index(drop=True)
        X = shuffled[X.columns]
        y = shuffled[y.columns]

    X = X.to_numpy(dtype=float)
    y = y.to_numpy().ravel()

    uniq = pnp.unique(y)
    y = pnp.where(y == uniq[0], 0, 1)

    assert len(pnp.unique(y)) == 2, f"Expected 2 unique, but got {pnp.unique(y)}"

    x_tr, x_te = X[slice_start:slice_start + train_len], X[slice_start + train_len:slice_start + train_len + test_len]
    y_tr, y_te = y[slice_start:slice_start + train_len], y[slice_start + train_len:slice_start + train_len + test_len]

    X_train = PCA(reduction_sz).fit_transform(x_tr)
    X_test = PCA(reduction_sz).fit_transform(x_te)

    def normalize_to_2pi(arr):
        arr_normed = []
        for x in arr:
            x = (x - x.min()) * (2 * pnp.pi / (x.max() - x.min()))
            arr_normed.append(x)
        return pnp.array(arr_normed)

    X_train = normalize_to_2pi(X_train)
    X_test = normalize_to_2pi(X_test)

    return X_train, X_test, y_tr, y_te


def new_data(batch_sz, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    data_store = {}
    data_store_raw_X = []
    data_store_raw_Y = []
    for i in range(batch_sz):
        n, m = pnp.random.randint(len(X)), pnp.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        Y_new.append(1 if Y[n] == Y[m] else 0)
        data_store_raw_X.append(X[n])
        data_store_raw_X.append(X[m])
        data_store_raw_Y.append(Y[n])
        data_store_raw_Y.append(Y[m])

    X1_new_array = pnp.array(X1_new)
    X1_new_tensor = torch.from_numpy(X1_new_array).float()

    X2_new_array = pnp.array(X2_new)
    X2_new_tensor = torch.from_numpy(X2_new_array).float()

    Y_new_array = pnp.array(Y_new)
    Y_new_tensor = torch.from_numpy(Y_new_array).float()

    data_store['raw_X'] = data_store_raw_X
    data_store['raw_Y'] = data_store_raw_Y
    data_store['processed'] = [X1_new_tensor, X2_new_tensor, Y_new_tensor]
    return X1_new_tensor, X2_new_tensor, Y_new_tensor, data_store
