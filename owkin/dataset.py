from pathlib import Path
import numpy as np
import pandas as pd


def find_expo_param(values):
    p0 = (values == 0).mean()
    list_p = np.linspace((9 * p0 + 1) / 10, (p0 + 9) / 10, 100)

    Y = -np.quantile(values, list_p) / np.log(1 - list_p)
    A = np.array([np.ones_like(list_p), 1 / np.log(1 - list_p)]).T

    res = np.linalg.inv(A.T @ A) @ A.T @ Y

    return res[0], res[1]


def build_dataset(
    val_center: str, normalizer_type: str = "None", data_path: str = "../data/"
):
    """
    Build `X_train_mean, X_val_mean, X_train, X_val, y_train, y_val`
    val_center: the name of the center which is used for val set, must be in ["C_1", "C_2", "C_5"]
    normalizer_type: the type of normelizer, must be in ["None", "mean", "lambda_exp"]
    data_path: path the the data from where this function is called
    TODO: add other aggregation that mean (e.g. max) ? Use a dict ?
    """

    # put your own path to the data root directory (see example in `Data architecture` section)
    data_dir = Path(data_path)

    # load the training and testing data sets
    train_features_dir = data_dir / "train_input" / "moco_features"
    test_features_dir = data_dir / "test_input" / "moco_features"
    df_train_val = pd.read_csv(data_dir / "supplementary_data" / "train_metadata.csv")
    df_test = pd.read_csv(data_dir / "supplementary_data" / "test_metadata.csv")

    # concatenate y_train_val and df_train_val
    y_train_val = pd.read_csv(data_dir / "train_output.csv")
    df_train_val = df_train_val.merge(y_train_val, on="Sample ID")

    # retrive train_val data
    X_train_val = []
    y_train_val = []
    centers_train_val = []

    for sample, label, center in df_train_val[
        ["Sample ID", "Target", "Center ID"]
    ].values:
        # load the coordinates and features (1000, 3+2048)
        _features = np.load(train_features_dir / sample)
        # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
        # and the MoCo V2 features
        coordinates, features = _features[:, :3], _features[:, 3:]  # Ks
        X_train_val.append(features)
        y_train_val.append(label)
        centers_train_val.append(center)
    # convert to numpy arrays
    X_train_val = np.array(X_train_val)
    y_train_val = np.array(y_train_val)
    centers_train_val = np.array(centers_train_val)

    # normalize the train_val data
    train_val_centers = ["C_1", "C_2", "C_5"]
    train_centers = train_val_centers.copy()
    train_centers.remove(val_center)

    X_normalized = dict()
    y = dict()
    PATH_DIR = Path(f"data_dir{normalizer_type}/")
    for center in train_val_centers:
        if normalizer_type != "None":
            normalizer = np.load(f"{PATH_DIR}/{center}.npy").astype("float32")
            X_normalized[center] = X_train_val[centers_train_val == center] / normalizer
        else:
            X_normalized[center] = X_train_val[centers_train_val == center]
        y[center] = y_train_val[centers_train_val == center]

    X_train_normalized = np.concatenate(
        [X_normalized[center] for center in train_centers]
    )
    X_train_normalized_mean = X_train_normalized.mean(axis=1)
    y_train = np.concatenate([y[center] for center in train_centers])

    X_val_normalized = X_normalized[val_center]
    X_val_normalized_mean = X_val_normalized.mean(axis=1)
    y_val = y_train_val[centers_train_val == val_center]

    # normalize the test data
    if normalizer_type != "None":
        test_normalizers = dict()
        for center in ["C_3", "C_4"]:
            test_normalizers[center] = np.load(f"{PATH_DIR}/{center}.npy").astype(
                "float32"
            )

    X_test_normalized = []
    for sample, center in df_test[["Sample ID", "Center ID"]].values:
        _features = np.load(test_features_dir / sample)
        coordinates, features = _features[:, :3], _features[:, 3:]
        if normalizer_type != "None":
            X_test_normalized.append(features / test_normalizers[center])
        else:
            X_test_normalized.append(features)

    X_test_normalized = np.array(X_test_normalized)
    X_test_normalized_mean = X_test_normalized.mean(axis=1)

    return (
        X_train_normalized,
        X_train_normalized_mean,
        y_train,
        X_val_normalized,
        X_val_normalized_mean,
        y_val,
        X_test_normalized,
        X_test_normalized_mean,
        df_test,
    )
