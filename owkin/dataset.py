from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


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
    val_center: the name of the center which is used for val set, must be in ["C_1", "C_2", "C_5", "None"]
    (if, we build the val set of the baseline)
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

    # get all the normalizers
    normalizer_dir = Path(f"{data_path}/normalizer/{normalizer_type}/")
    normalizers = dict()

    for center in ["C_1", "C_2", "C_5", "C_3", "C_4"]:
        if normalizer_type != "None":
            normalizers[center] = np.load(f"{normalizer_dir}/{center}.npy").astype(
                "float32"
            )
        else:
            normalizers[center] = 1

    # retrive train_val data
    X_train_val = []
    y_train_val = []
    centers_train_val = []
    patients_train_val = []

    for sample, label, center, patient in df_train_val[
        ["Sample ID", "Target", "Center ID", "Patient ID"]
    ].values:
        # load the coordinates and features (1000, 3+2048)
        _features = np.load(train_features_dir / sample)
        # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
        # and the MoCo V2 features
        coordinates, features = _features[:, :3], _features[:, 3:]  # Ks
        X_train_val.append(features / normalizers[center])
        y_train_val.append(label)
        centers_train_val.append(center)
        patients_train_val.append(patient)

    # convert to numpy arrays
    X_train_val = np.array(X_train_val)
    y_train_val = np.array(y_train_val)
    centers_train_val = np.array(centers_train_val)
    patients_train_val = np.array(patients_train_val)

    # split by centers
    if val_center != "None":
        X = dict()
        y = dict()

        # Split train and val sets
        train_val_centers = ["C_1", "C_2", "C_5"]
        train_centers = train_val_centers.copy()
        train_centers.remove(val_center)
        for center in train_val_centers:
            X[center] = X_train_val[centers_train_val == center]
            y[center] = y_train_val[centers_train_val == center]

        X_train = np.concatenate([X[center] for center in train_centers])
        y_train = np.concatenate([y[center] for center in train_centers])

        X_val = X[val_center]
        y_val = y_train_val[centers_train_val == val_center]

    # split like in baseline
    else:
        patients_unique = np.unique(patients_train_val)
        y_unique = np.array(
            [np.mean(y_train_val[patients_train_val == p]) for p in patients_unique]
        )

        kfold = StratifiedKFold(5, shuffle=True, random_state=42)
        # split is performed at the patient-level
        for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
            # retrieve the indexes of the samples corresponding to the
            # patients in `train_idx_` and `val_idx_`
            train_idx = np.arange(len(X_train_val))[
                pd.Series(patients_train_val).isin(patients_unique[train_idx_])
            ]
            val_idx = np.arange(len(X_train_val))[
                pd.Series(patients_train_val).isin(patients_unique[val_idx_])
            ]
            # set the training and validation folds
            X_train = X_train_val[train_idx]
            X_val = X_train_val[val_idx]

            y_train = y_train_val[train_idx]
            y_val = y_train_val[val_idx]
            break

    X_train_mean = X_train.mean(axis=1)
    X_val_mean = X_val.mean(axis=1)

    # create test_set
    X_test = []
    for sample, center in df_test[["Sample ID", "Center ID"]].values:
        _features = np.load(test_features_dir / sample)
        coordinates, features = _features[:, :3], _features[:, 3:]
        if normalizer_type != "None":
            X_test.append(features / normalizers[center])
        else:
            X_test.append(features)

    X_test = np.array(X_test)
    X_test_mean = X_test.mean(axis=1)

    return (
        X_train,
        X_train_mean,
        y_train,
        X_val,
        X_val_mean,
        y_val,
        X_test,
        X_test_mean,
        df_test,
    )
