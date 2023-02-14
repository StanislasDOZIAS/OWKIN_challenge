from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def build_dataset(data_path: str = "data/"):
    """
    Build `X_train_mean, X_val_mean, X_train, X_val, y_train, y_val`
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

    X_train_val_mean = []
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

        # slide-level averaging
        X_train_val_mean.append(np.mean(features, axis=0))
        X_train_val.append(features)
        y_train_val.append(label)

        centers_train_val.append(center)
        patients_train_val.append(patient)

    # convert to numpy arrays
    X_train_val_mean = np.array(X_train_val_mean)
    X_train_val = np.array(X_train_val)
    y_train_val = np.array(y_train_val)

    centers_train_val = np.array(centers_train_val)
    patients_train_val = np.array(patients_train_val)

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

        X_train_mean = X_train_val_mean[train_idx]
        X_val_mean = X_train_val_mean[val_idx]

        y_train = y_train_val[train_idx]
        y_val = y_train_val[val_idx]
        break

    X_test = []
    X_test_mean = []

    # load the data from `df_test` (~ 1 minute)
    for sample in df_test["Sample ID"].values:
        _features = np.load(test_features_dir / sample)
        coordinates, features = _features[:, :3], _features[:, 3:]
        X_test.append(features)
        X_test_mean.append(np.mean(features, axis=0))

    X_test = np.array(X_test)
    X_test_mean = np.array(X_test_mean)

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
