# OWKIN_challenge
Repository for the data challenge "Détection de la mutation PIK3CA dans le cancer du sein" by OWKIN.

## Notation

If not val set per center, given train set split in train and dev is set (see `build_dataset` in `owkin\dataset.py`). Important line:
```
kfold = StratifiedKFold(5, shuffle=True, random_state=42)
```

Given test set is kept like that, without any label

## Models

To simplify logs, see `owkin\models\base_models.py` to see what a model shoud have to make easier logging.
