# OWKIN_challenge
Repository for the data challenge Détection de la mutation PIK3CA dans le cancer du sein par OWKIN.

## Notation

Given train set split in train and dev set
Given test set is kept like that, without any label

## Models

To simplify logs, the models should have a `name` and a `model_type` (usefull for `MaxModel` which copy them recursively). A `best_path` is also usefull.

## Some papers

https://paperswithcode.com/task/multiple-instance-learning => regroup a set of papers

https://paperswithcode.com/paper/using-neural-network-formalism-to-solve => idea of SmoothMax

https://paperswithcode.com/paper/real-world-anomaly-detection-in-surveillance => idea of better loss:
    - `l(neg_lime, pos_lime) = max(0, 1 - max{ model(pos_X_all) } + max{ model(neg_X_all) })`
