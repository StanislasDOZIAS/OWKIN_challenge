## 

## Clustering [BIG HOPE ON THIS TO AVOID OVERFITTING]

Do clustering with all squares (see `Kmeans.ipynb`), doesn't work for now ==> maybe later ?

NO UPDATED AT ALL, JUST SOME IDEAS

## MLP

MLP on the mean of all the 1000 square's features (see `MLP.ipynb`)

Globally, all the models behave the same whatever num_layers >= 2 and inside_dim. num_layers=1 is a special case (we retrieve a logistic regression)

Just be carefull with num_layer = 3 and inside_dim = 2048 : little overfitting

They all perform quite good on the val set (val_auc_roc_auc_score = 0.72). 

CCL: prefer little parameters

BUT:
    - 1 layer (i.e logistic regression): 0.58 on test set
    - 2 layer: 0.6434 on test set

=> two different behaviors (repartition not the same between test and train)

Hyperparameter kept for BCELoss:
```
num_layers = 2
inside_dim = 512

lr=1e-6
batch_size = 64
nb_epochs = 2000
```

Hyperparameter kept for RocLoss:
```
num_layers = 2
inside_dim = 512

lr=1e-5
batch_size = 64
nb_epochs = 2000
```

## MaxModel

MaxModel takes a model which maps features of shape 2048 to a probability (for example a model trained on the mean of the 1000 features of a lime). The max model compute this model on each of the 1000 squares of the lime and output the max probability. The training is quite longer, but no so much (we have to reduce batch_size).

    - If put on an untrained MLP, it doesn't work at all
    - If put on a trained MLP, it goes quite well to 0.68 of val score, but go down after
    - generalization on test set: bad (0.54...)


## SmoothMaxModel

Instead of taking the max, we take a soft max.

(torch.log(torch.exp(model_predictions).sum(axis=-2)) / model_predictions.shape[-2])


## Other loss
roc_loss: max(0, 1 - y_predict_pos + y_predict_neg) for every ps/neg couple

For now no big difference (even lower ?)