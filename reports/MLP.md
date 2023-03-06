## Best submission

`num_layer=2, inside_dim=512, val_score = 0.808, wd = 0 => 0.6642`


## Big grid search

### Tested paramters

```
num_layers = 5
inside_dim = 512
criterion = nn.BCELoss()

mono_batch_size = 64
mono_nb_epochs = 5000
batch_size = 16
nb_epochs = 1000

list_wd = [0, 1e-3, 3e-3, 1e-2]
list_mono_lr = [1e-6, 3e-6, 1e-5]
list_lr = [1e-6, 3e-6]
```

### Results

- See W&B or `images\comparison_Smooth_MaxMLP_wd_lr_mono_lr`

- `lr = 1e-6` same as `lr = 3e-6` but slower and steadier => keep `lr=1e-6`
- `mono_lr = 3e-6` looks the more robust => keep `mono_lr=3e-6`
- test results:
    - `wd = 1e-2: 0.6178`
    - `wd = 1e-3: 0.6201`





## MLP (mono_model)

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

MaxModel takes a model which maps features of shape 2048 to a probability (for example a model trained on the mean of the 1000 features of a slide). The max model compute this model on each of the 1000 squares of the slide and output the max probability. The training is quite longer, but no so much (we have to reduce batch_size).

    - If put on an untrained MLP, it doesn't work at all
    - If put on a trained MLP, it goes quite well to 0.68 of val score, but go down after
    - generalization on test set: bad (0.54...)


## SmoothMaxModel

Instead of taking the max, we take a soft max. Work a lot better than just max.

(torch.log(torch.exp(model_predictions).sum(axis=-2)) / model_predictions.shape[-2])

super val_score of 0.837 but 0.657 on test_set (best submission)


## Other loss
roc_loss: max(0, 1 - y_predict_pos + y_predict_neg) for every ps/neg couple

For now quite bad