## Clustering approach

Do clustering with all squares (see `Kmeans.ipynb`), doesn't work for now ==> maybe later ?

## MLP approach

MLP on the mean of all the 1000 square's features (see `MLP.ipynb`)

Globally, all the models behave the same whatever num_layers >= 2 and inside_dim. num_layers=1 is a special case (we retrieve a logistic regression)

Just be carefull with num_layer = 3 and inside_dim = 2048 : little overfitting

They all perform quite good on the val set (val_auc_roc_auc_score = 0.72). 

CCL: prefer little parameters

BUT:
    - 1 layer (i.e logistic regression): 0.58 on test set
    - 2 layer: 0.6434 on test set

=> two different behaviors (repartition not the same between test and train)

