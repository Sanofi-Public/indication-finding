# Indication Finding with Matrix Factorization

This is an Indication Finding pipeline implemented using Non-Negative Matrix Factorization.

In fact this code provides dual MF implementation using Spark and Sklearn. Due to constraints on computing resources with spark clusters, Sklearn MF works on full IF cohort dataset and is selected as a primary MF implementation. Spark MF is inactive but could be activated if adequate resources are available.

Matrix Factorization objective is not the same as Indication Finding (IF) objective. MF learns patient and feature embeddings and attempts to reproduce input rating matrix. IF on the other hand utilizes the embeddings to rank features. The raking quality is validated externally.


## Configuration options

The MF options for training and inference are defined in project file `config/parameters/base/mf.yaml` The options are as follows:

```yaml
mf.embedding_size:
- 20
- 50
mf.regularization:
- 0.0
- 0.005
- 0.01
- 0.015
mf.max_iterations:
- 200
mf.train_split: 0.9
mf.split_seed: 42
mf.distance_measure: cosine
mf.finetune_measure: similarity_variance
mf.finetune_threads: 10
```

The parameters with multiple values such as `mf.embedding_size` are fine-tuned during training. The training configurations are created from all combination of the above parameters and multiple models are trained. Then each model is evaluated using criteria defined in `mf.finetune_measure` and model if the beset score is selected.


### Supported evaluation metrics

The fine-tuning evaluation metrics are defined in parameter `mf.finetune_measure` as one of the following values

- MSE - MSE(X,Y) where X is rating matrix and Y is predicted rating matrix
- similarity_variance - var(S) where S is feature similarity matrix computed as a product F * F, where F is a matrix of feature embeddings


### Supported distance measure

Feature embeddings are used for ranking. The ranking process compares features and referentials using distance measure defined in `mf.distance_measure`. The following values are supported:

- cosine - cosine distance
- euclidean - euclidean distance


## Running the pipeline

Due to Sklearn implementation the pipeline runs mostly local and requires adequate resources, such as RAM and number of CPU cores. Sklearn MF is single threaded, however when training with multiple fine-tuning parameters we can train multiple models in parallel. The number of fine-tuning threads is defined in `mf.finetune_threads`.

The training consists of two steps:

1. Model training - when we learn patient and feature embeddings
2. Model evaluation - when we evaluate the embeddings and pick the best model (fine-tuning)

Model training is CPU bound and requires less memory, therefore the training is done concurrently using multiple threads. Model evaluation is memory bound and requires more memory, therefore evaluation is done sequentially after training.


### Calculating resources needed for the pipeline

Example pipeline environment:

- cohort size: 12.2M patients
- train cohort size: 11M
- test cohort size: 1.2M
- feature count: 3000
- Suggested RAM: 256GB
- Suggested CPU: 8 cores

Example pipeline parameters:

- mf.finetune_threads: 10
- mf.embedding_size: 100


### RAM required for training (fine-tuning):

A model for the above cohort takes about 20-40 GB, depending on parameter configuration. Therefore with 256GB RAM we can train up to 10 models in parallel.

- R - RAM in workbench
- P - RAM per model
- C - number of threads/cores

Number of training cores:

    `C = R / P`


### RAM required for evaluation (fine-tuning):

RAM limitations are critical for meta-param fine tuning because in the evaluation we predict the rating matrix from the test cohort. The predicted rating matrix is dense and it requires memory size proportional to the product of its dimensions. Given T is a test rating matrix of dimensions NxM, the size of required RAM is proportional to NxM.

- T - test rating matrix
- N - number of test patients
- M - number of features
- R - RAM size in GB required for T

RAM in GB for evaluation:

    `R = N * M * 8 / 1e9`

Example:

    `R = 1,200,000 * 3,000 * 8 / 1e9 = 28 GB`


## Training evaluation metrics

Model training outputs evaluation metrics with the following values:

- MAE - Mean Absolute Error on rating predictions
- MAE_active - MAE on active (positive) features
- MAE_inactive - MAE on inactive (negative) features
- MSE - Mean Squared Error on rating predictions
- MSE_active - MSE on active (positive) features
- MSE_inactive - MSE on inactive (negative) features
- active_features - number of active (positive) feature in cohort
- inactive_features - number of inactive (negative) feature in cohort
- similarity_variance - variance of feature similarity

## Indication ranking

The pipeline output is a ranked list of features. The ranking is based on feature embedding distance. The distance measure for feature ranking is defined in `mf.distance_measure`.

## Code references

Two files for Spark MF implementation were copied from Python `recommenders` package. The copies were made because the package could not be installed side by side with other project packages due to conflicting dependencies.

- mf_constants.py - copied from Microsoft `recommenders` package
- mf_evaluation.py - copied from Microsoft `recommenders` package
