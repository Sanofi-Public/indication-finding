# (c) Sanofi 2023 - Present
#
# Matrix factorization features

import logging
import sys
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import pandas
import pyarrow
import pyspark
import pyspark.sql.functions as F
import sklearn
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from src.indication_finding.mf.rating_matrix import COL_ITEM, COL_RATING, COL_USER

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calc_mf_features(
    rating_matrix_train: pyarrow.Table,
    rating_matrix_test: pyarrow.Table,
    embedding_size: List[int],
    regularization: List[float],
    max_iterations: List[int],
    finetune_measure: str = "MSE",
    finetune_workers: int = None,
) -> (pandas.DataFrame, pandas.DataFrame):
    """
    Calculate feature embeddings using matrix factorization.

    Arguments:

    rating_matrix_train - train rating matrix in narrow format
    rating_matrix_test - test rating matrix in narrow format
    embedding_size - list of embedding dimentions
    regularization - list of regularization parameters
    max_iterations - list of maximum number of iterations
    finetune_measure - fine-tune measure to select optimal parameters
    finetune_workers - number of fine-tunning workers

    Returns:

    pandas.DataFrame - feature vectors (embeddings)
    pandas.DataFrame - validation metrics on test split
    """

    train_matrix = np.array(rating_matrix_train)
    test_matrix = np.array(rating_matrix_test)

    rows_train = int(np.max(train_matrix[:, 0]) + 1)
    cols_train = int(np.max(train_matrix[:, 1]) + 1)

    rows_test = int(np.max(test_matrix[:, 0]) + 1)
    cols_test = int(np.max(test_matrix[:, 1]) + 1)

    csr_shape = (max(rows_train, rows_test), max(cols_train, cols_test))

    x_train = _index_array_to_sparse_matrix(train_matrix, shape=csr_shape)
    x_test = _index_array_to_sparse_matrix(test_matrix, shape=csr_shape)

    best_F = None
    best_metrics = None
    best_config = None
    best_score = -np.inf

    # run fine tuning in parallel on multiple cores
    with Pool(processes=finetune_workers) as pool:
        configs = []

        # create list of configurations to run fine-tunning on
        for es in embedding_size:
            for re in regularization:
                for mi in max_iterations:
                    configs.append(
                        {
                            "dataset": x_train,
                            "embedding_size": es,
                            "regularization": re,
                            "max_iterations": mi,
                        }
                    )

        # run training tasks in parallel, get results as tasks complete
        results = [r for r in pool.imap(_mf_task, configs)]

        # evaluate the results when all tasks are complete and ram freed
        for result in results:
            config = result[0]
            model = result[1]
            metrics = _evaluate("test", model, x_test)
            score = _get_perf_score(metrics, measure=finetune_measure)
            if score > best_score:
                best_F = model.components_
                best_metrics = metrics
                best_config = config
                best_score = score

    logger.info(f"MF fine-tune measure: {finetune_measure}")
    logger.info(f"MF best embedding_size: {best_config['embedding_size']}")
    logger.info(f"MF best regularization: {best_config['regularization']}")
    logger.info(f"MF best max_iterations: {best_config['max_iterations']}")
    logger.info(f"MF best score: {best_score}")

    logger.info(f"MF re-fitting on full dataset")
    x_train = vstack([x_train, x_test])
    best_config["dataset"] = x_train
    config, model = _mf_task(best_config)
    best_F = model.components_

    columns = [f"emb_vec_{col}" for col in range(best_F.shape[0])]
    F_df = pandas.DataFrame(best_F.transpose(), columns=columns)

    return (F_df, best_metrics)


def _get_perf_score(metrics: pandas.DataFrame, measure: str) -> float:
    """
    Calculate performance score from MF evaluation metrics.

    Arguments:

    metrics - MF evaluation metrics

    Returns:

    float - MF performance score
    """
    if len(metrics.index) == 0:
        return 0

    metrics = metrics.set_index("metric")

    if measure == "MSE":
        return 1 / (1 + metrics.loc["MSE", "value"])

    if measure == "similarity_variance":
        return metrics.loc["similarity_variance", "value"]

    raise Exception(f"Unsupported finetune metric: {measure}")


def _mf_task(config):
    """
    Run MF task with a config

    Arguments:

    config - taks config containng input, output and meta-params

    Returns:

    Input config
    Trained MF model
    """

    model = _run_mf(
        x_train=config["dataset"],
        embedding_size=config["embedding_size"],
        regularization=config["regularization"],
        max_iterations=config["max_iterations"],
    )

    return (config, model)


def _run_mf(
    x_train: csr_matrix,
    embedding_size: int,
    regularization: float,
    max_iterations: int,
) -> (np.ndarray, pandas.DataFrame):
    """
    Run MF with one set of meta parameters.

    Arguments:

    x_train - train rating matrix in narrow format
    embedding_size - embedding dimentions
    regularization - regularization parameters
    max_iterations - maximum number of iterations

    Returns:

    np.ndarray - feature vectors (embeddings)
    pandas.DataFrame - validation metrics on test dataset
    """

    logger.info(f"MF run embedding_size: {embedding_size}")
    logger.info(f"MF run regularization: {regularization}")
    logger.info(f"MF run max_iterations: {max_iterations}")

    model = NMF(
        n_components=embedding_size,
        init="random",
        random_state=42,
        alpha_W=regularization,
        max_iter=max_iterations,
    )
    logger.info(f"MF model: {type(model)}")
    model.fit(x_train)

    return model


def _index_array_to_sparse_matrix(
    data: np.ndarray, shape: Tuple[int, int]
) -> csr_matrix:
    """
    Create sparse matrix from index array

    Arguments:

    data - index array of format [[row, col, value]]
    shape - the shape of the matrix. This is imporant, because it is set at the time of training. So when creating the test matrix
            you need make sure it is the same shape as the training matrix.

    Returns:

    scipy.sparse.csr_matrix - Scipy sparse matrix
    """
    rows_index = data[:, 0]
    cols_index = data[:, 1]
    values = data[:, 2]

    return csr_matrix((values, (rows_index, cols_index)), shape=shape)


def _get_active_values(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_active = np.nonzero(X)
    Y_active = Y[X_active]


def _evaluate(
    name: str,
    model: NMF,
    X: csr_matrix,
) -> pandas.DataFrame:
    """
    Evaluate matrix factorization model ratings on test dataset

    Arguments:

    name - evaluation dataset name
    model - trained MF model to evaluate
    X - test dataset

    Returns:

    Evaluation metrics table
    """
    columns = [
        "metric",
        "value",
        "dataset",
    ]

    if X is None or np.prod(X.shape) == 0:
        logger.warning(f"Skipping validation on empty {name} dataset.")
        return pandas.DataFrame(columns=columns)

    F = model.components_
    P = model.transform(X)

    FF = cosine_similarity(F.transpose(), F.transpose())
    similarity_variance = np.var(FF)

    X = X.astype(dtype=np.float32).toarray()
    Y = P.dot(F).astype(dtype=np.float32)

    mae = sklearn.metrics.mean_absolute_error(X, Y)
    mse = sklearn.metrics.mean_squared_error(X, Y)

    active = np.nonzero(X)
    inactive = np.nonzero(1 - X)

    active_features = active[0].shape[0]
    inactive_features = inactive[0].shape[0]

    mae_active = sklearn.metrics.mean_absolute_error(X[active], Y[active])
    mse_active = sklearn.metrics.mean_squared_error(X[active], Y[active])

    mae_inactive = sklearn.metrics.mean_absolute_error(X[inactive], Y[inactive])
    mse_inactive = sklearn.metrics.mean_squared_error(X[inactive], Y[inactive])

    metrics = [
        ("MAE", mae, name),
        ("MAE_active", mae_active, name),
        ("MAE_inactive", mae_inactive, name),
        ("MSE", mse, name),
        ("MSE_active", mse_active, name),
        ("MSE_inactive", mse_inactive, name),
        ("active_features", active_features, name),
        ("inactive_features", inactive_features, name),
        ("similarity_variance", similarity_variance, name),
    ]

    return pandas.DataFrame(data=metrics, columns=columns)
