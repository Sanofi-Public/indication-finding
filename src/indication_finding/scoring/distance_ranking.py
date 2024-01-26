import logging
import re
import sys

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import functions as F
from pyspark.sql.window import Window as W
from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(
    logging.StreamHandler(sys.stdout)
)  # guarantees printing to terminal for debugging


def features_to_ids(feature_ids):
    """
    Convert feature table to distionary of features and corresponding ids

    Args:
        feature_ids: Table with featgure names and ids

    Returns:
        Dictionary of features and ids

    Example of feature-id dictionary output:

    {"feature_A": 0, "feature_B": 1, ...}
    """
    if not isinstance(feature_ids, pd.DataFrame):
        feature_ids = feature_ids.toPandas()
    feature_to_ids = {}
    for index, row in feature_ids.iterrows():
        feature_to_ids[row["feature_name"]] = row["feature_id"]
    return feature_to_ids


def referentials_to_features(
    feature_id_map,
    composite_referentials,
    referential_filter,
):
    """
    Convert referentails table to distionary of features and weights

    Args:
        feature_id_map: Dictionary of features and feature ids
        composite_referentials: Table with composite referentials
        referential_inclusion_list: Referential inclusion list

    Returns:
        Dictionary of features and weights

    Example of feature-id dictionary output:

    {
        'ref_composite_A':
        {
            'feature_names': ['feature_A'],
            'feature_weights': [1.0]
        },
        'ref_composite_B': {
            'feature_names': ['feature_B', 'feature_C'],
            'feature_weights': [1.0, 0.5]}
        }
    }
    """
    referentials = {}

    # add single referentials
    for feature in sorted(list(feature_id_map.keys())):
        if not matches(feature, referential_filter):
            continue

        referentials[feature] = {
            "feature_names": [feature],
            "feature_weights": [1.0],
        }

    # add composite referentials
    for index, row in composite_referentials.iterrows():
        ref = row["referential_name"]
        feature = row["feature_name"]
        weight = float(row.get("weight", 1.0))

        if not matches(ref, referential_filter):
            continue

        if feature not in feature_id_map:
            Logger().warning(
                f"Composite referential '{feature}' not defined in features."
            )
            continue

        features = referentials.get(ref, {})
        feature_names = features.get("feature_names", [])
        feature_weights = features.get("feature_weights", [])

        feature_names.append(feature)
        feature_weights.append(weight)

        features["feature_names"] = feature_names
        features["feature_weights"] = feature_weights
        referentials[ref] = features

    return referentials


def similarity(a, b, name: str = "cosine"):
    """
    Calculate cosine similarity

    Args:
        a: Ndarray 1D vector
        b: Ndarray 1D vector
        name: Distance measure

    Returns:
        Similarity measure
    """
    if name == "cosine":
        return cosine_similarity([a], [b]).item()

    if name == "euclidean":
        return 1.0 / (1.0 + euclidean(a, b))

    raise ValueError(f"Invalid distance measure {name}")


def matches(text, filters):
    """
    Checks is text matches a given set of filters.

    Args:
        a: text to match
        b: filters to check

    Returns:
        True if matches, False otherwise
    """
    if filters is None:
        return True

    for f in filters:
        if re.match(f, text):
            return True

    return False


def score_features(
    feature_vectors,
    feature_ids,
    composite_referentials,
    referential_filter,
    ranking_filter,
    distance_measure,
):
    """
    Compute feature scores relative to referentials.

    The features are compared to referentials based on vector similarity
    between features and referentials

    Args:
        feature_vectors: Feature vectors to compute scores for
        feature_ids: Table with feature names and ids
        composite_referentials: Table with composite referentials
        referential_filter: Referential inclusion lis,
        ranking_filter: Feature ranking includion list
        distance_measure: Distance measure

    Returns:
        List of features in alphabetical order
        Dictionary of features and scores
    """
    feature_id_map = features_to_ids(feature_ids)
    feature_ranking = referentials_to_features(
        feature_id_map, composite_referentials, referential_filter
    )
    if type(feature_vectors) is not np.ndarray:
        feature_vectors_np = feature_vectors.toarray()
    else:
        feature_vectors_np = feature_vectors
    ref_names = list(feature_ranking.keys())

    # filter features to score
    feature_names = []
    for feature in sorted(feature_id_map.keys()):
        if matches(feature, ranking_filter):
            feature_names.append(feature)

    # calculate referential scores
    for ref_name in ref_names:
        ref_f_names = feature_ranking[ref_name]["feature_names"]
        ref_f_weights = feature_ranking[ref_name]["feature_weights"]
        ref_f_score = []

        for f_name in feature_names:
            similarity_score = 0
            f_id = feature_id_map[f_name]
            f_vector = feature_vectors_np[f_id, :]

            for ref_f_name, ref_f_weight in zip(ref_f_names, ref_f_weights):
                ref_f_id = feature_id_map[ref_f_name]
                ref_f_vector = feature_vectors_np[ref_f_id, :]

                similarity_score += ref_f_weight * similarity(
                    f_vector, ref_f_vector, distance_measure
                )

            ref_f_score.append(similarity_score / sum(ref_f_weights))

        feature_ranking[ref_name]["feature_score"] = ref_f_score

    return feature_names, feature_ranking


def rank_features(
    feature_vectors,
    feature_ids,
    composite_referentials,
    referential_filter,
    ranking_filter,
    distance_measure="cosine",
    top_k=30,
):
    """
    Rank features relative to referentials.

    The features are compared to referentials based on vector similarity
    between features and referentials

    Args:
        feature_vectors: Feature vectors to compute scores for
        feature_ids: Table with feature names and ids
        composite_referentials: Table with composite referentials
        referential_filter: Referential inclusion list,
        ranking_filter: Feature ranking includion list,
        distance_measure: Distance measure
        top_k: Number of rankings in top K results

    Returns:
        List of features in alphabetical order
        Table with ranked features agains referentials
    """
    feature_names, feature_scores = score_features(
        feature_vectors,
        feature_ids,
        composite_referentials,
        referential_filter,
        ranking_filter,
        distance_measure,
    )
    ref_names = list(feature_scores.keys())

    # convert results to Pandas table
    rank_df = pd.DataFrame(data={"feature_name": feature_names})
    for ref_name in ref_names:
        rank_df[f"{ref_name}_score"] = feature_scores[ref_name]["feature_score"]

    # calculate rank
    for ref_name in ref_names:
        rank_df[f"{ref_name}_rank"] = rank_df[f"{ref_name}_score"].rank(
            method="min", ascending=False
        )

    # calculate mean and median rank
    ref_rank_names = [f"{r_n}_rank" for r_n in ref_names]
    ref_score_names = [f"{r_n}_score" for r_n in ref_names]
    rank_df["mean_score"] = rank_df[ref_score_names].mean(axis=1)
    rank_df["median_score"] = rank_df[ref_score_names].median(axis=1)
    rank_df["mean_rank"] = rank_df[ref_rank_names].mean(axis=1)
    rank_df["median_rank"] = rank_df[ref_rank_names].median(axis=1)

    # calculate number of features in top K
    rank_df[f"number_in_top_{top_k}"] = (rank_df[ref_rank_names] <= top_k).sum(axis=1)

    # sort results by median rank
    return rank_df.sort_values(by="mean_score", ascending=False)
