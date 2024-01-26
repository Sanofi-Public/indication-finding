# (c) Sanofi 2023 – Present
#
# Matrix factorization ranked list

import logging
import sys

import pandas
import pyspark
import pyspark.sql.functions as F
from scipy import sparse
from src.indication_finding.scoring.distance_ranking import rank_features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_mf_ranked_list(
    feature_vectors: pandas.DataFrame,
    feature_ids: pyspark.sql.DataFrame,
    composite_referentials: pyspark.sql.DataFrame,
    query_filter: pandas.DataFrame,
    ranking_filter: pandas.DataFrame,
    distance_measure: str = "cosine",
    top_k: int = 30,
) -> pandas.DataFrame:
    """
    Create ranked list with features from matrix factorization.

    Arguments:

    feature_vectors: - Matrix factorization features embeddings
    feature_ids: - Mapping between feature names and ids
    query_filter: The query to use to rank other indications
    ranking_filter: Feature filter
    distance_measure: Distance measure
    top_k: Number of rankings in top K results

    Returns:

    Ranked list of features
    """
    sparse_features = sparse.csr_matrix(feature_vectors.to_numpy())
    feature_ids = feature_ids.withColumn("feature_id", F.col(COL_ITEM))

    ranked_list = rank_features(
        sparse_features,
        feature_ids,
        composite_referentials,
        query_filter,
        ranking_filter,
        distance_measure,
        top_k,
    )

    return ranked_list
