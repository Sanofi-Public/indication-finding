# (c) Sanofi 2023 – Present
#
# Matrix factorization ranked list

import pandas
import pyspark
import pyspark.sql.functions as F
from scipy import sparse

from src.utils.indication_finding.core.mf.rating_matrix import COL_ITEM
from src.utils.indication_finding.core.scoring.distance_ranking import rank_features


def get_mf_ranked_list(
    feature_vectors: pandas.DataFrame,
    feature_ids: pyspark.sql.DataFrame,
    composite_referentials: pandas.DataFrame,
    referential_filter: pandas.DataFrame,
    ranking_filter: pandas.DataFrame,
    distance_measure: str = "cosine",
    top_k: int = 30,
) -> pandas.DataFrame:
    """
    Create ranked list with features from matrix factorization.

    Arguments:

    feature_vectors: - Matrix factorization features embeddings
    feature_ids: - Mapping between feature names and ids
    composite_referentials: Referentials
    referential_filter: Referential filter
    ranking_filter: Feature filter
    distance_measure: Distance measure
    top_k: Number of rankings in top K results

    Returns:

    Ranked list of features
    """
    sparse_features = sparse.csr_matrix(feature_vectors.to_numpy())
    feature_ids = feature_ids.withColumn("feature_id", F.col(COL_ITEM))

    ranked_list = rank_features(
        feature_vectors=sparse_features,
        feature_ids=feature_ids,
        composite_referentials=composite_referentials,
        referential_filter=referential_filter,
        ranking_filter=ranking_filter,
        distance_measure=distance_measure,
        top_k=top_k,
    )

    return ranked_list