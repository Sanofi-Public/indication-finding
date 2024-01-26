# (c) Sanofi 2023 – Present
#
# Matrix factorization ratings

import logging
import sys
from typing import List, Set

import numpy as np
import pandas
import pyarrow
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window

COL_USER = "user"
COL_ITEM = "item"
COL_RATING = "rating"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_full_narrow_master(
    narrow_master: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    Create full narrow master table. A full master table is created by addinng
    records with inactive features. To distingush actative and inactive features
    a new column 'active' is addded with value 1 set for active features,
    and value 0 otherwise.

    Arguments:

    narrow_master - narrow master table with patient ids and feature names

    Returns:

    pyspark.sql.DataFrame - full narrow master table with inactive features
    """

    # Pivot narrow master to wide master
    pivot_master = narrow_master.groupBy("person_id").pivot("feature_name").count()

    # Define columns to unpivot
    feature_ids = (
        narrow_master.select("feature_name").distinct().orderBy("feature_name")
    )
    to_melt = set(feature_ids.rdd.map(lambda x: x.feature_name).collect())
    new_names = ["feature_name", "active"]

    # Unpivot wide master with null values
    melt_str = ",".join([f"'{c}', `{c}`" for c in to_melt])
    unpivot_master = pivot_master.select(
        *(set(pivot_master.columns) - to_melt),
        F.expr(f"stack({len(to_melt)}, {melt_str}) ({','.join(new_names)})"),
    )

    return unpivot_master.fillna(0)


def create_dense_rating_matrix(
    narrow_master: pyspark.sql.DataFrame,
) -> (pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame):
    """
    Create dense rating matrix for matrix factorization.

    Arguments:

    narrow_master - narrow master table with patient ids and feature names

    Returns:

    pyspark.sql.DataFrame - rating matrix for Spark ALS matrix factorization
    pyspark.sql.DataFrame - mapping of patient ids to integer ids in ranking matrix
    pyspark.sql.DataFrame - mapping of feature ids to integer ids in ranking matrix
    """

    # Get patient and features ids
    patient_ids = narrow_master.select("person_id").distinct().orderBy("person_id")
    feature_ids = (
        narrow_master.select("feature_name").distinct().orderBy("feature_name")
    )

    # Index patients and features
    patient_ids = patient_ids.withColumn(
        COL_USER,
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )
    feature_ids = feature_ids.withColumn(
        COL_ITEM,
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )

    # Get get full narrow master
    full_master = create_full_narrow_master(narrow_master)
    full_master = full_master.orderBy(full_master.columns)

    # Add integer indexing to full master table
    full_master_ids = full_master.join(patient_ids, on="person_id")
    full_master_ids = full_master_ids.join(feature_ids, on="feature_name")

    # create table with sparse rating matrix for factorization
    full_master_ids = full_master_ids.withColumn(COL_RATING, F.col("active"))
    ratings = full_master_ids.select([COL_USER, COL_ITEM, COL_RATING])

    return (ratings, patient_ids, feature_ids)


def create_sparse_rating_matrix(
    narrow_master: pyspark.sql.DataFrame,
) -> (pyspark.sql.DataFrame, pyspark.sql.DataFrame, pyspark.sql.DataFrame):
    """
    Create sparse rating matrix for matrix factorization.

    Arguments:

    narrow_master - narrow master table with patient ids and feature names

    Returns:

    pyspark.sql.DataFrame - sparse rating matrix for matrix factorization
    pyspark.sql.DataFrame - mapping of patient ids to integer ids in ranking matrix
    pyspark.sql.DataFrame - mapping of feature ids to integer ids in ranking matrix
    """

    # Get patient and features ids
    patient_ids = narrow_master.select("person_id").distinct().orderBy("person_id")
    feature_ids = (
        narrow_master.select("feature_name").distinct().orderBy("feature_name")
    )

    # Index patients and features
    patient_ids = patient_ids.withColumn(
        COL_USER,
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )
    feature_ids = feature_ids.withColumn(
        COL_ITEM,
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )

    # Get sparse narrow master
    sparse_master = narrow_master.withColumn(COL_RATING, F.lit(1))
    sparse_master = sparse_master.orderBy(sparse_master.columns)

    # Add integer indexing to full master table
    sparse_master = sparse_master.join(patient_ids, on="person_id")
    sparse_master = sparse_master.join(feature_ids, on="feature_name")
    ratings = sparse_master.select([COL_USER, COL_ITEM, COL_RATING])

    return (ratings, patient_ids, feature_ids)


def split_rating_matrix(
    rating_matrix: pyspark.sql.DataFrame,
    patient_ids: pyspark.sql.DataFrame,
    split_ratio: float,
    split_seed: int = 42,
) -> (pyspark.sql.DataFrame, pyspark.sql.DataFrame):
    """
    Split rating mtrix into train and test datasets by ratio.

    Arguments:

    rating_matrix - rating matrix to split
    patient_ids - patients ids to sample from
    split_ratio - train / test split ratio

    Returns:

    pyspark.sql.DataFrame - train samples
    pyspark.sql.DataFrame - test samples
    """
    # Cache for consistency
    patient_ids = patient_ids.persist(pyspark.StorageLevel.DISK_ONLY)
    patient_ids = patient_ids.repartition(100, COL_USER)

    # Split patients
    train_ids, test_ids = patient_ids.randomSplit(
        weights=[split_ratio, 1.0 - split_ratio],
        seed=split_seed,
    )

    col_index = f"{COL_USER}_index"

    # Re-index train and test patients
    train_ids = train_ids.withColumn(
        col_index,
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )
    test_ids = test_ids.withColumn(
        col_index,
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )

    train_ids = train_ids.orderBy(train_ids.columns)
    test_ids = test_ids.orderBy(test_ids.columns)

    # Create train/test samples
    train = (
        rating_matrix.join(train_ids, COL_USER)  # join with train ids
        .withColumn(COL_USER, F.col(col_index))  # set the new index
        .select([COL_USER, COL_ITEM, COL_RATING])  # select cols for MF
    )
    test = (
        rating_matrix.join(test_ids, COL_USER)  # join with test ids
        .withColumn(COL_USER, F.col(col_index))  # set the new index
        .select([COL_USER, COL_ITEM, COL_RATING])  # select cols for MF
    )

    train = train.orderBy(train.columns)
    test = test.orderBy(test.columns)

    return (train, test)


def create_feature_cooccurrence_table(
    narrow_master, patient_col="person_id", feature_col="feature_name"
) -> pyspark.sql.DataFrame:
    """
    Create feature-feature cooccurrence table
    """

    cooc = narrow_master.join(
        narrow_master.select(
            patient_col,
            F.col(feature_col).alias(f"{feature_col}_other"),
        ),
        on=patient_col,
    )

    cooc = cooc.groupBy(
        feature_col,
        f"{feature_col}_other",
    ).agg(F.count(F.lit(1)).alias("combination_count"))

    cooc = cooc.orderBy(feature_col, f"{feature_col}_other")

    cooc = (
        cooc.groupBy(feature_col)
        .pivot(f"{feature_col}_other")
        .sum("combination_count")
        .fillna(0)
        .orderBy(feature_col)
    )
    cooc = cooc.select(cooc.columns[0:1] + sorted(cooc.columns[1:]))

    return cooc


def create_cooccurrence_rating_matrix(cooc_table):
    """
    Create feature-feature cooccurrence sparse rating table
    """

    # get item ids index
    feature_ids = cooc_table.select("feature_name").orderBy("feature_name")
    feature_ids = feature_ids.withColumn(
        COL_ITEM,
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )

    # get artificial user ids for train/test split to work
    patient_ids = cooc_table.select("feature_name").orderBy("feature_name")
    patient_ids = patient_ids.withColumn(
        COL_USER,
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1,
    )

    feature_list = cooc_table.columns[1:]

    # get cooccurrence matrix
    cooc_matrix = np.array(cooc_table.select(feature_list).collect())

    # convert cooccurrence matrix to sparse rating matrix
    nonzero = np.nonzero(cooc_matrix)

    user = nonzero[0]
    item = nonzero[1]
    rating = cooc_matrix[nonzero[0], nonzero[1]]

    rating_matrix = pandas.DataFrame(
        data=np.stack([user, item, rating]).transpose(),
        columns=[COL_USER, COL_ITEM, COL_RATING],
    )

    spark = SparkSession.builder.getOrCreate()
    rating_matrix = spark.createDataFrame(rating_matrix)

    return (rating_matrix, patient_ids, feature_ids)
