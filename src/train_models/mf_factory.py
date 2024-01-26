# Sanofi 2023 – Present
#
# Indication Finding based on Matrix Factorization
"""
This module defines the Matrix Factorization (MF) pipeline.
"""
import argparse
import logging
import os
import pickle as pkl
import sys
from datetime import datetime
from config.config import PROJECT_DIR

from pyspark.sql import SparkSession

from src.indication_finding.mf.mf_features_sklearn import calc_mf_features
from src.indication_finding.mf.rating_matrix import (
    create_cooccurrence_rating_matrix,
    create_feature_cooccurrence_table,
    create_sparse_rating_matrix,
    split_rating_matrix,
)
from src.indication_finding.scoring.mf_ranking import get_mf_ranked_list
from src.utils.utils import parse_yaml_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(narrow_master, args):
    """Runs matrix factorization and creates ranked list."""

    if args["test"]:
        unique_people = narrow_master.select("person_id").dropDuplicates(["person_id"])
        random_sample = unique_people.sample(0.01)  # select only 1 % of people
        narrow_master = narrow_master.join(random_sample, ["person_id"], "inner")

    mf_rating_matrix, mf_patient_ids, mf_feature_ids = create_sparse_rating_matrix(
        narrow_master
    )
    mf_rating_matrix_train, mf_rating_matrix_test = split_rating_matrix(
        mf_rating_matrix, mf_patient_ids, args["mf_train_split"], args["mf_split_seed"]
    )

    mf_rating_matrix_train.cache()
    mf_rating_matrix_test.cache()
    mf_rating_matrix_train.show()
    mf_rating_matrix_test.show()

    mf_rating_matrix_train = mf_rating_matrix_train.toPandas()
    mf_rating_matrix_test = mf_rating_matrix_test.toPandas()

    mf_feature_vectors, mf_validation_metrics = calc_mf_features(
        mf_rating_matrix_train,
        mf_rating_matrix_test,
        args["mf_embedding_size"],
        args["mf_regularization"],
        args["mf_max_iterations"],
        args["mf_finetune_measure"],
        args["mf_finetune_workers"],
    )

    return mf_feature_vectors, mf_feature_ids, mf_validation_metrics


def main_cooc(narrow_master, args):
    if args["test"]:
        unique_people = narrow_master.select("person_id").dropDuplicates(["person_id"])
        random_sample = unique_people.sample(0.01)  # select only 1 % of people
        narrow_master = narrow_master.join(random_sample, ["person_id"], "inner")

    cooccurrence_table = create_feature_cooccurrence_table(narrow_master)
    (
        cooccurrence_rating_matrix,
        cooccurrence_patient_ids,
        cooccurrence_feature_ids,
    ) = create_cooccurrence_rating_matrix(cooccurrence_table)
    (
        mf_cooccurrence_rating_matrix_train,
        mf_cooccurrence_rating_matrix_test,
    ) = split_rating_matrix(
        cooccurrence_rating_matrix,
        cooccurrence_patient_ids,
        args["mf_cooccurrence_train_split"],
        args["mf_cooccurrence_split_seed"],
    )

    mf_cooccurrence_rating_matrix_train.cache()
    mf_cooccurrence_rating_matrix_test.cache()

    mf_cooccurrence_rating_matrix_train = mf_cooccurrence_rating_matrix_train.toPandas()
    mf_cooccurrence_rating_matrix_test = mf_cooccurrence_rating_matrix_test.toPandas()

    (
        mf_cooccurrence_feature_vectors,
        mf_cooccurrence_validation_metrics,
    ) = calc_mf_features(
        mf_cooccurrence_rating_matrix_train,
        mf_cooccurrence_rating_matrix_test,
        args["mf_cooccurrence_embedding_size"],
        args["mf_cooccurrence_regularization"],
        args["mf_cooccurrence_max_iterations"],
        args["mf_cooccurrence_finetune_measure"],
    )

    return mf_coccurence_feature_vector, cooccurrence_feature_ids, mf_cooccurrence_validation_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser("An argument parser for matrix factorization.")
    parser.add_argument("--input_data", type=str, dest="input_data", required=True)
    args = parser.parse_args()

    now = datetime.now()
    output_location = f"mf_training_{now.strftime('%Y_%m_%d_%H_%M_%S')}"

    spark = SparkSession.builder.getOrCreate()
    # load some libraries to read from s3
    spark.conf.set("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
    spark.conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    narrow_master = spark.read.parquet(args.input_data)

    assert not os.path.exists(output_location)
    os.makedirs(output_location)
    training_config = parse_yaml_args(os.path.join(PROJECT_DIR, "config", "training_mf.yaml"))

    ## Train vanilla MF
    mf_feature_vectors, mf_feature_ids, mf_validation_metrics = main(narrow_master, training_config)

    # save the feature vectors and validation metrics to the output directory
    output_mf_features = os.path.join(output_location, "mf_features.csv")
    mf_feature_vectors.to_csv(output_mf_features)

    output_mf_feature_ids = os.path.join(output_location, "mf_feature_ids.csv")
    mf_feature_ids.to_csv(output_mf_feature_ids)

    output_validation_metrics = os.path.join(
        output_location, "validation_metrics.pkl"
    )
    with open(output_validation_metrics, "wb") as f:
        pkl.dump(mf_validation_metrics, f)

    ## Train cooc MF
    mf_cooc_feature_vectors, mf_cooc_feature_vectors, mf_cooc_validation_metrics = main_cooc(
        narrow_master, training_config
    )

    # save the feature vectors and validation metrics to the output directory
    output_mf_cooc_features = os.path.join(output_location, "mf_features_cooc.csv")
    mf_cooc_feature_vectors.to_csv(output_mf_cooc_features)

    output_mf_cooc_feature_ids = os.path.join(output_location, "mf_cooc_feature_ids.csv")
    mf_cooc_feature_ids.to_csv(output_mf_cooc_feature_ids)

    output_validation_metrics_cooc = os.path.join(
        output_location, "validation_metrics_cooc.pkl"
    )
    with open(output_validation_metrics_cooc, "wb") as f:
        pkl.dump(mf_cooc_validation_metrics, f)
