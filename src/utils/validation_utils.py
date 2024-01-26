def calc_recall_vs_fraction_recommended(
    validation_dataset,
    prediction_key="preds",
    ranked_list_column_name="feature_name",
):
    """
    Given a validation dataset containing query indications,
    target indications and predictions, calculate the fraction of target indications recalled
    as a function of the number of indications recommended.


    args:
        validation dataset: A dataset following the same format as the created by generate_validation_dataset in src/utils/indication_finding/core/validation/generate_validation_referential_list.py
        prediction_key: a key for retrieving the list of predicted indications from a validation dataset point. The validation dataset is a list of dictionaries, each representing one query, target and prediction.
        ranked_list_column_name: the name of the column of ranked indications in the prediction
    """
    max_len = -100
    rank_of_target = []
    for validation_point in validation_dataset:
        ranked_list = validation_point[prediction_key]
        ranked_list = list(ranked_list[ranked_list_column_name].values)
        max_len = max(max_len, len(ranked_list))
        query = validation_point["query"]
        for query_indication in query:
            if query_indication in ranked_list:
                ranked_list.remove(query_indication)
        target = validation_point["target"][0]
        if target in ranked_list:
            rank_of_target.append(ranked_list.index(target))
        else:
            print("Target", target, "not in ranked list")
            rank_of_target.append(100000000000000)

    recall = []  # how many indications recalled
    x = []  # how many indications recommended
    for k in range(0, max_len + 1):
        present = sum([1 * (rank < k) for rank in rank_of_target])
        x.append(k)
        total = len(rank_of_target)
        recall.append(present / total)

    return x, recall
