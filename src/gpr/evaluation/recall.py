'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/evaluation/__init__.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/evaluation
Created Date: Friday, March 4th 2022, 4:25:15 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''

import numpy as np
from sklearn.neighbors import KDTree


def get_recall(
    reference_feature: np.ndarray,
    queries_feature: np.ndarray,
    true_threshold: int = 2,
    num_neighbors: int = 50,
):
    """Analyze the recall of references and queries.
    Args:
        reference_feature [N1, M]: N1 frames reference feature
        queries_feature [N2, M]: N2 frames query features
        true_threshold [int]: threshold for true place recognition
        num_neighbors [int]: Knn search
    Return:
        topN_recall, one_percent_recall
    """
    database_nbrs = KDTree(reference_feature)
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    one_percent_threshold = max(int(round(len(reference_feature) / 100.0)), 1)

    num_evaluated = 0
    dists, indices = database_nbrs.query(queries_feature, k=num_neighbors + 1)

    for i in range(true_threshold, len(queries_feature) - true_threshold):
        true_neighbors = i + np.arange(-true_threshold, true_threshold + 1)
        true_neighbors = np.setdiff1d(true_neighbors, [i])  # exclude itself
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        retrieved_indices = indices[i, 1:]  # exclude itself
        for j in range(0, len(retrieved_indices)):
            if retrieved_indices[j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_feature[i], reference_feature[retrieved_indices[j]]
                    )
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        # judge one percent recall
        one_per_set = set(retrieved_indices[0:one_percent_threshold]).intersection(
            set(true_neighbors)
        )
        if len(one_per_set) > 0:
            one_percent_retrieved += 1

    one_percent_recall = one_percent_retrieved / float(num_evaluated)
    topN_recalls = np.cumsum(recall) / float(num_evaluated)

    return topN_recalls, one_percent_recall
