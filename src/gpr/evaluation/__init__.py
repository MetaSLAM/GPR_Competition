'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/evaluation/__init__.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/evaluation
Created Date: Friday, March 4th 2022, 4:25:15 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''

import numpy as np
from sklearn.neighbors import KDTree


def get_recall(reference_feature, queries_feature, true_threshold=2, num_neighbors=50):
    '''reference_feature [N1, M]: N1 frames reference feature
    queries_feature [N2, M]: N2 frames query features
    true_threshold [int]: threshold for true place recognition
    num_neighbors [int]: Knn search
    return: topN_recall, top1_similarity_score, one_percent_recall
    '''
    reference_output = reference_feature
    queries_output = queries_feature

    database_nbrs = KDTree(reference_output)
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(reference_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(true_threshold, len(queries_output) - true_threshold):

        true_neighbors = i + np.arange(-true_threshold, true_threshold + 1)
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors
        )
        indices = indices[0]

        for j in range(0, len(indices)):
            if indices[j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], reference_output[indices[j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    topN_recalls = (np.cumsum(recall) / float(num_evaluated)) * 100

    return topN_recalls, one_percent_recall
