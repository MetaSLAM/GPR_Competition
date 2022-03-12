"""This script provides functions for grading. 
It is aimed for the organizers.

Author: Haowen Lai
"""

import numpy as np
from sklearn.neighbors import KDTree


def get_mixed_up_recall(
    ref_query_feature: np.ndarray, true_threshold: int, num_neighbors: int, seed: int
):
    """Analyze the recall of references and queries.
    This function is used for grading from mixed up references and queries.
    Args:
        ref_query_feature [N_ref+N_query, M]: N_ref frames of reference features
            and N_query frames of query features. You need the correct seed to
            find out the true order.
        true_threshold [int]: threshold for true place recognition
        num_neighbors [int]: Knn search, determine the N for top-N recall
    Return:
        topN_recall, one_percent_recall
    """
    # Find the true correct order
    np.random.seed(seed=seed)
    N_ref_query = ref_query_feature.shape[0]
    random_ids = np.random.permutation(N_ref_query)  # id:val {after_id:before_id}
    correct_ids = [0] * N_ref_query  # id:val {before_id:after_id}

    for after_id, before_id in enumerate(random_ids):
        correct_ids[before_id] = after_id
    correct_ids = np.array(correct_ids)

    # extract references and queries
    reference_feature = ref_query_feature[correct_ids[: N_ref_query // 2]]
    queries_feature = ref_query_feature[correct_ids[N_ref_query // 2 :]]

    # The following is the same as get_recall()
    database_nbrs = KDTree(reference_feature)
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    one_percent_threshold = max(int(round(len(reference_feature) / 100.0)), 1)

    num_evaluated = 0
    dists, indices = database_nbrs.query(queries_feature, k=num_neighbors)

    for i in range(true_threshold, len(queries_feature) - true_threshold):
        true_neighbors = i + np.arange(-true_threshold, true_threshold + 1)
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        retrieved_indices = indices[i]
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


def main(submission_file: str, seed: int) -> float:
    """Entry point for grading.
    Args:
        submission_file: the file name of the submitted *.npy file. Each row is
            the feature for a submap/frame. So the size is N_submap * feature_dim.
        seed: the seed to initialize random operation
    Return:
        top_1_recall: the top one recall
    """
    ref_query_feature = np.load(submission_file)

    topN_recalls, one_percent_recall = get_mixed_up_recall(
        ref_query_feature, true_threshold=1, num_neighbors=18, seed=seed
    )

    return topN_recalls[0]


if __name__ == '__main__':
    submission_file = 'datasets/pitts_query_sample_fea.npy'
    seed = 9216034

    top_1_recall = main(submission_file, seed)
    print(f'top 1 recall is {top_1_recall:.2%}')
