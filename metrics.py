from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import euclidean_distances

def calculate_mmrr(predictions, ground_truths):
    """
    Calculate the Mean Reciprocal Rank (MRR) for multiple queries.
    
    :param predictions: List of lists, where each inner list contains the document IDs returned for a query.
    :param ground_truths: List of lists, where each inner list contains the relevant document IDs for a query.
    :return: MRR score.
    """
    total_mrr = 0
    num_queries = len(predictions)
    # print(predictions, ground_truths)
    
    for preds, truths in zip(predictions, ground_truths):
        rank = 1
        # print("prediction", preds)
        # print("truths", truths)
        for doc_id in preds:
            if doc_id in truths:
                total_mrr += 1 / rank
                # break
            rank += 1
    
    mrr = total_mrr / num_queries if num_queries > 0 else 0
    # print("predictions", predictions)
    # print("ground truth",  ground_truths)
    # print("mrr", mrr)
    
    return mrr

def calculate_mrr(predictions, ground_truths):
    """
    Calculate the Mean Reciprocal Rank (MRR) for multiple queries.
    
    :param predictions: List of lists, where each inner list contains the document IDs returned for a query.
    :param ground_truths: List of lists, where each inner list contains the relevant document IDs for a query.
    :return: MRR score.
    """
    total_mrr = 0
    num_queries = len(predictions)
    # print(predictions, ground_truths)
    
    for preds, truths in zip(predictions, ground_truths):
        rank = 1
        # print("prediction", preds)
        # print("truths", truths)
        for doc_id in preds:
            if doc_id in truths:
                total_mrr += 1 / rank
                break
            rank += 1
    
    mrr = total_mrr / num_queries if num_queries > 0 else 0
    # print("predictions", predictions)
    # print("ground truth",  ground_truths)
    # print("mrr", mrr)
    
    return mrr


def calculate_sum_hits_percentage(predictions, ground_truths):
    hit_ratio = 0
    ranks = []
    for pred, gt in zip(predictions, ground_truths):
        sum_hit = 0
        rank = []
        for idx, p in enumerate(pred):
            if p in gt:
                # print(f"found {p} in {gt}")
                sum_hit += 1
                rank.append(idx)
        
        current_hit_ratio = sum_hit / len(gt)
        ranks.append({ "recalled_idx": rank, "len(prediction)": len(pred), "len(gt)": len(gt), "hit_rate": current_hit_ratio})
        # print("current hit ratio", current_hit_ratio)
        hit_ratio += current_hit_ratio
    return hit_ratio / len(ground_truths) * 100, ranks


def calculate_cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)


def calculate_euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points using scikit-learn.

    Parameters:
    point1 (list or array-like): Coordinates of the first point.
    point2 (list or array-like): Coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    point1 = [point1]
    point2 = [point2]
    
    distance = euclidean_distances(point1, point2)[0][0]
    return distance
