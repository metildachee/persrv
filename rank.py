


from metrics import calculate_cosine_similarity
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

def clean_centroid_vector_string(centroid_vector_str):
    cleaned_str = centroid_vector_str.strip()
    if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
        cleaned_str = cleaned_str[1:-1].strip()
    cleaned_str = re.sub(r'\s+', ' ', cleaned_str)
    return np.array([float(x) for x in cleaned_str.split()])

def rank(docs, count, verbose=False):
    """
    Ranks documents based on their scores in descending order.
    :param docs: List of documents with 'score' field.
    :return: Ranked list of documents.
    """
    # for i, s in enumerate(sort):
    #     print(f'{i}, {s["score"]}, {s["source"]}')
    sort = sorted(docs, key=lambda x: x['score'], reverse=True)
    return sort[:count]


def rank_by_img_similarity(response, userid, image_similarity_boost_mode="multiply", image_similarity_weight=0.3, userid_to_threshold_csv="path to csv"):
    user_data = pd.read_csv(userid_to_threshold_csv)
    user_data['userid'] = user_data['userid'].astype(str)
    user_rows = user_data[user_data['userid'] == userid]
    if user_rows.empty:
        print(f"No data found for user.")
        return []

    boosted_image_similarity_results = []
    for hit in response:
        clip_cn_embedding = np.array(hit['clip_cn_embedding'])
        curr_score = 0
        best_score = hit['score']
        final_similarity = 0
        final_distance = 0
        final_cluster = -1

        for _, row in user_rows.iterrows():
            centroid_vector_str = row['centroid_vector']
            user_centroid = clean_centroid_vector_string(centroid_vector_str)
            threshold = row['threshold']
            
            similarity = cosine_similarity([user_centroid], [clip_cn_embedding])[0][0]
            distance = euclidean(user_centroid, clip_cn_embedding)
            cluster_id = row['clusterid']

            if distance > threshold:
                continue
            
            if image_similarity_boost_mode == "sum":
                curr_score =  hit['score'] + image_similarity_weight * similarity
            elif image_similarity_boost_mode == "multiply":
                curr_score = hit['score'] * (1 + image_similarity_weight * similarity)

            if curr_score > best_score:
                best_score = curr_score
                final_similarity = similarity
                final_distance = distance
                final_cluster = cluster_id
        
        if best_score > hit['score']:
            hit['score'] = best_score
            hit['euclidean_distance'] = final_distance
            hit['cosine_similarity'] = final_similarity
            hit['cluster_id'] = final_cluster
            hit['source'] = "search_easyocr_and_rawqueries_popularity_rank_img_sim"
        boosted_image_similarity_results.append(hit)
    return boosted_image_similarity_results

