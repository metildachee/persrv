from rank import rank, rank_by_img_similarity
from recall import  create_client
from refinement import tokenise_english_query, is_en_or
import numpy as np
import re

def clean_centroid_vector_string(centroid_vector_str):
    cleaned_str = centroid_vector_str.strip()
    if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
        cleaned_str = cleaned_str[1:-1].strip()
    cleaned_str = re.sub(r'\s+', ' ', cleaned_str)
    return np.array([float(x) for x in cleaned_str.split()])

es = None

def init():
    global es
    global subset_client
    es = create_client()

from recall import match_paddleocr_and_rawqueries_llava_desc_emotions_finetuned_keywords_versatility
from bert import get_embedding
def predict_by_paddleocr_rawqueries_llava_desc_emotions_finetuned_keywords_versatility(query, uid, emotions_boost=0.1, desc_boost=0.1, llava_keyword_boost=0.1, size=10, userid_centroids_file_path="csv file path"):
    global es
    if es is None:
        init()

    pool = size * 5

    if is_en_or(query):
        query = tokenise_english_query(query)

    vector = get_embedding(query)
    vector = vector.cpu().numpy().tolist()
    response = match_paddleocr_and_rawqueries_llava_desc_emotions_finetuned_keywords_versatility(es, query, emotions_boost=emotions_boost, desc_boost=desc_boost, finetune_boost=llava_keyword_boost, size=pool)
    boosted_image_similarity_results = rank_by_img_similarity(response, uid, userid_centroids_file_path=userid_centroids_file_path)
    rerank_by_img_similarity = rank(boosted_image_similarity_results, size)
    return rerank_by_img_similarity