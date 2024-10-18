from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

tokenizer = None 
model = None 

def init():
    global tokenizer 
    global model
    
    tokenizer = AutoTokenizer.from_pretrained("<path to bert tokenizer>")
    model = AutoModel.from_pretrained("<path to bert model>")

def get_embedding(text):
    if tokenizer is None or model is None:
        init()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    inputs = {key: value for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state
    pooled_output = embeddings.mean(dim=1).squeeze()
    
    return pooled_output.cpu()


def compare_similarity_embeddings(embedding1, embedding2s):
    similarities = []
    distances = []
    
    for embedding in embeddings2:
        similarity = 1 - cosine(embedding1, embedding)
        distance = 1 - similarity
        # print(f"{text1}, {text2}, cosine Similarity: {similarity}")
        # print(f"Distance: {distance}"
        similarities.append(similarity)
        distances.append(distance)
    return similarities, distances

def compare_similarity_text(text1, text2s, verbose=False):
    similarities = []
    distances = []
    embedding1 = get_embedding(text1)
    for text2 in text2s:
        embedding2 = get_embedding(text2)
        similarity = 1 - cosine(embedding1, embedding2)
        distance = 1 - similarity
        if verbose:
            print(f"{text1}, {text2}, cosine Similarity: {similarity}")
            print(f"Distance: {distance}")
        similarities.append(similarity)
        distances.append(distance)
    return similarities, distances

def load_embeddings(filepath):
    docid_to_embeddings = torch.load(filepath)
    