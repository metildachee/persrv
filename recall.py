from elasticsearch import Elasticsearch
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

config = {
    'elasticsearch': {
        'url': 'https://localhost:9200',
        'api_key': 'YOUR API KEY',
        'index': 'emoji_data'
    },
}

def create_client(config=config):
    es_config = {
        'hosts': [config['elasticsearch']['url']],
        'verify_certs': False, 
        'api_key': config['elasticsearch']['api_key'] if config['elasticsearch']['api_key'] else None
    }
    es = Elasticsearch(**es_config)
    if not es.ping():
        raise Exception("Elasticsearch cluster is down!")
    return es

def match_by_all_text_versatility(client, query, emotions_boost=0.1, finetune_boost=0.1, desc_boost=0.1, size=10):
    query_body = {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "paddleocr": {
                                        "query": query,
                                        "boost": 1.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "raw_queries": {
                                        "query": query,
                                        "boost": 0.5
                                    }
                                }
                            },
                            {
                                "match": {
                                    "llava34b_image_desc": {
                                        "query": query,
                                        "boost": desc_boost
                                    }
                                }
                            },
                            {
                                "match": {
                                    "llava_keywords": {
                                        "query": query,
                                        "boost": finetune_boost
                                    }
                                }
                            },
                            {
                                "match": {
                                    "llava34b_image_emotions": {
                                        "query": query,
                                        "boost": emotions_boost
                                    }
                                }
                            }
                         ]
                    }
                },
                "functions": [
                    {
                        "field_value_factor": {
                            "field": f"versatility",
                            "factor": 0.3,
                            "modifier": "sqrt", 
                            "missing": 1.05,
                        }
                    }
                ],
                "boost_mode": "sum"
            }
        },
        "_source": ["raw_queries", "easyocr_texts", "llava34b_image_desc", "llava34b_image_emotions", "id", "clip_cn_embedding", "type"]
        }
        
    response = client.search(
        index="emoji_data",
        body=query_body,
        size=size
        )
    hits = []
    for hit in response['hits']['hits']:
        doc = hit['_source']
        doc["score"] = hit['_score']
        doc["source"] = 'match_by_all_text_versatility'
        hits.append(doc)

    return hits