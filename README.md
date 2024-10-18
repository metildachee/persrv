# PerSRV: Personalized Sticker Retrieval with Vision-Language Model

To run:
```bash
python3 main.py --experiment predict_by_paddleocr_rawqueries_llava_desc_emotions_zeroshot_keywords_versatility --test_file <path_to_test_file> --similarity_file "/path/to/similarity_file.csv" --output_dir <path_to_output_directory> --counts 10 20 50 --hyp1 0.3 --hyp2 0 --hyp3 1.0 --size 100
```

* `--test_file`: `userid,query,docid` csv file.
* `--similarity_file`: `userid,clusterid,centroid_vector,centroid_docid,threshold` csv file.
* The expected data is in the format of:
```json
{
  "clicks_per_doc": 0,
  "clip_cn_embedding": [],
  "filepath": "",
  "id": "",
  "llava34b_image_desc": "",
  "llava34b_image_emotions": "",
  "llava_keywords": "",
  "paddleocr": "",
  "paddleocr_probabilities": 0,
  "paddleocr_raw_queries_embeddings": {
    "vector": []
  },
  "paddleocr_texts": "",
  "paddleocr_texts_embedding": [],
  "raw_queries": "",
  "raw_queries_embeddings": {
    "vector": []
  },
  "type": "",
  "unique_users_per_doc": 0,
  "versatility": 0
}

```

The finetuned model will be released online upon the publication of our paper.