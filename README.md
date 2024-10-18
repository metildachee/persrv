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
  "clicks_per_doc": 12.5,
  "clip_cn_embedding": [0.0123, 0.0567, 0.0987, ...],  // 512 x 512 
  "filepath": "/images/emojis/sunglasses.png",
  "id": "1001",
  "llava34b_emotions": "joy, confidence",
  "llava34b_image_desc": "A cool emoji wearing sunglasses",
  "llava34b_image_emotions": "happy, excited",
  "llava_keywords": "emoji, sunglasses, cool",
  "opus_ch_caption": "A smiley face emoji with sunglasses.",
  "paddleocr": "smiling face with sunglasses",
  "paddleocr_probabilities": 0.85,
  "paddleocr_raw_queries_embeddings": {
    "vector": [0.1345, 0.2456, 0.0876, ...]  // 512 x 512 
  },
  "paddleocr_texts": "emoji with glasses",
  "paddleocr_texts_embedding": [0.0234, 0.0678, 0.0345, ...],
  "raw_queries": "emoji, smile, cool glasses",
  "raw_queries_embeddings": {
    "vector": [0.2234, 0.4876, 0.3456, ...]
  },
  "type": ".jpg",
  "unique_users_per_doc": 125.5,
  "versatility": 0.75
}
```

The finetuned model will be released online upon the publication of our paper.