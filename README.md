# PerSRV: Personalized Sticker Retrieval with Vision-Language Model

### 1. Data preparation
We need to prepare two files: input files and a similarity file.

1. Input file has the the following CSV format;
```
uid,query,docid
0,hello,1234
```

2. To prepare `similarity_file`
```bash 
python3 gen_personalise.py --data_file "/path/to/input_file.csv" --image_dir "/path/to/image_dir" --output_dir "/path/to/output_file"
```
The `similarity_file` will be located at `/path/to/output_file` with headers `userid,clusterid,centroid_vector,centroid_docid,threshold`.

### 2. Run
```bash
python3 main.py --experiment "name of experiment" --input_file "/path/to/input_file.csv" --similarity_file "/path/to/similarity_file.csv" --output_dir "/path/to/output_dir" --counts 10 20 50 --hyp1 0.3 --hyp2 0 --hyp3 1.0 --size 100
```

* The expected data is in the format of;
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