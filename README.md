# PerSRV: Personalized Sticker Retrieval with Vision-Language Model
  - [Running the Code](#running-the-code)
    - [1. Data preparation](#1.-data-preparation)
    - [2. Run](#2.-run)
  - [Huggingface Model](#huggingface-model)
  - [Citation](#citation)

## Running the Code
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

## Huggingface Model
If you are interested in eliciting query keywords from stickers, please try out our huggingface model, released [here](https://huggingface.co/metchee/persrv).

## Citation
If you find our code or model helpful please do cite us! :-)
```
@misc{chee2024persrvpersonalizedstickerretrieval,
  title={PerSRV: Personalized Sticker Retrieval with Vision-Language Model},
  author={Heng Er Metilda Chee and Jiayin Wang and Zhiqiang Guo and Weizhi Ma and Min Zhang},
  year={2024},
  eprint={2410.21801},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2410.21801},
}
```
