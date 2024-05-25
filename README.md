# Tune RAG Parameters and track with Mlflow

This project shows how to tune different parameters of a RAG application(chunk size, top k etc) and track all the metrics, parameters and artifacts using mlflow. This helps to compare different techniques and iterate quickly over experiments.


## How to run

```bash
python tune_rag.py
```

You can pass in CLI parameters this way

```bash
python tune_rag.py --chunk_size 256 --top_k 5 --model_name 'DifferentModel/zephyr-7b-beta' --embedder_name 'BAAI/bge-large-en-v1.5' --dataset_name 'new_dataset.json'
```

Access the mlflow dashbaord using

```bash
mlflow ui
```
