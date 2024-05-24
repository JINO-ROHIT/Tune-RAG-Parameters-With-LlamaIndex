import argparse
import asyncio

import mlflow
import nest_asyncio
import pandas as pd


nest_asyncio.apply()
import warnings


warnings.filterwarnings('ignore')

import pandas as pd
import torch
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, set_global_tokenizer
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    generate_question_context_pairs,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, BitsAndBytesConfig


system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided. Only use the context provided and STRICTLY say you dont know if you dont know."
query_wrapper_prompt = "<|USER|>{query_str}<|ASSISTANT|>"

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

llm = HuggingFaceLLM(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        context_window=3900,
        max_new_tokens=256,
        model_kwargs={"quantization_config": bnb_config},
        generate_kwargs={"temperature": 0.1},
        device_map="cuda:0",
    )

set_global_tokenizer(
    AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").encode
)
embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

def parse_args():
    parser = argparse.ArgumentParser(description="Tune RAG Model")
    parser.add_argument('--dataset_dir', type=str, default="./data", help="Directory for the dataset")
    parser.add_argument('--chunk_size', type=int, default=512, help="Chunk size for splitting documents")
    parser.add_argument('--top_k', type=int, default=2, help="Top K similar nodes to retrieve")
    parser.add_argument('--model_name', type=str, default='HuggingFaceH4/zephyr-7b-beta', help="Model name")
    parser.add_argument('--embedder_name', type=str, default='BAAI/bge-small-en-v1.5', help="Embedder name")
    parser.add_argument('--dataset_name', type=str, default='pg_eval_dataset.json', help="Dataset name")
    parser.add_argument('--chunk_questions', type=int, default=1, help="Number of questions per chunk")
    return parser.parse_args()

async def tune_rag(args):
    documents = SimpleDirectoryReader(args.dataset_dir).load_data()
    node_parser = SentenceSplitter(chunk_size=args.chunk_size)
    nodes = node_parser.get_nodes_from_documents(documents)

    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    vector_index = VectorStoreIndex(nodes)
    retriever = vector_index.as_retriever(similarity_top_k=args.top_k)

    qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=args.chunk_questions
    )

    qa_dataset.save_json(args.dataset_name)
    qa_dataset = EmbeddingQAFinetuneDataset.from_json(args.dataset_name)

    metrics = ["mrr", "hit_rate"]
    retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    with mlflow.start_run():
        mlflow.log_metric("Hit Rate", hit_rate)
        mlflow.log_metric("MRR", mrr)

        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("embedder_name", args.embedder_name)
        mlflow.log_param("top_k", args.top_k)
        mlflow.log_param("chunk_size", args.chunk_size)
        mlflow.log_param("chunk_questions", args.chunk_questions)

        mlflow.log_artifact("data/paul_graham_essay.txt")
        mlflow.log_artifact(args.dataset_name)

    print("Run has been completed. View your results at http://127.0.0.1:5000")

if __name__ == "__main__":
    args = parse_args()
    mlflow.set_experiment("rag_tuning")
    asyncio.run(tune_rag(args))
