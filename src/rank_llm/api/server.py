import argparse
import logging

import torch
from flask import Flask, jsonify, request

from rank_llm.rerank import (
    IdentityReranker,
    PromptMode,
    get_azure_openai_args,
    get_openai_api_key,
)
from rank_llm.rerank.listwise import RankListwiseOSLLM, SafeOpenai
from rank_llm.rerank.listwise.rank_fid import RankFiDDistill, RankFiDScore
from rank_llm.rerank.pointwise.monot5 import MonoT5
from rank_llm.retrieve import RetrievalMethod, RetrievalMode
from rank_llm.retrieve_and_rerank import retrieve_and_rerank

""" API URL FORMAT

http://localhost:{host_name}/api/model/{model_name}/index/{index_name}/{retriever_base_host}?query={query}&hits_retriever={top_k_retriever}&hits_reranker={top_k_reranker}&qid={qid}&num_passes={num_passes}&retrieval_method={retrieval_method}

hits_retriever, hits_reranker, qid, and num_passes are OPTIONAL
Default to 20, 10, None, and 1 respectively

"""


def create_app(model, port, use_azure_openai=False):
    app = Flask(__name__)

    global default_agent
    default_agent = None
    print(model)
    # Load specified model upon server initialization
    if model == "first_mistral":
        print(f"Loading {model} model...")
        default_agent = RankListwiseOSLLM(
            model=f"castorini/first_mistral",
            name=model,
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            device="cuda",
            num_gpus=1,
            variable_passages=True,
            window_size=20,
            system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
            use_alpha=True,
            use_logits=True,
            vllm_batched=True,
        )
    elif "monot5" in model:
        print(f"Loading {model} model...")
        default_agent = MonoT5(
            model=f"castorini/{model}-3b-msmarco-10k",
            prompt_mode=PromptMode.MONOT5,
            context_size=512,
            device="cuda",
            batch_size=64,
        )
    elif "lit5_distill" in model.lower():
        if "v2" in model.lower():
            window_size = 100
            step_size = 50
        else:
            window_size = 20
            step_size = 10

        full_path_mapping = {
            "lit5_distill_v2": "castorini/LiT5-Distill-large-v2",
            "lit5_distill": "castorini/LiT5-Distill-large",
        }
        full_model_path = full_path_mapping[model.lower()]

        print(f"Loading {model} model...")
        default_agent = RankFiDDistill(
            model=full_model_path,
            context_size=150,
            prompt_mode=PromptMode.LiT5,
            num_few_shot_examples=0,
            window_size=window_size,
            step_size=step_size,
            precision="bfloat16",
            device="cuda",
            batched=False,
        )
    elif "lit5_score" in model.lower():
        print(f"Loading {model} model...")
        full_model_path = "castorini/LiT5-Score-large"
        default_agent = RankFiDScore(
            model=full_model_path,
            context_size=150,
            prompt_mode=PromptMode.LiT5,
            num_few_shot_examples=0,
            window_size=100,
            precision="bfloat16",
            device="cuda",
            batched=False,
        )
    elif model == "rank_zephyr":
        print(f"Loading {model} model...")
        default_agent = RankListwiseOSLLM(
            model=f"castorini/{model}_7b_v1_full",
            name=model,
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            device="cuda",
            num_gpus=1,
            variable_passages=True,
            window_size=20,
            vllm_batched=True,
            system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        )
    elif model == "rank_vicuna":
        print(f"Loading {model} model...")
        default_agent = RankListwiseOSLLM(
            model=f"castorini/{model}_7b_v1",
            name=model,
            context_size=4096,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            device="cuda",
            num_gpus=1,
            variable_passages=False,
            window_size=20,
            vllm_batched=True,
        )
    elif "gpt" in model:
        print(f"Loading {model} model...")
        openai_keys = get_openai_api_key()
        print(openai_keys)
        default_agent = SafeOpenai(
            model=model,
            context_size=8192,
            prompt_mode=PromptMode.RANK_GPT,
            num_few_shot_examples=0,
            keys=openai_keys,
            **(get_azure_openai_args() if use_azure_openai else {}),
        )
    elif model in ["identity_reranker", "unspecified", "identity"]:
        print(f"Loading {model} model...")
        default_agent = IdentityReranker()
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Start server
    @app.route(
        "/api/model/<string:model_path>/index/<string:dataset>/<string:retriever_host>",
        methods=["GET"],
    )
    def search(model_path, dataset, retriever_host):
        """retrieve and rerank (search)

        Args:
            - model_path (str): name of reranking model (e.g., rank_zephyr)
            - dataset (str): dataset from which to retrieve
            - retriever_host (str): host of Anserini API
        """

        # query to search for
        query = request.args.get("query", type=str)
        # search all of dataset and return top k candidates
        top_k_retrieve = request.args.get("hits_retriever", default=20, type=int)
        # rerank top_k_retrieve candidates from retrieve stage and return top_k_rerank candidates
        top_k_rerank = request.args.get("hits_reranker", default=10, type=int)
        # qid of query
        qid = request.args.get("qid", default=None, type=str)
        # number of passes reranker goes through
        num_passes = request.args.get("num_passes", default=1, type=int)
        # retrieval method to use
        retrieval_method = request.args.get(
            "retrieval_method", default="bm25", type=str
        )

        if "bm25" in retrieval_method.lower():
            _retrieval_method = RetrievalMethod.BM25
        elif "lance_fts" in retrieval_method.lower():
            _retrieval_method = RetrievalMethod.LANCE_FTS
        elif "lance_arctic_embed_l" in retrieval_method.lower():
            retrieval_method = RetrievalMethod.LANCE_ARCTIC_EMBED_L
        elif "lance_hybrid" in retrieval_method.lower():
            retrieval_method = RetrievalMethod.LANCE_HYBRID
        else:
            return jsonify({"error": str("Retrieval method must be BM25")}), 500

        # If the request model is not the default model
        global default_agent
        if default_agent is not None and model_path != default_agent.get_name():
            # Delete the old agent to clear up the CUDA cache
            del default_agent  # this line is required for clearing the cache
            torch.cuda.empty_cache()
            default_agent = None
        try:
            # calls Anserini retriever API and reranks
            (response, agent) = retrieve_and_rerank(
                dataset=dataset,
                retrieval_mode=RetrievalMode.DATASET,
                query=query,
                model_path=model_path,
                host="http://localhost:" + retriever_host,
                interactive=True,
                top_k_rerank=top_k_rerank,
                top_k_retrieve=top_k_retrieve,
                qid=qid,
                populate_exec_summary=False,
                default_agent=default_agent,
                num_passes=num_passes,
                retrieval_method=retrieval_method,
                print_prompts_responses=False,
            )

            # set the default reranking agent to the most recently used reranking agent
            default_agent = agent

            return jsonify(response[0]), 200
        except Exception as e:
            logging.error(f"Error during search: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return app, port


def main():
    parser = argparse.ArgumentParser(description="Start the RankLLM Flask server.")
    parser.add_argument(
        "--model",
        type=str,
        default="rank_zephyr",
        help="The model to load (e.g., rank_zephyr).",
    )
    parser.add_argument(
        "--port", type=int, default=8082, help="The port to run the Flask server on."
    )
    parser.add_argument(
        "--use_azure_openai", action="store_true", help="Use Azure OpenAI API."
    )
    args = parser.parse_args()

    app, port = create_app(args.model, args.port, args.use_azure_openai)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
