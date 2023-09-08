from rank_llm import RankLLM, PromptMode
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template, add_model_args
from tqdm import tqdm
from pyserini_retriever import PyseriniRetriever
from topics_dict import TOPICS
import argparse
from ftfy import fix_text
import re

def replace_number(s):
    return re.sub(r'\[(\d+)\]', r'(\1)', s)

class RankVicuna(RankLLM):
    def __init__(self, model, context_size, dataset, prompt_mode, device, num_gpus):
        super().__init__(model, context_size, dataset, prompt_mode)
        # ToDo: Make repetition_penalty and max_new_tokens configurable
        self.llm_, self.tokenizer_ = load_model(model, device=device, num_gpus=num_gpus)
        self.device_ = device

    def run_llm(self, messages):
        inputs = self.tokenizer_([messages])
        inputs = {k: torch.tensor(v).to(self.device_) for k, v in inputs.items()}
        output_ids = self.llm_.generate(
            **inputs,
            do_sample=False,
            temperature=0
        )

        if self.llm_.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self.tokenizer_.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs, output_ids.size(0)

    def num_output_tokens(self):
        return 200
    
    def _add_prefix_prompt(self, query, num, conv):
        return f'I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n'

    def _add_post_prompt(self, query, num, conv):
            return f'Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain.'

    def create_prompt(self, retrieved_result, rank_start=0, rank_end=100):
        query = retrieved_result['query']
        num = len(retrieved_result['hits'][rank_start: rank_end])

        max_length = 300
        while True:
            # conv = get_conversation_template(self.model_)
            conv = get_conversation_template('vicuna_v1.1')
            prefix = self._add_prefix_prompt(query, num, conv)
            rank = 0
            input_context = f"{prefix}\n"
            for hit in retrieved_result['hits'][rank_start: rank_end]:
                rank += 1
                content = hit['content']
                content = content.replace('Title: Content: ', '')
                content = content.strip()
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = ' '.join(content.split()[:int(max_length)])
                input_context += f'[{rank}] {replace_number(content)}\n'
            
            input_context += (self._add_post_prompt(query, num, conv))
            conv.append_message(conv.roles[0], input_context)
            prompt = conv.get_prompt() + " ASSISTANT:"
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(1, (num_tokens - self.max_tokens() + self.num_output_tokens()) // (rank_end - rank_start))
        return prompt, self.get_num_tokens(prompt)

    def get_num_tokens(self, messages):
        return len(self.tokenizer_([messages]))
    
    def cost_per_1k_token(self, input_token:bool):
        return 0

def main(args):
    context_size = 4096
    dataset = args.dataset
    prompt_mode = PromptMode.RANK_GPT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = RankVicuna(args.model_path, context_size, dataset, prompt_mode, device, args.num_gpus)
    retriever = PyseriniRetriever(dataset)
    from pathlib import Path

    candidates_file = Path(f'retrieve_results/retrieve_results_{dataset}.json')
    if not candidates_file.is_file():
        print('Retrieving:')
        retriever.retrieve_and_store(k=100)
    else:
        print('Reusing existing retrieved results.')
    import json
    with open(candidates_file, 'r') as f:
        retrieved_results = json.load(f)

    print('\nReranking:')
    rerank_results = []
    input_token_counts = []
    output_token_counts = []
    aggregated_prompts = []
    aggregated_responses = [] 
    for result in tqdm(retrieved_results):
        rerank_result, in_token_count, out_token_count, prompts, responses  = agent.sliding_windows(result, rank_start=0, rank_end=args.rank_end, window_size=20, step=10)
        rerank_results.append(rerank_result)
        input_token_counts.append(in_token_count)
        output_token_counts.append(out_token_count)
        aggregated_prompts.extend(prompts)
        aggregated_responses.extend(responses)
    print(f'input_tokens_counts={input_token_counts}')
    print(f'total input token count={sum(input_token_counts)}')
    print(f'output_token_counts={output_token_counts}')
    print(f'total output token count={sum(output_token_counts)}')
    file_name = agent.write_rerank_results(rerank_results, input_token_counts, output_token_counts, aggregated_prompts, aggregated_responses, args.model_name)
    from trec_eval import EvalFunction
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.1', TOPICS[dataset], file_name])
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.5', TOPICS[dataset], file_name])
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', TOPICS[dataset], file_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--rank_end', type=int, default=20, help='Rank end')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()
    main(args)
