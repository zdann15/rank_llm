"""
Genereate retrieve results json cache for a TREC run file.

TREC run file format:
qid Q0 docid rank score run_id

Corpus format:
did\tcontent\n

Query format:
qid\tquery\n

Output format:
{"query": {"qid": "264014", "text": "how long is life cycle of flea"},  "candidates": [ {"doc": "..." (pyserini raw output, a dictionary),        "docid": "5611210",        "score": 15.780599594116211}, ... ]}
     

Usage:
    python3 scripts/generate_retrieve_results_jsonl_cache.py --trec_file <run_file_path> \
        --collection_file <collection_file_path> \
        --query_file <query_file_path> \
        --output_file <output_file_path>
Example:
    python3 scripts/generate_retrieve_results_jsonl_cache.py --trec_file retrieve_results/BM25/trec_results_train_random_n1000.txt \
        --collection_file /store/scratch/rpradeep/msmarco-v1/collections/official/collection.tsv \
        --query_file /store/scratch/rpradeep/msmarco-v1/collections/official/queries.train.tsv \
        --output_file retrieve_results/BM25/retrieve_results_train_random_n1000.jsonl \
        --topk 20
    # msmarco-v2.1-doc-segmented example
    python3 src/rank_llm/scripts/generate_retrieve_results_jsonl_cache.py --trec_file ${ANSERINI}/runs/fs4_bm25+rocchio_snowael_snowaem_gtel.raggy-dev_top3000.txt \
        --collection_file msmarco-v2.1-doc-segmented \
        --query_file rag24.raggy-dev \
        --output_file ${PYSERINI_CACHE}/runs/retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel.raggy-dev_top3000.jsonl \
        --topk 3000 \
        --num_threads 64
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from pyserini.search import LuceneSearcher, get_qrels, get_topics
from tqdm import tqdm

from rank_llm.retrieve import TOPICS

sys.path.append(os.getcwd())


def load_trec_file(file_path, topk=100, qrels=None):
    data = defaultdict(list)
    with open(file_path, "r") as file:
        for line in file:
            qid, _, docid, rank, score, _ = line.strip().split()
            
            if qrels and type(list(qrels.items())[0][0]) is int:
                qid = int(qid)
            if qrels is not None and qid not in qrels:
                continue
            if int(rank) > topk:
                continue
            data[qid].append(
                {"qid": qid, "docid": docid, "rank": int(rank), "score": float(score)}
            )
    print(f"Loaded run for {len(data)} queries from {file_path}")
    return data


def load_tsv_file(file_path):
    examples = {}
    with open(file_path, "r") as file:
        for line in file:
            exid, text = line.strip().split("\t")
            examples[exid] = text
    return examples


def load_pyserini_indexer(collection_file, trec_data, topk):
    examples = {}
    # index_reader = LuceneSearcher.from_prebuilt_index(collection_file)
    for qid, hits in tqdm(trec_data.items()):
        rank = 0
        for hit in hits:
            rank += 1
            if rank > topk:
                break
            document = index_reader.doc(hit["docid"])
            content = json.loads(document.raw())
            if "title" in content:
                content = content["title"].strip() + ". " + content["text"]
            elif "contents" in content:
                content = content["contents"]
            else:
                content = content["passage"]
            examples[hit["docid"]] = content
    print(f"Loaded {len(examples)} examples from {collection_file}")
    return examples


def load_pyserini_indexer_parallel(collection_file, trec_data, topk, num_threads):
    examples = {}       
    all_docids = []
    for i in trec_data:
        all_docids.extend([hit["docid"] for hit in trec_data[i] if hit["rank"] <= topk])
    print(f"Total number of docids: {len(set(all_docids))}")
    index_reader = LuceneSearcher.from_prebuilt_index(collection_file)
    all_docs = index_reader.batch_doc(list(set(all_docids)), threads=num_threads)
    examples = {docid: json.loads(doc.raw()) for docid, doc in all_docs.items()}
    # Drop docid if exists, start_char if exists, and end_char if exists
    for docid in examples:
        if "docid" in examples[docid]:
            del examples[docid]["docid"]
        if "start_char" in examples[docid]:
            del examples[docid]["start_char"]
        if "end_char" in examples[docid]:
            del examples[docid]["end_char"]
    print(f"Loaded {len(examples)} examples from {collection_file}")
    return examples


def generate_retrieve_results(
    trec_file,
    collection_file,
    query_file,
    topk=100,
    output_trec_file=None,
    num_threads=1,
):
    if query_file in TOPICS:
        if TOPICS[query_file] == "dl22-passage":
            query_data = get_topics("dl22")
        elif TOPICS[query_file] == "dl21-passage":
            query_data = get_topics("dl21")
        else:
            query_data = get_topics(TOPICS[query_file])
        for qid, query in query_data.items():
            query_data[qid] = query["title"]
        qrels = get_qrels(TOPICS[query_file])
        print(f"Loaded {len(query_data)} queries from {query_file}")
        print(f"Loaded {len(qrels)} qrels from {query_file}")
    else:
        query_data = load_tsv_file(query_file)
        qrels = None
    trec_data = load_trec_file(trec_file, topk, qrels)
    print(f"Loaded {len(trec_data)} queries from {trec_file}")
    if collection_file in [
        "msmarco-v2.1-doc-segmented",
        "msmarco-v2-passage",
        "beir-v1.0.0-trec-covid.flat",
        "beir-v1.0.0-trec-news.flat",
    ]:
        print(f"Loading index from {collection_file}")
        collection_data = load_pyserini_indexer_parallel(
            collection_file, trec_data, topk, num_threads
        )
    else:
        collection_data = load_tsv_file(collection_file)

    results = []
    for qid, hits in tqdm(trec_data.items()):
        if qid not in query_data:
            qid = int(qid)
        query = query_data[qid]
        candidates = []
        for hit in hits:
            candidates.append(
                {
                    "doc": collection_data[hit["docid"]],
                    "docid": hit["docid"],
                    "score": hit["score"],
                }
            )
        results.append({"query": {"qid": qid, "text": query}, "candidates": candidates})
    if output_trec_file is not None:
        runtag = os.path.basename(output_trec_file).replace(".trec", "").replace(".txt", "").replace(".run", "")
        with open(output_trec_file, "w") as file:
            for result in results:
                for hit in result["candidates"]:
                    file.write(
                        f"{result['query']['qid']} Q0 {hit['docid']} {hit['score']} {runtag}\n"
                    )
    return results


def write_output_file(output_file_path, data):
    with open(output_file_path, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_file", required=True)
    parser.add_argument("--collection_file", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--output_trec_file", type=str, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    results = generate_retrieve_results(
        args.trec_file,
        args.collection_file,
        args.query_file,
        args.topk,
        args.output_trec_file,
        args.num_threads,
    )
    write_output_file(args.output_file, results)


if __name__ == "__main__":
    main()
