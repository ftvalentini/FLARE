"""Reduced version of https://github.com/ftvalentini/FLARE/blob/main/prep.py
that only creates index with ElasticSearch BM25 and makes changes to read
from tsv instead of json.
"""

import logging
import csv
from typing import List, Tuple, Any, Union, Dict, Set, Callable
import argparse
import random
import json
import time
import glob
from tqdm import tqdm


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", "%m/%d/%Y %H:%M:%S")
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def build_elasticsearch(
    beir_corpus_file_pattern: str,
    index_name: str,
    get_id: Callable = None,
):
    beir_corpus_files = glob.glob(beir_corpus_file_pattern)
    logger.info(f'number of files = {len(beir_corpus_files)}')
    from beir.retrieval.search.lexical.elastic_search import ElasticSearch
    config = {
        "hostname": 'localhost',
        "index_name": index_name,
        "keys": {"title": "title", "body": "txt"},
        "timeout": 100,
        "retry_on_timeout": True,
        "maxsize": 24,
        "number_of_shards": 'default',
        "language": 'english',
    }
    es = ElasticSearch(config)

    # create index
    logger.info(f'Deleting index {index_name} if it exists')
    es.delete_index()
    time.sleep(5)
    logger.info(f'Creating index {index_name}')
    es.create_index()

    get_id = get_id or (lambda x: str(x['_id']))
    # generator
    def generate_actions():
        for beir_corpus_file in beir_corpus_files:
            with open(beir_corpus_file) as fin:
                reader = csv.reader(fin, delimiter="\t")
                # skip header:
                next(reader)
                # read chunks:
                for row in reader:
                    # print(f"row: {row}")
                    doc = {"_id": row[0], "title": row[2], "text": row[1]}
                    es_doc = {
                        "_id": get_id(doc),
                        "_op_type": "index",
                        "refresh": "wait_for",
                        config['keys']['body']: doc['text'],
                        config['keys']['title']: doc['title'],
                    }
                    yield es_doc

    # index
    # compute amount of docs to index based on the number of lines of each file:
    total_docs = 0
    for beir_corpus_file in beir_corpus_files:
        with open(beir_corpus_file) as f:
            total_docs += sum(1 for line in f) - 1  # skip header
    progress = tqdm(unit='docs', total=total_docs)
    logger.info(f'Adding docs to index {index_name}')
    es.bulk_add_to_index(
        generate_actions=generate_actions(),
        progress=progress)
    # FV NOTE this processes the input file in blocks of 500 docs by default
    # see elasticsearch.helpers.actions.streaming_bulk


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to perform', choices=[
        'eval', 'build_elasticsearch', 'jsonl_to_keyvalue'])
    parser.add_argument('--inp', type=str, default=None, nargs='+', help='input file')
    parser.add_argument('--dataset', type=str, default='2wikihop', help='input dataset', choices=[
        'strategyqa', '2wikihop', 'wikiasp', 'asqa'])
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0301', help='model name', choices=[
        'code-davinci-002', 'gpt-3.5-turbo-0301'])
    parser.add_argument('--out', type=str, default=None, help='output file')
    args = parser.parse_args()

    # set random seed to make sure the same examples are sampled across multiple runs
    random.seed(2022)

    if args.task == 'eval':
        raise NotImplementedError
    
    elif args.task == 'jsonl_to_keyvalue':
        raise NotImplementedError

    elif args.task == 'build_elasticsearch':
        beir_corpus_file_pattern, index_name = args.inp  # 'wikipedia_dpr'
        get_id_default = lambda doc: str(doc['_id'])
        get_id_lm = lambda doc: doc['metadata']['line'] + '.' + str(doc['_id'])
        build_elasticsearch(beir_corpus_file_pattern, index_name, get_id=get_id_default)
