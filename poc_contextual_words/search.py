
"""


For identifying words, I will assume that words do no mean punctuation or spaces.

I interpret "closely related word" to mean that the two words do not have to be the same stem with possibly different
inflections, but they can be completely different words that mean very similar things (like synonyms). Ideally I would
try different methods (levels of complexity) to see which works best and whether the added complexity, resource use,
latency, etc. is worth it. These levels of complexity include:

1. an exact lookup in the reference dictionary
2. A non-contextual word embedding, like word2vec
3. Contextual embeddings (try several models)

This module contains soome logging statements for demonstrative purposes, but more logging should be added.

Example usage:
    python search.py
"""

from typing import List, Dict
import logging
import re
import json 

from voyager import Index, Space
import transformers
import numpy as np
import spacy
import pandas as pd

import embedding_search
import llm_search


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





def main():
    model_name = 'bert-base-uncased'
    spacy_model_name = 'en_core_web_md'
    llm_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dictionary_path = './dictionary.csv'
 #"meta-llama/Meta-Llama-3.1-8B-Instruct"

    device = 'cpu'

    logger.info('Using model: %s' % model_name)
    logger.info('Using spacy model: %s' % spacy_model_name)
    logger.info('Using LLM model: %s' % llm_model_name)

    embedding_searcher = embedding_search.EmbeddingSearch(model_name, spacy_model_name)
    llm_searcher = llm_search.LlmSearch(llm_model_name, device)

    logger.info('Loading dictionary')
    dictionary = pd.read_csv(dictionary_path)

    logger.info('Creating search space')
    
    texts = ['Apple revenue grew. Apples are delicious tasting fruit.',
             #'The quick brown fox jumps over the lazy dog.',
             'The balance beam is the hardest discipline in gymnastics.',
             #'The Amazon rainforest has a lot of trees.'
             ]

    for text in texts:
        print('**********')
        logger.info('Getting words')
        
        logger.info('Conducting searches for words')
        search_results = embedding_searcher.search(text, dictionary, threshold=100)

        results = llm_searcher.search(text, dictionary, embedding_searcher)
        
        print(results)
#        for word_index, line_number in search_results.items():
#            if line_number is not None:
#                print(f'{words[word_index]}: has a match in the dictionary at line {line_number}')
#                print(dictionary[dictionary['line_number'] == line_number]['definition'].values[0])
#            else:
#                print(f'{words[word_index]}: has no match in the dictionary')



if __name__ == '__main__':
    main()

