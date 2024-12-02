
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
"""

from typing import List, Dict
import logging
import re
import json 

from voyager import Index, Space
from transformers import BertModel, BertTokenizer, AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import numpy as np
import spacy
import pandas as pd

import embedding_search


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)









def conduct_llm_searches(dictionary: pd.DataFrame, vector_search_results: Dict[int, List[int]], text: str, words: List[str], llm_model, llm_tokenizer, device) -> Dict[int, int]:
    """
    Conducts searches for the embeddings in the search space

    Args:
        index (Index): The search space
        dictionary (pd.DataFrame): The dictionary of potential word variants
        vector_search_results (dict): A dictionary where the key is the index of the word and the value is a list of the
            line numbers in the dictionary CSV of the closest matches.
    Returns:
        dict: A dictionary where the key is the index of the word and the value is a list of the line numbers in the dictionary CSV
            of the closest matches. If there is no match, the value is an empty list.
    """

    results = []
    for i in list(vector_search_results.keys()):
        if len(vector_search_results[i]) == 0:
            print(f'{words[i]}: has no match in the dictionary')
            results.append(None)
            continue

        word = words[i]

        prompt = f"""What is the best definition of "{word}" in the text "{text}"? The options are: """

        for line_number in vector_search_results[i]:
            definition = dictionary[dictionary['line_number'] == line_number]['definition'].values[0]
            prompt += f"""
{line_number}: {definition}"""

        prompt += """
    Output only the definition number.
    """
        messages = [
            {"role": "system", "content": "You are a tool to find the best definition of a word in a text. For every word, you will be given several options of definitions. Choose the best option of the definition for the word as it is used in that text."},
            {"role": "user", "content": prompt}
        ]

        print(prompt)
        input_text=llm_tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = llm_tokenizer.encode(input_text, return_tensors="pt").to(device)
        llm_outputs = llm_model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)

        # The model output is a 2 dimensional array, but only one dimension is filled,
        # so llm_outputs[0] is the entirety of the output, just with that extra dimension
        # removed 
        results.append(parse_response(llm_tokenizer.decode(llm_outputs[0])))

    return results




def parse_response(text: str) -> int:
    """Parses a response from the model, returning the model's response as an integer.

    Args:
        text: Response from the model.
    """
    pattern = r"<|im_start|>assistant\n(.*?)<|im_end|>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        matches = [m for m in matches if len(m) > 0]
        try:
            definition_line = int(matches[0])
        except ValueError:
            raise ValueError(f"Could not convert response to integer. Response: {text}")
    return definition_line


def main():
    model_name = 'bert-base-uncased'
    spacy_model_name = 'en_core_web_md'
    llm_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
 #"meta-llama/Meta-Llama-3.1-8B-Instruct"

    device = 'cpu'

    logger.info('Using model: %s' % model_name)
    logger.info('Using spacy model: %s' % spacy_model_name)
    logger.info('Using LLM model: %s' % llm_model_name)

    embedding_searcher = embedding_search.EmbeddingSearch(model_name, spacy_model_name)

    logger.info('Loading LLM model')
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)



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
        search_results = embedding_searcher.search(text, threshold=100)

        results = conduct_llm_searches(dictionary, search_results, text, words, llm_model, llm_tokenizer, device)
        
        print(results)
#        for word_index, line_number in search_results.items():
#            if line_number is not None:
#                print(f'{words[word_index]}: has a match in the dictionary at line {line_number}')
#                print(dictionary[dictionary['line_number'] == line_number]['definition'].values[0])
#            else:
#                print(f'{words[word_index]}: has no match in the dictionary')



if __name__ == '__main__':
    main()

