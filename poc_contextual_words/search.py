
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

from voyager import Index, Space
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import spacy
import pandas as pd

import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_words(text: str, spacy_model: spacy.Language) -> List[str]:
    """
    Returns a list of words in the text

    Args:
        text (str): The text to get the words from
        spacy_model (spacy.Language): The spacy model to use to get the words

    Returns:
        list: A list of words in the text
    """

    doc = spacy_model(text)
    words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

    return words


def tokenize_text(text: str, tokenizer: BertTokenizer) -> List[str]:
    """
    Tokenizes a text using a BERT tokenizer. Converts an extended piece of text into a list of words & subwords

    Note that this uses a BERT tokenizer, which is a subword tokenizer. This means that the resulting tokens ARE NOT
    the same as the list of words that make up the text. It is possible that more than 1 token combine to form a word.

    Args:
        text (str): The text to tokenize
        tokenizer (BertTokenizer): The tokenizer to use

    Returns:
        list: A list of tokens
    """

    tokens = tokenizer.tokenize(text)

    return tokens


def embed_tokens(tokens: List[str], tokenizer: BertTokenizer, model: BertModel, tokens_to_keep: int = None) -> np.ndarray:
    """
    Embeds a text using a BERT model. Averages together the embeddings of the tokens to result in an array of size
    (768)

    Args:
        tokens (list): A list of tokens to embed
        tokenizer (BertTokenizer): The tokenizer that converts the string tokens into the IDs that the model can understand
        model (BertModel): The model that does the embedding
        tokens_to_keep (int): The number of tokens to keep. If None, then all tokens are kept
    Returns:

    """

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state[0].numpy()

    if tokens_to_keep is not None:
        embeddings = embeddings[0:tokens_to_keep, :]

    embeddings = np.mean(embeddings, axis=0)
    return embeddings

def embed_words(word_to_tokens: Dict[int, List[int]], tokens: List[str], tokenizer: BertTokenizer, model: BertModel) -> Dict[int, np.ndarray]:
    """
    Embeds a list of words using a BERT model. Averages together the embeddings of the tokens that make up each word to
    result in an array of size (768) for each word.

    Args:
        word_to_tokens (dict): A dictionary where the key is the index of the word and the value is a list of token
        indices that comprise that word
        tokens (list): A list of tokens that corresponds to the indices in word_to_tokens
        tokenizer (BertTokenizer): The tokenizer that converts the string tokens into the IDs that the model can understand
        model (BertModel): The model that does the embedding

    Returns:
        dict: A dictionary where the key is the index of the word and the value is the embedding of the word
    """

    embeddings = {}
    for word_index, token_indices in word_to_tokens.items():
        tokens_to_embed = [tokens[token_index] for token_index in token_indices]
        embeddings[word_index] = embed_tokens(tokens_to_embed, tokenizer, model)

    return embeddings


def map_tokens_to_words(tokens: List[str], words: List[str]) -> Dict[int, List[int]]:
    """
    Maps tokens to words. Returns a dictionary where the key is the index of the word and the value is list of token
    indices that comprise that word. This function assumes that the tokens are always the same length or shorter than
    a word AND that the tokens that make up a word comprise that word and nothing more.


    TODO: Write tests to verify that those assumptions are always true.

    Args:
        tokens (list): A list of tokens
        words (list): A list of words

    Returns:
        list: A list of indices where the ith element is the index of the word that the ith token corresponds to
    """

    word_to_tokens = {}
    last_token = -1

    for word_index in range(len(words)):

        #This loops through the tokens, concatenating them as it goes, to see if the concatenation equals the word.
        # If not, then it starts with the next token and tries again. E.g. if the tokens are [';', 'delicious', 'ness']
        # and the word is 'deliciousness', then it checks ';', then ';delicious', then ';deliciousness',
        # and then 'delicious', and finally 'deliciousness',
        while last_token < len(tokens):
            token_index = last_token + 1
            token_string = ''
            while token_index < len(tokens):
                token_string += tokens[token_index].replace('#', '').lower()
                if token_string == words[word_index]:
                    word_to_tokens[word_index] = list(range(last_token + 1, token_index + 1))
                    last_token = token_index
                    break
                else:
                    token_index += 1
            if token_index == len(tokens):
                last_token += 1
            else:
                break

    return word_to_tokens

def load_dictionary(path: str, tokenizer: BertTokenizer, model: BertModel) -> pd.DataFrame:
    """
    Loads the given dictionary from a file. It makes all words lowercase and adds a line number column.

    Args:
        path (str): The path to the file
        tokenizer (BertTokenizer): The tokenizer to use
        nodel (BertModel): The model to use

    Returns:
        pd.DataFrame: The dictionary
    """

    dictionary = pd.read_csv(path)
    dictionary['word'] = dictionary['word'].apply(lambda x: x.lower())
    dictionary['line_number'] = range(2, len(dictionary.index) + 2)

    dictionary['tokens'] = dictionary['word'].apply(lambda x: tokenize_text(x, tokenizer))
    dictionary['tokens2'] = dictionary['definition'].apply(lambda x: tokenize_text(x, tokenizer))
    dictionary['num_tokens'] = dictionary['tokens'].apply(lambda x: len(x))

    #dictionary['all_tokens'] = dictionary.apply(lambda x: x['tokens'] + [':'] + x['tokens2'], axis=1)
    #Presumably this would work better, but with BERT this seems to work worse than just embedding the word

    dictionary['all_tokens'] = dictionary['tokens']

    dictionary['embeddings'] = dictionary.apply(lambda x: embed_row(x, tokenizer, model), axis=1)

    return dictionary


def create_search_space(dictionary: pd.DataFrame) -> Index:
    """
    Creates a vector search space from the given dictionary

    Args:
        dictionary (pd.DataFrame): The dictionary (with embeddings) to create the search space from

    Returns:
        Index: The search space
    """
    if len(dictionary.index) == 0:
        print('Error: dictionary does not contain any words.')
        return None

    index = Index(Space.Euclidean, num_dimensions=len(dictionary['embeddings'].iloc[0]))
    index.add_items(np.array(list(dictionary['embeddings'].values)), dictionary['line_number'].values)

    return index


def conduct_vector_searches(index: Index, embeddings: Dict[int, np.ndarray], number_results: int = 5, threshold: float = 1) -> Dict[int, int]:
    """
    Conducts searches for the embeddings in the search space

    Args:
        index (Index): The search space
        embeddings (dict): A dictionary where the key is the index of the word and the value is the embedding of the
            word
        number_results (int): The number of results for each word to return
        threshold (float): The threshold for a match. A distance above this value results in no match for the word

    Returns:
        dict: A dictionary where the key is the index of the word and the value is a list of the line numbers in the dictionary CSV
            of the closest matches. If there is no match, the value is an empty list.
    """

    search_results = {}
    for word_index, embedding in embeddings.items():
        neighbors, distances = index.query(embedding, k=number_results)
        search_results[word_index] = []

        for i in range(len(distances)):
            if distances[i] < threshold:
                search_results[word_index].append(neighbors[i])

    return search_results


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

        input_text=llm_tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = llm_tokenizer.encode(input_text, return_tensors="pt").to(device)
        llm_outputs = llm_model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)

        results.append(parse_response(llm_tokenizer.decode(llm_outputs[0])))

    return results


def embed_row(row: pd.Series, tokenizer: BertTokenizer, model: BertModel) -> np.ndarray:
    """
    Embeds a row of a dataframe

    Args:
        row (pd.Series): The row to embed
        tokenizer (BertTokenizer): The tokenizer to use
        model (BertModel): The model to use

    Returns:
        np.ndarray: The embedding of the row
    """

    embedding = list(embed_tokens(row['all_tokens'], tokenizer, model, tokens_to_keep=row['num_tokens']))

    return embedding


def parse_response(text: str) -> str:
    """Parses a response from the model, returning either the
    parsed list with the tool calls parsed, or the
    model thought or response if couldn't generate one.

    Args:
        text: Response from the model.
    """
    pattern = r"<|im_start|>assistant\n(.*?)<|im_end|>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        matches = [m for m in matches if len(m) > 0]
        return matches[0]
    return text


def main():
    model_name = 'bert-base-uncased'
    spacy_model_name = 'en_core_web_md'
    llm_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
 #"meta-llama/Meta-Llama-3.1-8B-Instruct"

    device = 'cpu'

    logger.info('Using model: %s' % model_name)
    logger.info('Using spacy model: %s' % spacy_model_name)
    logger.info('Using LLM model: %s' % llm_model_name)


    model = BertModel.from_pretrained(model_name) #The model to create embeddings with
    tokenizer = BertTokenizer.from_pretrained(model_name)

    spacy_model = spacy.load(spacy_model_name) #the model to separate the text into words

    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)


    logger.info('Loading dictionary')
    dictionary = load_dictionary('./dictionary.csv', tokenizer, model) #The dictionary of words & variants to point to

    logger.info('Creating search space')
    index = create_search_space(dictionary) #The search space

    texts = ['Apple revenue grew. Apples are delicious tasting fruit.',
             'The quick brown fox jumps over the lazy dog.',
             'The balance beam is the hardest discipline in gymnastics.',
             'The Amazon rainforest has a lot of trees.']

    for text in texts:
        print('**********')
        logger.info('Getting words')
        words = get_words(text, spacy_model)

        logger.info('Tokenizing text')
        tokens = tokenize_text(text, tokenizer)

        token_mapping = map_tokens_to_words(tokens, words)

        logger.info('Embedding words')
        embedding_map = embed_words(token_mapping, tokens, tokenizer, model)

        logger.info('Conducting searches for words')
        search_results = conduct_vector_searches(index, embedding_map, threshold=100)

        conduct_llm_searches(dictionary, search_results, text, words, llm_model, llm_tokenizer, device)
    
#        for word_index, line_number in search_results.items():
#            if line_number is not None:
#                print(f'{words[word_index]}: has a match in the dictionary at line {line_number}')
#                print(dictionary[dictionary['line_number'] == line_number]['definition'].values[0])
#            else:
#                print(f'{words[word_index]}: has no match in the dictionary')



if __name__ == '__main__':
    main()

