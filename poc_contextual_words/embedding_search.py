""" 
This module contains a class that find the best N definitions for each word in a text
by creating an embedding for each word in the text and each word in a reference dictionary.
It than conducts a nearest-neighbor vector search between each word's embedding and the embeddings
of the words in the reference dictionary.
"""


from typing import List, Dict
import logging
from voyager import Index, Space
from transformers import BertModel, BertTokenizer
import transformers
import torch
import numpy as np
import spacy
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingSearch:
    def __init__(self, model_name: str, spacy_model_name: str):
        self._model_name = model_name
        self._spacy_model_name = spacy_model_name
        self._model = BertModel.from_pretrained(model_name) #The model to create embeddings with
        self._tokenizer = BertTokenizer.from_pretrained(model_name)

        self._spacy_model = spacy.load(spacy_model_name) #the model to separate the text into words

        self.reset()


    def reset(self) -> None:
        self._search_results = None
        self._words = None


    def map_tokens_to_words(self, tokens: List[str], words: List[str]) -> Dict[int, List[int]]:
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

    def separate_words(self, text: str) -> None:
        """
        Returns a list of words in the text

        Args:
            text (str): The text to get the words from
            
        Returns:
            list: A list of words in the text
        """

        doc = self._spacy_model(text)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]

        self._words = words


    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenizes a text using a BERT tokenizer. Converts an extended piece of text into a list of words & subwords

        Note that this uses a BERT tokenizer, which is a subword tokenizer. This means that the resulting tokens ARE NOT
        the same as the list of words that make up the text. It is possible that more than 1 token combine to form a word.

        Args:
            text (str): The text to tokenize
            
        Returns:
            list: A list of tokens
        """

        tokens = self._tokenizer.tokenize(text)

        return tokens


    def embed_tokens(self, tokens: List[str], tokens_to_keep: int = None) -> np.ndarray:
        """
        Embeds a text using a BERT model. Averages together the embeddings of the tokens to result in an array of size
        (768)

        Args:
            tokens (list): A list of tokens to embed
            tokens_to_keep (int): The number of tokens to keep. If None, then all tokens are kept
        Returns:

        """

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        with torch.no_grad():
            outputs = self._model(input_ids)
            embeddings = outputs.last_hidden_state[0].numpy()

        if tokens_to_keep is not None:
            embeddings = embeddings[0:tokens_to_keep, :]

        embeddings = np.mean(embeddings, axis=0)
        return embeddings

    def embed_words(self, word_to_tokens: Dict[int, List[int]], tokens: List[str]) -> Dict[int, np.ndarray]:
        """
        Embeds a list of words using a BERT model. Averages together the embeddings of the tokens that make up each word to
        result in an array of size (768) for each word.

        Args:
            word_to_tokens (dict): A dictionary where the key is the index of the word and the value is a list of token
            indices that comprise that word
            tokens (list): A list of tokens that corresponds to the indices in word_to_tokens

        Returns:
            dict: A dictionary where the key is the index of the word and the value is the embedding of the word
        """

        embeddings = {}
        for word_index, token_indices in word_to_tokens.items():
            tokens_to_embed = [tokens[token_index] for token_index in token_indices]
            embeddings[word_index] = self.embed_tokens(tokens_to_embed)

        return embeddings


    def embed_row(self, row: pd.Series) -> list:
        """
        Embeds a row of a dataframe

        Args:
            row (pd.Series): The row to embed

        Returns:
            list: The embedding of the row
        """

        embedding = list(self.embed_tokens(row['all_tokens'], tokens_to_keep=row['num_tokens']))

        return embedding


    def load_dictionary(self, dictionary: pd.DataFrame) -> pd.DataFrame:
        """
        Loads the given dictionary from a file. It makes all words lowercase and adds a line number column.

        Args:
            dictionary (pd.DataFrame): The dictionary as a pandas Dataframe with the columns 'word' and definition'

        Returns:
            pd.DataFrame: The dictionary
        """

        dictionary['word'] = dictionary['word'].apply(lambda x: x.lower())
        dictionary['line_number'] = range(2, len(dictionary.index) + 2)

        dictionary['tokens'] = dictionary['word'].apply(lambda x: self.tokenize_text(x))
        dictionary['tokens2'] = dictionary['definition'].apply(lambda x: self.tokenize_text(x))
        dictionary['num_tokens'] = dictionary['tokens'].apply(lambda x: len(x))

        #dictionary['all_tokens'] = dictionary.apply(lambda x: x['tokens'] + [':'] + x['tokens2'], axis=1)
        #Presumably this would work better, but with BERT this seems to work worse than just embedding the word

        dictionary['all_tokens'] = dictionary['tokens']

        dictionary['embeddings'] = dictionary.apply(lambda x: self.embed_row(x), axis=1)

        return dictionary

    def create_search_space(self, dictionary: pd.DataFrame) -> Index:
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



    def search(self, text: str, dictionary: dict, threshold=100, number_results=5) -> None:
        """ 
        Searches for the best definition in the dictionaryfor each word in the text to the dictionary

        Args:
            text (str): The text to search
            dictionary (dict): The reference dictionary of words & variants to point to
            threshold (int): The maximum distance in the vector search in order to keep a result. 
                Default is 100
            number_results (int): The number of results (potential definitions) for each word to return. 
                Default is 5
        """
        
        self.reset()
        self.separate_words(text)
        words = self.get_words()

        logger.info('Tokenizing text')
        tokens = self.tokenize_text(text)

        token_mapping = self.map_tokens_to_words(tokens, words)

        logger.info('Embedding words')
        embedding_map = self.embed_words(token_mapping, tokens)

        logger.info('Loading dictionary')
        dictionary = self.load_dictionary(dictionary) #The dictionary of words & variants to point to

        logger.info('Creating search space')
        index = self.create_search_space(dictionary) #The search space


        search_results = {}
        for word_index, embedding in embedding_map.items():
            neighbors, distances = index.query(embedding, k=number_results)
            search_results[word_index] = []

            for i in range(len(distances)):
                if distances[i] < threshold:
                    search_results[word_index].append(neighbors[i])

        self._search_results = search_results

    def get_search_results(self):
        if self._search_results is None:
            raise Exception('You must call search() first')

        return self._search_results

    def get_words(self):
        if self._words is None:
            raise Exception('You must call search() first')

        return self._words