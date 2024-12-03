"""
This module contains a class that uses a large language model to find the best definition 
for each word in a text. It chooses from the results of the embedding searcher in the file
embedding_search.py.
"""
import logging
from typing import List
import re

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import embedding_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlmSearch:
    def __init__(self, llm_model_name: str, device: str):
        self._device = device
        self._llm_model_name = llm_model_name
        logger.info('Loading LLM model')
        self._llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    
        self._llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)

    def parse_response(self, text: str) -> int:
        """Parses a response from the model, returning the model's response as an integer.

        Args:
            text: Response from the model.

        Returns:
            int: The model's response as an integer
        """
        if self._llm_model_name == "HuggingFaceTB/SmolLM2-1.7B-Instruct":
            pattern = r"<|im_start|>assistant\n(.*?)<|im_end|>"
        else:
            raise Exception(f'Parsing for model "{self._llm_model_name}" not implemented yet')

        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            matches = [m for m in matches if len(m) > 0]
            try:
                definition_line = int(matches[0])
            except ValueError:
                raise ValueError(f"Could not convert response to integer. Response: {text}")
        return definition_line


    def search(self, text: str, dictionary: pd.DataFrame, embedding_searcher: embedding_search.EmbeddingSearch) -> List[int]:
        """
        Outputs the best definition for each word in text according to the LLM. The LLM chooses from the results of the
        embedding searcher (The embedding searcher got the top N definitions, and then LLM chooses the best one from those)

        Args:
            text (str): The text to search
            dictionary (pd.DataFrame): The dictionary of potential word variants
            embedding_searcher (embedding_search.EmbeddingSearch): The embedding searcher that has already conducted a search
                based on embeddings to get the top options for each word

        Returns:
            list: The best definition for each word in text, where each entry is the line number in dictionary of the definition
        """
        results = []
        vector_search_results = embedding_searcher.get_search_results()
        words = embedding_searcher.get_words()

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

        #TODO: Fix the prompt to handle when the same word appears multiple times in the text

            messages = [
                {"role": "system", "content": "You are a tool to find the best definition of a word in a text. For every word, you will be given several options of definitions. Choose the best option of the definition for the word as it is used in that text."},
                {"role": "user", "content": prompt}
            ]

            print(prompt)
            input_text = self._llm_tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self._llm_tokenizer.encode(input_text, return_tensors="pt").to(self._device)
            llm_outputs = self._llm_model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)

            # The model output is a 2 dimensional array, but only one dimension is filled,
            # so llm_outputs[0] is the entirety of the output, just with that extra dimension
            # removed 
            results.append(self.parse_response(self._llm_tokenizer.decode(llm_outputs[0])))

        return results

