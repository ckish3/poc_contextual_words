
"""
This file contains unit tests for the search.py file. There should be more tests.
This is just a demonstrative stub
"""

import spacy
import pytest
from kara.search import get_words

SPACY_MODEL_NAME = 'en_core_web_trf'

def test_get_words_with_normal_text():
    spacy_model = spacy.load(SPACY_MODEL_NAME)
    text = "Hello, world! This is a test of deliciousness."
    expected_output = ['hello', 'world', 'this', 'is', 'a', 'test', 'of', 'deliciousness']
    assert get_words(text, spacy_model) == expected_output

def test_get_words_with_empty_string():
    spacy_model = spacy.load(SPACY_MODEL_NAME)
    text = ""
    expected_output = []
    assert get_words(text, spacy_model) == expected_output

def test_get_words_with_no_words():
    spacy_model = spacy.load(SPACY_MODEL_NAME)
    text = "!!! ,,, . . ."
    expected_output = []
    assert get_words(text, spacy_model) == expected_output

def test_get_words_with_mixed_case():
    spacy_model = spacy.load(SPACY_MODEL_NAME)
    text = "Hello, World! This Is A Test."
    expected_output = ['hello', 'world', 'this', 'is', 'a', 'test']
    assert get_words(text, spacy_model) == expected_output
