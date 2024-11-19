# poc_contextual_words
A Proof-of-Concept system to find which variant of homonyms are used in a piece of text.

As a proof-of-concept, this code lacks a lot that would be necessary for a production system, like tests, logging, accuracy monitoring, bias/ethical monitoring, an interface (like an API) to receive requests, etc.

This system takes a piece of text, and finds which variant of each word is used in that text. For example, given the text:

"Amazon profits grew" 

It would identify that "Amazon" refers to the business, while in the text:

"Rainfall in the Amazon grew"

It would identify that "Amazon" refers to the rainforest.

Currently, as a proof-of-concept, the text is a few examples hard-coded in search.py. Obviously, in a production system, this would not be the case.
