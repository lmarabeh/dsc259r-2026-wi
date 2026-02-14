# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    # Respect robots.txt (Pause for 0.5 seconds before request)
    time.sleep(0.5)
    
    # Download the content
    response = requests.get(url)
    
    # Ensure correct encoding (Gutenberg usually uses UTF-8)
    response.encoding = 'utf-8'
    text = response.text
    
    # Replace Windows newline characters with standard newlines
    text = text.replace('\r\n', '\n')
    
    # Define Regex patterns for Gutenberg START and END markers
    start_pattern = r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_pattern =   r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    
    # Search for the markers
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    end_match = re.search(end_pattern, text, re.IGNORECASE)
    
    # Extract content between markers
    if start_match and end_match:
        start_index = start_match.end()
        end_index = end_match.start()
        
        return text[start_index:end_index]
    else:
        print("Warning: Start or End marker not found.")
        return text


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    # Define markers
    START_TOKEN = '\x02'
    END_TOKEN = '\x03'
    
    # Normalize and Clean
    text = book_string.replace('\r\n', '\n').strip()
    
    # Define Regex Pattern
    pattern = re.compile(r'\n{2,}|\w+|[^\w\s]')
    
    # Find all matches
    raw_matches = pattern.findall(text)
    
    # Construct the final list
    tokens = [START_TOKEN]
    
    for match in raw_matches:
        if match.startswith('\n'):
            # If hit a paragraph break, close the previous para and start a new one
            tokens.append(END_TOKEN)
            tokens.append(START_TOKEN)
        else:
            # Otherwise, it's a word or punctuation token
            tokens.append(match)
            
    # End the document
    tokens.append(END_TOKEN)
    
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    def __init__(self, tokens):
        """
        Initializes a UniformLM object.

        Args:
            tokens (list): A list of tokens (strings) containing the training data.
        """
        self.mdl = self.train(tokens)

    def train(self, tokens):
        # Get unique tokens
        unique_tokens = list(set(tokens))
        
        # Calculate uniform probability
        prob = 1.0 / len(unique_tokens)
        
        # Create the Series
        return pd.Series(data=prob, index=unique_tokens)

    def probability(self, words):
        """
        Computes the probability of a sequence of words.

        Args:
            words (list or tuple): A sequence of tokens.

        Returns:
            float: The probability of the sequence occurring.
        """
        # Check if all words exist in our vocabulary
        for w in words:
            if w not in self.mdl.index:
                return 0.0
        
        # Compute probability
        single_word_prob = self.mdl.iloc[0]
        
        return single_word_prob ** len(words)

    def sample(self, M):
        # Randomly select M tokens from the vocabulary (index of our Series).
        random_tokens = np.random.choice(self.mdl.index, size=M, replace=True)
        
        # Join them with spaces to form the "sentence"
        return " ".join(random_tokens)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

class UnigramLM(object):

    def __init__(self, tokens):
        self.mdl = self.train(tokens)

    def train(self, tokens):
        # Count occurrences of each token
        counts = pd.Series(tokens).value_counts()
        
        # Normalize to get probabilities
        # Divide counts by the total number of tokens in the corpus
        return counts / len(tokens)

    def probability(self, words):
        prob = 1.0
        
        for w in words:
            if w not in self.mdl.index:
                return 0.0
            
            prob *= self.mdl[w]
            
        return prob

    def sample(self, M):
        # np.random.choice allows us to specify probabilities 'p'
        # We sample from the index (the words) using the values (the probs) as weights
        random_tokens = np.random.choice(
            self.mdl.index, 
            size=M, 
            p=self.mdl.values,
            replace=True
        )
        
        return " ".join(random_tokens)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        ...
        
    def train(self, ngrams):
        ...
    
    def probability(self, words):
        ...
    

    def sample(self, M):
        ...
