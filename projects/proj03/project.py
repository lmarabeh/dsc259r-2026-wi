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

    # ----------------------------------------------------------------------
    # 5.1: Create N-Grams
    # ----------------------------------------------------------------------
    def create_ngrams(self, tokens):
        """
        Creates a list of N-Grams from the given tokens.
        """
        ngrams = []
        for i in range(len(tokens) - self.N + 1):
            window = tokens[i : i + self.N]
            ngrams.append(tuple(window))
        
        return ngrams

    # ----------------------------------------------------------------------
    # 5.2: Train 
    # ----------------------------------------------------------------------
    def train(self, ngrams):
        """
        Trains the N-Gram language model on the given N-Grams.
        """
        # 1. Initialize DataFrame with the passed N-grams
        df = pd.DataFrame({'ngram': ngrams})
        
        # 2. Create (N-1)-gram context column
        df['n1gram'] = df['ngram'].apply(lambda x: x[:-1])
        
        # 3. Compute Counts
        ngram_counts = df['ngram'].value_counts()
        n1gram_counts = df['n1gram'].value_counts()
        
        # 4. Build Model
        mdl = df.drop_duplicates(subset=['ngram']).copy()
        
        # 5. Calculate Probabilities
        mdl['prob'] = (mdl['ngram'].map(ngram_counts) / 
                       mdl['n1gram'].map(n1gram_counts))
        
        return mdl[['ngram', 'n1gram', 'prob']].reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 5.3: Probability
    # ----------------------------------------------------------------------
    def probability(self, words):
        """
        Computes the probability of a sequence of words.
        """
        # Case 1: Sequence is shorter than N (Recursive Backoff)
        if len(words) < self.N:
            return self.prev_mdl.probability(words)

        # Case 2: Sequence is long enough
        # A. Handle the "Warm-up" (The first N-1 tokens)
        prefix = words[:self.N - 1]
        current_prob = self.prev_mdl.probability(prefix)
        start_index = self.N - 1

        # B. Multiply by conditional probabilities of the full N-grams
        for i in range(start_index, len(words)):
            current_ngram = tuple(words[i - (self.N - 1) : i + 1])
            
            row = self.mdl[self.mdl['ngram'] == current_ngram]
            
            if row.empty:
                return 0.0
            
            current_prob *= row['prob'].values[0]
            
        return current_prob

    # ----------------------------------------------------------------------
    # 5.4: Sample
    # ----------------------------------------------------------------------
    def sample(self, M):
        """
        Generates a random sentence of length M.
        """
        sentence = ['\x02']
        
        for i in range(M):
            if i == M - 1:
                sentence.append('\x03')
                continue
            
            next_token = self._get_next_token(sentence)
            sentence.append(next_token)
            
        return " ".join(sentence)

    def _get_next_token(self, current_sentence):
        """Helper for sampling"""
        req_context_len = self.N - 1
        
        # CASE 1: Not enough context
        if len(current_sentence) < req_context_len:
            if self.N == 2:
                return np.random.choice(
                    self.prev_mdl.mdl.index, 
                    p=self.prev_mdl.mdl.values
                )
            else:
                # Recursive call for N > 2
                return self.prev_mdl._get_next_token(current_sentence)
        
        # CASE 2: Enough context
        if req_context_len == 0:
            context = ()
        else:
            context = tuple(current_sentence[-req_context_len:])
            
        candidates = self.mdl[self.mdl['n1gram'] == context]
        
        # Handle Dead Ends
        if candidates.empty:
            return '\x03'
        
        possible_words = candidates['ngram'].apply(lambda x: x[-1])
        probabilities = candidates['prob']
        
        return np.random.choice(possible_words, p=probabilities)