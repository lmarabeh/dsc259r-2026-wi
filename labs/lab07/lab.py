# lab.py


import pandas as pd
import numpy as np
import os
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


import re

def match_1(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    pattern = r"^.{2}\[.{2}\].*$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    pattern = r"^\(858\) \d{3}-\d{4}$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    pattern = r"^[a-zA-Z0-9\s?]{5,9}\?$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$$$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$!@$")
    False
    """
    pattern = r"^\$[^abc$]*\$[aA]+[bB]+[cC]+$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_5("dsc259.py")
    True
    >>> match_5("dsc259py")
    False
    >>> match_5("dsc259..py")
    False
    >>> match_5("dsc259+.py")
    False
    """
    pattern = r"^[a-zA-Z0-9_]+\.py$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """
    pattern = r"^[a-z]+_[a-z]+$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = r"^_.*_$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_8(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo!!!!!!! !!!!!!!!!")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = r"^[^Oi1]+$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_9(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('NY-32-LAX-0000') # If the state is NY, the city can be any 3 letter code, including LAX or SAN!
    True
    >>> match_9('TX-32-SAN-4491')
    False
    '''
    pattern = r"^(CA-\d{2}-(SAN|LAX)-\d{4}|NY-\d{2}-[A-Z]{3}-\d{4})$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_10('ABCdef')
    ['bcd']
    >>> match_10(' DEFaabc !g ')
    ['def', 'bcg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10('Ab..DEF')
    ['bde']
    
    '''
    # Convert string to lowercase
    s = string.lower()
    
    # Remove all non-alphanumeric characters (not in \w) and the letter 'a'
    s = re.sub(r'[^\w]|a', '', s)
    
    # Return a list of non-overlapping three-character substrings
    return re.findall(r'...', s)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_personal(s):
    # Emails: alphanumeric usernames and domains
    emails = re.findall(r'[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z0-9]+', s)
    
    # SSN: standard 3-2-4 format
    ssn = re.findall(r'\d{3}-\d{2}-\d{4}', s)
    
    # Bitcoin: strict alphanumeric boundaries for long strings
    # use lookarounds to ensure it's not part of an email or file path
    bitcoin = re.findall(r'(?<![a-zA-Z0-9@\.])[a-zA-Z0-9]{26,35}(?![a-zA-Z0-9@\.])', s)
    
    # Addresses: non-greedy matching confined to a single line
    # re.IGNORECASE makes sure we catch "st", "St.", "STREET", etc.
    address_pattern = r'\d{1,5}[ \t]+[a-zA-Z0-9 \t]+?(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl|Square|Sq)\b\.?'
    addresses = re.findall(address_pattern, s, flags=re.IGNORECASE)
    
    # Return as a tuple of lists as requested
    return emails, ssn, bitcoin, addresses

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def tfidf_data(reviews_ser, review):
    # Split the review into a list of words
    words = review.split()
    
    # Get the unique words to act as our index
    unique_words = list(set(words))
    
    total_reviews = len(reviews_ser)
    total_words = len(words)
    
    results = []
    
    for word in unique_words:
        # Count (cnt)
        cnt = words.count(word)
        
        # Term Frequency (tf)
        tf = cnt / total_words
        
        # Inverse Document Frequency (idf)
        # use \b to ensure we match the exact word, not substrings
        matches = reviews_ser.str.contains(rf'\b{word}\b', regex=True)
        docs_with_word = matches.sum()
        
        idf = np.log(total_reviews / docs_with_word)
        
        # TF-IDF
        tfidf = tf * idf
        
        results.append({
            'word': word,
            'cnt': cnt,
            'tf': tf,
            'idf': idf,
            'tfidf': tfidf
        })
        
    return pd.DataFrame(results).set_index('word')

def relevant_word(tfidf_df):
    return tfidf_df['tfidf'].idxmax()


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def hashtag_list(text_series):
    return text_series.str.findall(r'#([^\s]+)')

def most_common_hashtag(hashtag_series):
    global_counts = hashtag_series.explode().value_counts()
    
    def get_best_hashtag(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            return np.nan
        
        return max(lst, key=lambda x: global_counts[x])
        
    return hashtag_series.apply(get_best_hashtag)

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def create_features(ira):
    df = ira.copy()
    
    # --- Feature Extraction ---
    
    # 1. num_hashtags
    hashtags_series = df['text'].str.findall(r'#([^\s]+)')
    df['num_hashtags'] = hashtags_series.apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # 2. mc_hashtags
    global_counts = hashtags_series.explode().value_counts()
    
    def get_most_common(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            return np.nan
        return max(lst, key=lambda tag: global_counts.get(tag, 0))
    
    df['mc_hashtags'] = hashtags_series.apply(get_most_common)
    
    # 3. num_tags
    tags_series = df['text'].str.findall(r'@[a-zA-Z0-9]+')
    df['num_tags'] = tags_series.apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # 4. num_links
    links_series = df['text'].str.findall(r'https?://[^\s]+')
    df['num_links'] = links_series.apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # 5. is_retweet
    df['is_retweet'] = df['text'].str.match(r'^RT')
    
    # --- Text Cleaning ---
    
    cleaned_text = df['text'].copy()
    
    # Replace meta-information with a space
    meta_pattern = r'(^RT|@[a-zA-Z0-9]+|https?://[^\s]+|#[^\s]+)'
    cleaned_text = cleaned_text.str.replace(meta_pattern, ' ', regex=True)
    
    # Replace everything other than letters, numbers, and spaces with a space
    cleaned_text = cleaned_text.str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    
    # Lowercase all letters
    cleaned_text = cleaned_text.str.lower()
    
    # Separate by exactly one space and strip leading/trailing whitespace
    cleaned_text = cleaned_text.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    df['text'] = cleaned_text
    
    # --- Final Output Formatting ---
    
    return df[['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']]