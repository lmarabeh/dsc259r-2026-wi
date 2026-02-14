# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    return ['NMAR', 'MD', 'MAR', 'NMAR', 'MAR']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    return ['MAR', 'MAR', 'MD', 'NMAR', 'MCAR']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def first_round():
    df = pd.read_csv('data/payment.csv')
    df['birth_year'] = pd.to_datetime(df['date_of_birth']).dt.year
    df['age'] = 2024 - df['birth_year']
    
    valid_ages = df.dropna(subset=['age'])
    
    missing_cc = valid_ages[valid_ages['credit_card_number'].isna()]['age']
    present_cc = valid_ages[~valid_ages['credit_card_number'].isna()]['age']
    
    obs_diff = abs(missing_cc.mean() - present_cc.mean())
    
    n_repetitions = 500
    differences = []
    combined = valid_ages['age'].values
    n_missing = len(missing_cc)
    
    for _ in range(n_repetitions):
        shuffled = np.random.permutation(combined)
        shuffled_missing = shuffled[:n_missing]
        shuffled_present = shuffled[n_missing:]
        
        diff = abs(shuffled_missing.mean() - shuffled_present.mean())
        differences.append(diff)
        
    p_val_numpy = (np.array(differences) >= obs_diff).mean()
    
    p_val = p_val_numpy.item() 
    
    decision = 'R' if p_val < 0.05 else 'NR'
    
    return [p_val, decision]

def second_round():
    df = pd.read_csv('data/payment.csv')
    df['birth_year'] = pd.to_datetime(df['date_of_birth']).dt.year
    df['age'] = 2024 - df['birth_year']
    
    valid_ages = df.dropna(subset=['age'])
    
    missing_cc = valid_ages[valid_ages['credit_card_number'].isna()]['age']
    present_cc = valid_ages[~valid_ages['credit_card_number'].isna()]['age']
    
    ks_result = stats.ks_2samp(missing_cc, present_cc)
    
    p_val = float(ks_result.pvalue)
    
    decision = 'R' if p_val < 0.05 else 'NR'
    
    final_conclusion = 'D' if decision == 'R' else 'ND'
    
    return [p_val, decision, final_conclusion]


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):
    p_values = {}
    
    cols_to_test = [c for c in heights.columns if c.startswith('child_')]
    
    for col in cols_to_test:
        father_missing = heights.loc[heights[col].isna(), 'father'].dropna()
        father_present = heights.loc[heights[col].notna(), 'father'].dropna()
        ks_result = stats.ks_2samp(father_missing, father_present)
        p_values[col] = ks_result.pvalue
        
    return pd.Series(p_values)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    father_quartiles = pd.qcut(new_heights['father'], 4)
    conditional_means = new_heights.groupby(father_quartiles)['child'].transform('mean')
    imputed_child = new_heights['child'].fillna(conditional_means)
    return imputed_child


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    observed = child.dropna()
    counts, bin_edges = np.histogram(observed, bins=10)
    probs = counts / counts.sum()
    chosen_bin_indices = np.random.choice(len(counts), size=N, p=probs)
    lows = bin_edges[chosen_bin_indices]
    highs = bin_edges[chosen_bin_indices + 1]
    return np.random.uniform(lows, highs)

def impute_height_quant(child):
    is_missing = child.isna()
    n_missing = is_missing.sum()
    fill_values = quantitative_distribution(child, n_missing)
    child_imputed = child.copy()
    child_imputed.loc[is_missing] = fill_values
    return child_imputed


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    mc_answers = [1, 2, 2, 1]
    urls = [
        'https://www.python.org/robots.txt', 
        'https://www.facebook.com/robots.txt'
    ]
    return mc_answers, urls
