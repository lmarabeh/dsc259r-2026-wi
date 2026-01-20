# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    # Observation 3: tricky_1 and tricky_2 have different column names
    # because read_csv handles duplicates by renaming them.
    return 3

def trick_bool():
    # 1. bools[True] -> Selects columns named True (2 cols, 4 rows) -> Choice 4
    # 2. bools[[T, T, F, F]] -> Row mask (2 rows, 4 cols) -> Choice 10
    # 3. bools[[True, False]] -> Column selection (4 cols, 4 rows) -> Choice 12
    return [4, 10, 12]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df):
    # 'num_nonnull': .count() returns the number of non-NA/null observations
    num_nonnull = df.count()
    
    # 'prop_nonnull': Divide the count of non-nulls by the total number of rows
    # len(df) gives the total number of rows (including nulls)
    prop_nonnull = num_nonnull / len(df)
    
    # 'num_distinct': .nunique() counts distinct observations
    # By default, dropna=True, so it ignores nulls, exactly as requested
    num_distinct = df.nunique()
    
    # 'prop_distinct': The ratio of distinct non-nulls to total non-nulls
    prop_distinct = num_distinct / num_nonnull
    
    # Combine Series into a DataFrame
    # The keys of the dictionary become the column names
    # The indices of the Series (original df columns) become the index
    return pd.DataFrame({
        'num_nonnull': num_nonnull,
        'prop_nonnull': prop_nonnull,
        'num_distinct': num_distinct,
        'prop_distinct': prop_distinct
    })


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df, N=10):
    # Initialize an empty DataFrame with an index from 0 to N-1
    result = pd.DataFrame(index=range(N))
    
    for col in df.columns:
        # Get the value counts for the column
        counts = df[col].value_counts()
        
        # Slice the first N entries using .iloc (as required by the hint)
        top_N = counts.iloc[:N]
        
        # Create the 'values' column
        values_col = pd.Series(top_N.index).reset_index(drop=True).reindex(range(N))
        result[f'{col}_values'] = values_col
        
        # Create the 'counts' column
        counts_col = pd.Series(top_N.values).reset_index(drop=True).reindex(range(N))
        result[f'{col}_counts'] = counts_col
        
    return result


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    # Distinguish between the 'Name' column and the 'Power' columns
    power_cols = powers.select_dtypes(include='bool').columns
    
    # Hero with the most superpowers
    power_counts = powers[power_cols].sum(axis=1)
    max_idx = power_counts.idxmax()
    
    # Retrieve the name
    if 'hero_names' in powers.columns:
        most_powers_hero = powers.loc[max_idx, 'hero_names']
    else:
        # If 'hero_names' isn't a column, we assume the index holds the names
        most_powers_hero = powers.index[max_idx]
        
        
    # Most common power among flyers (excluding Flight)
    
    # Filter the DataFrame to only include rows where 'Flight' is True
    flyers = powers[powers['Flight'] == True]
    
    # Sum down the columns (axis=0) to count how common each power is among flyers
    flyer_power_counts = flyers[power_cols].sum(axis=0)
    
    # Drop 'Flight' itself, as it would obviously be 100%
    most_common_flying = flyer_power_counts.drop('Flight').idxmax()
    
    
    # Most common power among those with only 1 power
    
    # Use 'power_counts' to find rows where the sum is exactly 1
    one_trick_ponies = powers[power_counts == 1]
    
    # Sum down the columns for this subset
    single_power_counts = one_trick_ponies[power_cols].sum(axis=0)
    
    # Find the power with the highest count
    most_common_single = single_power_counts.idxmax()
    
    return [most_powers_hero, most_common_flying, most_common_single]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace(['-', -99.0], np.nan)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return [
        '504',        # Q0
        'George Lucas',     # Q1
        'bad',              # Q2
        'Marvel Comics',    # Q3
        'NBC - Heroes',     # Q4
        '302'             # Q5
    ]

# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(df):
    cleaned = df.copy()
    
    # Clean 'institution': Replace newlines with ', '
    cleaned['institution'] = cleaned['institution'].str.replace('\n', ', ', regex=False)
    
    # Convert 'broad_impact' to int
    cleaned['broad_impact'] = cleaned['broad_impact'].astype(int)
    
    # Handle 'national_rank' splitting
    # Split string by ", " into two columns
    splits = cleaned['national_rank'].str.split(', ', expand=True)
    cleaned['nation'] = splits[0]
    cleaned['national_rank_cleaned'] = splits[1].astype(int)
    
    # Fix the 3 inconsistent country names (Longer name wins)
    country_replacements = {
        'Czechia': 'Czech Republic',
        'USA': 'United States',
        'UK': 'United Kingdom'
    }
    cleaned['nation'] = cleaned['nation'].replace(country_replacements)
    
    # Drop the old column
    cleaned = cleaned.drop(columns=['national_rank'])
    
    # Create 'is_r1_public' column
    # Condition A: Must be 'Public'
    is_public = cleaned['control'] == 'Public'
    
    # Condition B: Must have non-null values in 'control', 'city', AND 'state'
    # .notna().all(axis=1) checks if all columns in that row are non-null
    has_r1_data = cleaned[['control', 'city', 'state']].notna().all(axis=1)
    
    # Combine conditions
    cleaned['is_r1_public'] = is_public & has_r1_data
    
    return cleaned

def university_info(cleaned):
    # State with lowest mean score (min 3 institutions)
    state_counts = cleaned['state'].value_counts()
    valid_states = state_counts[state_counts >= 3].index
    
    # .idxmin() returns the index label (string), so no casting needed here
    lowest_score_state = cleaned[cleaned['state'].isin(valid_states)] \
        .groupby('state')['score'].mean().idxmin()

    # Proportion of top 100 world rank that are top 100 faculty
    top_100_world = cleaned[cleaned['world_rank'] <= 100]
    
    # Wrap the result in float() to convert numpy.float64 -> python float
    prop_top_faculty = float((top_100_world['quality_of_faculty'] <= 100).mean())

    # Number of states with >= 50% "private" institutions
    def is_mostly_private(x):
        return (~x).mean() >= 0.5
        
    # Wrap the result in int() to convert numpy.int64 -> python int
    num_private_states = int(cleaned.groupby('state')['is_r1_public'] \
        .apply(is_mostly_private).sum())

    # Worst world_rank among national #1s
    national_champions = cleaned[cleaned['national_rank_cleaned'] == 1]
    worst_champ_idx = national_champions['world_rank'].idxmax()
    worst_champ_inst = national_champions.loc[worst_champ_idx, 'institution']

    return [lowest_score_state, prop_top_faculty, num_private_states, worst_champ_inst]

