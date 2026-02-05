# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 259R preferred styles
pio.templates["dsc259R"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc259R"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    # Create a copy to avoid SettingWithCopy warnings
    df = loans.copy()
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['term'] = df['term'].str.replace(' months', '').astype(int)
    df['emp_title'] = df['emp_title'].str.lower().str.strip()
    df['emp_title'] = df['emp_title'].replace({'rn': 'registered nurse'})
    df['term_end'] = df.apply(
        lambda row: row['issue_d'] + pd.DateOffset(months=row['term']), 
        axis=1
    )
    return df


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def correlations(df, pairs):
    data = {}
    
    for col1, col2 in pairs:
        r = df[col1].corr(df[col2])
        key = f"r_{col1}_{col2}"
        data[key] = r
    return pd.Series(data)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    df = loans.copy()
    
    # Bin the credit scores
    bins = [580, 670, 740, 800, 850]
    
    # right=False gives intervals like [580, 670)
    df['score_category'] = pd.cut(
        df['fico_range_low'], 
        bins=bins, 
        right=False
    )
    
    # Convert to string
    df['score_category'] = df['score_category'].astype(str)
    
    # Define the specific order for the x-axis so they don't get jumbled
    category_order = ['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)']
    
    # Create the boxplot
    fig = px.box(
        df,
        x='score_category',
        y='int_rate',
        color='term',
        color_discrete_map={
            36: 'purple',
            60: 'gold'
        },
        category_orders={
            'score_category': category_order
        },
        title='Interest Rate vs. Credit Score',
        labels={
            'score_category': 'Credit Score Range',
            'int_rate': 'Interest Rate (%)',
            'term': 'Loan Length (Months)'
        }
    )
    
    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    # 1. Identify the groups
    # 'desc' is the standard column for personal statements in LendingClub data
    # "With Statement" -> desc is not null
    # "Without Statement" -> desc is null
    
    # Create a boolean mask for having a statement
    has_statement = loans['desc'].notna()
    
    # Extract the interest rates
    # We can convert to numpy array for faster permutation
    rates = loans['int_rate'].values
    
    # 2. Calculate Observed Statistic
    # (Mean with statement) - (Mean without statement)
    mean_with = rates[has_statement].mean()
    mean_without = rates[~has_statement].mean()
    observed_diff = mean_with - mean_without
    
    # 3. Permutation Test
    simulated_diffs = []
    
    # Cache the count to speed up the loop
    n_with = has_statement.sum()
    n_total = len(rates)
    
    for _ in range(N):
        # Shuffle the interest rates
        shuffled_rates = np.random.permutation(rates)
        
        # Split into two groups of the same size as the original groups
        # The first n_with elements act as the "with statement" group
        shuffled_with = shuffled_rates[:n_with]
        shuffled_without = shuffled_rates[n_with:]
        
        # Calculate statistic
        sim_diff = shuffled_with.mean() - shuffled_without.mean()
        simulated_diffs.append(sim_diff)
        
    # 4. Calculate P-Value
    # Alternative Hypothesis: With > Without (One-sided)
    # Proportion of simulated differences >= observed difference
    simulated_diffs = np.array(simulated_diffs)
    p_value = (simulated_diffs >= observed_diff).mean()
    
    return p_value

def missingness_mechanism():
    # Interpret the result:
    # The prompt implies that there IS a difference (higher rates for those with statements).
    # If the distributions of 'int_rate' differ between missing and non-missing groups,
    # the missingness depends on the observed variable 'int_rate'.
    # Dependency on observed variables = MAR.
    return 2

def argument_for_nmar():
    string = (
    '''
    Borrowers who intend to use the loan for purposes they are embarrassed to 
    disclose may intentionally omit a personal statement, making the missingness 
    dependent on the unobserved content of the statement.
    '''
    )

    return string


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    total_tax = 0.0
    
    # Iterate through each bracket
    for i in range(len(brackets)):
        rate, lower_limit = brackets[i]
        
        # Determine the upper limit of the current bracket.
        # If there is a next bracket, the upper limit is the start of that next bracket.
        # If it's the last bracket, the upper limit is effectively infinity.
        if i < len(brackets) - 1:
            upper_limit = brackets[i+1][1]
        else:
            upper_limit = float('inf')
            
        # If the income hasn't reached this bracket's lower limit, break.
        if income <= lower_limit:
            break
            
        taxable_chunk = min(income, upper_limit) - lower_limit
        
        total_tax += taxable_chunk * rate
        
    return total_tax


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw):
    # Create a copy to avoid modifying the original DataFrame
    df = state_taxes_raw.copy()
    
    # 1. Drop rows full of null values (separator rows)
    df = df.dropna(how='all')
    
    # --- Helper Functions for the Pipeline ---
    
    def clean_state(df):
        # Replace garbage strings (like footnotes '(a, b, c)') with NaN
        df['State'] = df['State'].replace(to_replace=r'^\(.*', value=np.nan, regex=True)
        
        # Forward fill the NaN values with the most recent valid state name
        df['State'] = df['State'].ffill()
        return df

    def clean_rate(df):
        # Convert to string to handle mixed types
        s = df['Rate'].astype(str).str.lower()
        
        # Handle 'none' by treating it as 0
        s = s.replace('none', '0')
        
        # Remove the '%' sign
        s = s.str.replace('%', '', regex=False)
        
        # Convert to numeric float
        rates = pd.to_numeric(s, errors='coerce')
        
        # Convert percentage to proportion
        df['Rate'] = (rates / 100).round(2)
        return df

    def clean_limit(df):
        # Fill NaN values with 0
        s = df['Lower Limit'].fillna(0).astype(str)
        
        # Remove currency symbols and commas
        s = s.str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        
        # Convert to integer
        df['Lower Limit'] = pd.to_numeric(s).astype(int)
        return df

    # Execution Pipeline
    return (df
            .pipe(clean_state)
            .pipe(clean_rate)
            .pipe(clean_limit)
           )

# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    # Sort by State and Lower Limit to ensure correct ordered tuples
    sorted_taxes = state_taxes.sort_values(['State', 'Lower Limit'])
    
    # Create the (Rate, Lower Limit) tuple for each row
    sorted_taxes['bracket_tuple'] = sorted_taxes.apply(
        lambda row: (row['Rate'], row['Lower Limit']), axis=1
    )
    
    # Group by State and aggregate the tuples into a list
    bracket_series = sorted_taxes.groupby('State')['bracket_tuple'].apply(list)
    
    # Convert Series to DataFrame with the specific column name 'bracket_list'
    return bracket_series.to_frame('bracket_list')
    
def combine_loans_and_state_taxes(loans, state_taxes):
    # Start by loading in the JSON file.
    # state_mapping is a dictionary; use it!
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
        
    # Get the DataFrame of bracket lists
    brackets_df = state_brackets(state_taxes)
    
    # Update the index of brackets_df to match loans (e.g. "Ala." -> "AL")
    brackets_df.index = brackets_df.index.map(state_mapping)
    
    # Prepare the loans DataFrame
    # Rename 'addr_state' to 'State' to match
    loans_clean = loans.rename(columns={'addr_state': 'State'})
    
    # Merge loans with brackets
    merged = loans_clean.merge(
        brackets_df, 
        left_on='State', 
        right_index=True, 
        how='left'
    )
    
    return merged


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    loans = loans_with_state_taxes.copy()
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    # Calculate Federal Tax Owed
    loans['federal_tax_owed'] = loans['annual_inc'].apply(
        lambda income: tax_owed(income, FEDERAL_BRACKETS)
    )
    
    # Calculate State Tax Owed
    def calculate_state_tax(row):
        # Extract the bracket list for this specific borrower/state
        brackets = row['bracket_list']
        
        # Safety check: In case the merge resulted in a NaN,
        # treat it as 0 tax. 'brackets' would be a float (nan) if missing.
        if not isinstance(brackets, list):
            return 0.0
            
        return tax_owed(row['annual_inc'], brackets)
    
    loans['state_tax_owed'] = loans.apply(calculate_state_tax, axis=1)
    
    # Calculate Disposable Income
    # Gross Income - Federal Tax - State Tax
    loans['disposable_income'] = (
        loans['annual_inc'] - loans['federal_tax_owed'] - loans['state_tax_owed']
    )
    
    return loans
# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    # Dictionary to store the results for each keyword
    results = {}
    
    for k in keywords:
        # Filter the DataFrame for rows where emp_title contains the keyword
        mask = loans['emp_title'].str.contains(k, na=False)
        subset = loans[mask]
        
        # Calculate the mean of the quantitative column grouped by the categorical column
        # This returns a Series indexed by the categories
        grouped_means = subset.groupby(categorical_column)[quantitative_column].mean()
        
        # Calculate the Overall mean for this keyword group
        overall_mean = subset[quantitative_column].mean()
        
        # Create the column name dynamically
        col_name = f'{k}_mean_{quantitative_column}'
        
        # Convert to dict to easily add the 'Overall' row
        data_dict = grouped_means.to_dict()
        data_dict['Overall'] = overall_mean
        
        # Store in our results dictionary
        results[col_name] = data_dict
        
    # Create the final DataFrame
    result_df = pd.DataFrame(results)
    
    # Reorder the index to ensure 'Overall' is the last row
    categories = sorted([idx for idx in result_df.index if idx != 'Overall'])
    new_index_order = categories + ['Overall']
    
    # Reindex the DataFrame
    result_df = result_df.reindex(new_index_order)
    
    # Ensure columns are in the same order as keywords input
    expected_cols = [f'{k}_mean_{quantitative_column}' for k in keywords]
    result_df = result_df[expected_cols]
    
    return result_df


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    # Aggregate the data using the function you wrote earlier
    df = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)
    
    # Separate subgroups (all rows except last) from overall (last row)
    subgroups = df.iloc[:-1]
    overall = df.iloc[-1]
    
    # Extract the two columns for comparison
    group_a = subgroups.iloc[:, 0].values
    group_b = subgroups.iloc[:, 1].values
    
    overall_a = overall.iloc[0]
    overall_b = overall.iloc[1]
    
    # Check for Simpson's Paradox
    # Condition 1: Group A > Group B in ALL subgroups, but Group A < Group B Overall
    trend_A_higher = (group_a > group_b).all()
    overall_A_lower = overall_a < overall_b
    paradox_1 = trend_A_higher and overall_A_lower
    
    # Condition 2: Group A < Group B in ALL subgroups, but Group A > Group B Overall
    trend_A_lower = (group_a < group_b).all()
    overall_A_higher = overall_a > overall_b
    paradox_2 = trend_A_lower and overall_A_higher
    
    return bool(paradox_1 or paradox_2)

def paradox_example(loans):
    import itertools
    # Dynamically search for a valid paradox example
    
    # Limit search space to the top 20 most common job titles
    common_jobs = loans['emp_title'].value_counts().head(20).index.tolist()
    
    # Define likely columns to check (based on the columns you provided)
    quant_cols = ['loan_amnt', 'annual_inc', 'int_rate', 'dti']
    cat_cols = ['term', 'home_ownership', 'verification_status', 'grade']
    
    # Define the restricted combination we CANNOT return
    restricted_keywords = {'engineer', 'nurse'}
    restricted_quant = 'loan_amnt'
    restricted_cat = 'home_ownership'
    
    # Iterate through all unique pairs of job titles
    for k1, k2 in itertools.combinations(common_jobs, 2):
        keywords = [k1, k2]
        
        # Iterate through quantitative and categorical columns
        for quant in quant_cols:
            for cat in cat_cols:
                
                # Skip the restricted example
                if (set(keywords) == restricted_keywords and 
                    quant == restricted_quant and 
                    cat == restricted_cat):
                    continue
                
                # Check if this combination creates a paradox
                if exists_paradox(loans, keywords, quant, cat):
                    return {
                        'loans': loans,
                        'keywords': keywords,
                        'quantitative_column': quant,
                        'categorical_column': cat
                    }
    
    return None
