# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    directory = Path(dirname)
    if not directory.exists():
        raise FileNotFoundError(f"The directory {dirname} does not exist.")
    all_dfs = []

    # Iterate dynamically
    for file_path in directory.glob('survey*.csv'):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.replace('_', ' ').str.lower()
        all_dfs.append(df)
    
    # Concatenate all dataframes
    df = pd.concat(all_dfs)
    return df.reset_index(drop=True)

def com_stats(df):
    stat_list = []
    is_ohio = df['university'].str.contains('Ohio', na=False)
    ohio_grads = df[is_ohio] 
    num = ohio_grads['job title'].str.contains('Programmer').sum()
    den = len(ohio_grads)

    programmer_proportion = num / den
    stat_list.append(programmer_proportion)

    unique_titles = pd.Series(df['job title'].unique())
    ends_in_eng = unique_titles.str.endswith('Engineer').sum()      
    stat_list.append(ends_in_eng)
    
    longest_idx = df['job title'].str.len().idxmax()
    max_title = df.loc[longest_idx, 'job title']
    stat_list.append(max_title)
    
    manager_count = df['job title'].str.contains('manager', case=False).sum()
    stat_list.append(manager_count)

    return stat_list


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    directory = Path(dirname)
    if not directory.exists():
        raise FileNotFoundError(f"The directory {dirname} does not exist.")
    
    dfs = []

    for file_path in directory.glob('favorite*.csv'):
        current_df = pd.read_csv(file_path)
        current_df.columns = current_df.columns.str.replace('_', ' ').str.lower()
        
        # Set the ID as the index
        if 'id' in current_df.columns:
            current_df = current_df.set_index('id')
        
        dfs.append(current_df)
    
    # Combine horizontally
    final_df = pd.concat(dfs, axis=1, sort=True)
    
    # Rename index to 'Student ID'
    final_df.index.name = 'Student ID'
    return final_df.sort_index()

def check_credit(df):
    # Create a copy to avoid SettingWithCopy warnings
    # Replace the specific "no genres" string with NaN
    scored_df = df.replace('(no genres listed)', np.nan)
    
    question_cols = [col for col in df.columns if col != 'name']
    
    col_participation = scored_df[question_cols].count() / len(df)
    
    num_high_participation = (col_participation >= 0.90).sum()
    
    class_points = min(num_high_participation, 2)
    
    student_participation = scored_df[question_cols].count(axis=1) / len(question_cols)
    
    individual_points = (student_participation >= 0.50).astype(int) * 5
    
    out_df = pd.DataFrame(index=df.index)
    out_df['name'] = df['name']
    out_df['ec'] = individual_points + class_points
    
    return out_df


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------

def most_popular_procedure(pets, procedure_history):
    merged = pd.merge(pets, procedure_history, on='PetID', how='inner')
    return merged['ProcedureType'].value_counts().idxmax()

def pet_name_by_owner(owners, pets):
    merged = pd.merge(owners, pets, on='OwnerID', suffixes=('_owner', '_pet'))
    grouped = merged.groupby(['OwnerID', 'Name_owner'])['Name_pet'].apply(list)
    
    def format_names(pet_list):
        if len(pet_list) == 1:
            return pet_list[0]
        return pet_list
    
    formatted_series = grouped.apply(format_names)
    
    return formatted_series.reset_index(level=0, drop=True)

def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    df = pd.merge(owners, pets, on='OwnerID', how='left')
    df = pd.merge(df, procedure_history, on='PetID', how='left')
    df = pd.merge(df, procedure_detail, on=['ProcedureType', 'ProcedureSubCode'], how='left')
    df['Price'] = df['Price'].fillna(0)
    return df.groupby('City')['Price'].sum()


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales):
    pivoted = sales.pivot_table(index='Name', values='Total', aggfunc='mean')
    return pivoted.rename(columns={'Total': 'Average Sales'})

def product_name(sales):
    return sales.pivot_table(
        index='Name', 
        columns='Product', 
        values='Total', 
        aggfunc='sum'
    )

def count_product(sales):
    return sales.pivot_table(
        index=['Product', 'Name'],  
        columns='Date',             
        values='Total', 
        aggfunc='count',            
        fill_value=0
    )

def total_by_month(sales):
    sales_copy = sales.copy()
    sales_copy['Month'] = pd.to_datetime(sales_copy['Date']).dt.month_name()
    
    return sales_copy.pivot_table(
        index=['Name', 'Product'], 
        columns='Month',            
        values='Total', 
        aggfunc='sum', 
        fill_value=0
    )
