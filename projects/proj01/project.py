# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    assignment_names = {
        'lab': [],
        'project': [],
        'midterm': [],
        'final': [],
        'disc': [],
        'checkpoint': []
    }
    for col in grades.columns:
        if ' - ' in col:
            continue
        if col.startswith('lab') and len(col) == 5 and col[3:].isdigit():
            assignment_names['lab'].append(col)
        elif col.startswith('project') and len(col) == 9 and col[7:].isdigit():
            assignment_names['project'].append(col)
        elif col.startswith('discussion') and len(col) == 12 and col[10:].isdigit():
            assignment_names['disc'].append(col)
        elif 'checkpoint' in col:
            assignment_names['checkpoint'].append(col)
        elif col == 'Midterm':
            assignment_names['midterm'].append(col)
        elif col == 'Final':
            assignment_names['final'].append(col)
            
    for key in assignment_names:
        assignment_names[key].sort()
        
    return assignment_names


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    # Get project IDs from Q1
    project_ids = get_assignment_names(grades)['project']
    
    scores = []
    for pid in project_ids:
        # Include Autograded + Free Response
        # Exclude 'checkpoint' to avoid double-counting bad checkpoint scores
        relevant_cols = [
            c for c in grades.columns 
            if c.startswith(pid) 
            and 'Max Points' not in c 
            and 'Lateness' not in c
            and 'checkpoint' not in c
        ]
        
        # Sum Scores and Max Points
        total_score = 0
        total_max = 0
        
        for col in relevant_cols:
            max_col = col + ' - Max Points'
            if max_col in grades.columns:
                total_score += grades[col].fillna(0)
                total_max += grades[max_col].fillna(0)
            
        # Calculate Ratio
        if hasattr(total_max, 'sum') and total_max.sum() == 0:
             scores.append(pd.Series(0, index=grades.index))
        else:
             scores.append(total_score / total_max)
        
    # Average the individual project ratios
    return pd.concat(scores, axis=1).mean(axis=1)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(lateness_series):
    def get_multiplier(lateness_str):
        parts = lateness_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        
        total_hours = hours + (minutes / 60) + (seconds / 3600)
        
        # Apply the Logic Gates (Order matters!)
        
        # Grace Period: strictly <= 2 hours is treated as on time
        if total_hours <= 2:
            return 1.0
        
        # 1 Week: Up to 168 hours
        # Exclude the 2 hour grace period to this limit.
        elif total_hours <= 168:
            return 0.9
            
        # 2 Weeks
        elif total_hours <= 336:
            return 0.7
            
        # Beyond 2 weeks
        else:
            return 0.4

    return lateness_series.apply(get_multiplier)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    # Get the list of lab assignments using Q1
    lab_cols = get_assignment_names(grades)['lab']
    
    # Create empty DataFrame with the same index as the input
    processed = pd.DataFrame(index=grades.index)
    
    for lab in lab_cols:
        # Helper columns
        max_col = lab + ' - Max Points'
        late_col = lab + ' - Lateness (H:M:S)'
        
        # Calculate the Raw Normalized Score (Score / Max)
        raw_score = grades[lab] / grades[max_col]
        
        # Calculate the Multiplier
        # Handle NaNs in the lateness column.
        lateness_filled = grades[late_col].fillna('00:00:00')
        multipliers = lateness_penalty(lateness_filled)
        
        # Apply Penalty
        final_score = raw_score * multipliers
        
        # Handle Missing Assignments
        processed[lab] = final_score.fillna(0)
        
    return processed


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    
    # Sum all the lab scores for each student
    sum_scores = processed.sum(axis=1)
    
    # Find the minimum score for each student
    min_scores = processed.min(axis=1)
    
    # Calculate the new total
    dropped_sum = sum_scores - min_scores
    
    # Calculate the denominator
    num_labs = processed.shape[1]
    
    # Avoid division by zero if there's only 1 lab
    if num_labs <= 1:
        return sum_scores
        
    # Return the new average
    return dropped_sum / (num_labs - 1)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    names = get_assignment_names(grades)
    
    # --- Labs (20%) ---
    processed_labs = process_labs(grades)
    lab_score = lab_total(processed_labs) * 0.20
    
    # --- Projects (30%) ---
    project_score = projects_total(grades) * 0.30
    
    # --- Checkpoints & Discussions (2.5% each) ---
    def calc_weighted_cat(cat_key):
        # Filter out the 'junk' columns that Q1 may have captured
        raw_cols = [
            c for c in names[cat_key] 
            if 'Max Points' not in c 
            and 'Lateness' not in c
        ]
        
        ratios = []
        for c in raw_cols:
            max_col = c + ' - Max Points'
            if max_col in grades.columns:
                ratios.append(grades[c] / grades[max_col])
        
        return pd.concat(ratios, axis=1).fillna(0).mean(axis=1)

    checkpoint_score = calc_weighted_cat('checkpoint') * 0.025
    discussion_score = calc_weighted_cat('disc') * 0.025
    
    # --- Midterm (15%) ---
    midterm_col = names['midterm'][0]
    midterm_score = (grades[midterm_col] / grades[midterm_col + ' - Max Points']).fillna(0) * 0.15
    
    # --- Final (30%) ---
    final_col = names['final'][0]
    final_score = (grades[final_col] / grades[final_col + ' - Max Points']).fillna(0) * 0.30
    
    # --- Summation ---
    return lab_score + project_score + checkpoint_score + discussion_score + midterm_score + final_score


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total_points_series):
    def get_grade(score):
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    return total_points_series.apply(get_grade)

def letter_proportions(total_points_series):
    letters = final_grades(total_points_series)
    props = letters.value_counts(normalize=True)
    all_grades = ['A', 'B', 'C', 'D', 'F']
    props = props.reindex(all_grades, fill_value=0.0)
    
    # Sort by Frequency (Descending)
    return props.sort_values(ascending=False)




# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    # Select the columns corresponding to the redemption questions
    redemption_cols = final_breakdown.iloc[:, question_numbers]
    
    # Calculate the Total Possible Points
    total_possible = redemption_cols.max().sum()
    
    # Calculate Student Totals
    student_totals = redemption_cols.sum(axis=1)
    
    # Calculate Proportion
    raw_scores = (student_totals / total_possible).fillna(0)
    
    # Create the output DataFrame
    return pd.DataFrame({
        'PID': final_breakdown.iloc[:, 0],
        'Raw Redemption Score': raw_scores
    })

def combine_grades(grades, redemption_scores):
    combined = pd.merge(grades, redemption_scores, on='PID', how='left')
    
    return combined


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(series):
    # ddof=0 ensures we use population standard deviation (dividing by N)
    mean_val = series.mean()
    std_val = series.std(ddof=0)
    
    # Avoid division by zero if all values are the same (std_val = 0)
    if std_val == 0:
        return series - mean_val
        
    return (series - mean_val) / std_val

def add_post_redemption(grades_combined):
    df = grades_combined.copy()
    
    # Calculate 'Midterm Score Pre-Redemption'
    names = get_assignment_names(df)
    midterm_col = names['midterm'][0]
    midterm_max_col = midterm_col + ' - Max Points'
    
    # Calculate proportion (0.0 to 1.0)
    pre_redemption_series = df[midterm_col] / df[midterm_max_col]
    
    # Fix missing midterms: If they didn't take it, score is 0.
    pre_redemption_filled = pre_redemption_series.fillna(0)
    
    # Store the Pre-Redemption Score (Filled) in the dataframe
    df['Midterm Score Pre-Redemption'] = pre_redemption_filled

    # Calculate Z-Scores
    mid_z = z_score(pre_redemption_filled)
    
    # Z-score for Redemption (already calculated in previous step)
    # We assume 'Raw Redemption Score' exists from combine_grades
    red_z = z_score(df['Raw Redemption Score'])
    
    # Apply Redemption Policy
    mid_mean = pre_redemption_filled.mean()
    mid_std = pre_redemption_filled.std(ddof=0)
    
    # Vectorized comparison:
    # If Redemption Z > Midterm Z, calculate replacement.
    # Otherwise, keep original.
    
    # Calculate the potential new score for everyone
    potential_new_score = (red_z * mid_std) + mid_mean
    
    # Use np.where to choose the higher of the two strategies (Redemption vs Original)
    post_redemption = np.where(
        red_z > mid_z,
        potential_new_score,
        pre_redemption_filled
    )
    
    # Cap at 1.0 (100%)
    df['Midterm Score Post-Redemption'] = np.minimum(post_redemption, 1.0)
    
    return df

# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    # Run the redemption logic to get the Pre and Post columns
    df_redemption = add_post_redemption(grades_combined)
    
    # Get the original course totals
    original_total = total_points(grades_combined)
    
    # Perform the "Surgical Swap"
    # New Total = Old Total - (Old Midterm Portion) + (New Midterm Portion)
    # The midterm is worth 15% of the grade.
    
    old_midterm_points = df_redemption['Midterm Score Pre-Redemption'] * 0.15
    new_midterm_points = df_redemption['Midterm Score Post-Redemption'] * 0.15
    
    # Calculate the difference (Delta)
    delta = new_midterm_points - old_midterm_points
    
    # Apply the delta to the original total
    return original_total + delta

def proportion_improved(grades_combined):
    """
    Returns the proportion of students whose letter grade improved.
    """
    # Calculate Old and New Scores
    old_scores = total_points(grades_combined)
    new_scores = total_points_post_redemption(grades_combined)
    
    # Convert to Letter Grades
    old_letters = final_grades(old_scores)
    new_letters = final_grades(new_scores)
    
    # Check for Improvement
    improved = (old_letters != new_letters)
    
    # Calculate Proportion
    return improved.mean()


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    # Determine which students improved (True/False)
    # Inequality means Improvement.
    improved = (
        grades_analysis['Letter Grade Pre-Redemption'] != 
        grades_analysis['Letter Grade Post-Redemption']
    )
    
    # Calculate proportion of improvement per section.
    section_proportions = improved.groupby(grades_analysis['Section']).mean()
    
    # Return the name of the section with the highest proportion
    return section_proportions.idxmax()

def top_sections(grades_analysis, t, n):
    # Calculate Final Exam Proportion (0.0 - 1.0)
    final_score_prop = (
        grades_analysis['Final'] / grades_analysis['Final - Max Points']
    ).fillna(0)
    
    # Filter for students who met the threshold 't'
    high_performers = grades_analysis[final_score_prop >= t]
    
    # Count how many high performers are in each section
    # value_counts returns a Series: Index=SectionName, Value=Count
    section_counts = high_performers['Section'].value_counts()
    
    # Filter for sections that met the count 'n'
    qualifying_sections = section_counts[section_counts >= n].index
    
    # Return sorted alphanumerically
    return np.sort(qualifying_sections)


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    # Prepare the data
    df = grades_analysis[['Section', 'PID', 'Total Points Post-Redemption']].copy()
    
    # Sort the data
    df = df.sort_values(['Section', 'Total Points Post-Redemption'], ascending=[True, False])
    
    # Assign Ranks
    df['Rank'] = df.groupby('Section').cumcount() + 1
    
    # Pivot the table
    ranked_df = df.pivot(index='Rank', columns='Section', values='PID')
    
    # Fill empty cells
    return ranked_df.fillna("")


# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    """
    Returns a plotly heatmap figure showing the proportion of each letter grade per section.
    """
    # Create the Frequency Table (Crosstab)
    heatmap_data = pd.crosstab(
        index=grades_analysis['Letter Grade Post-Redemption'],
        columns=grades_analysis['Section'],
        normalize='columns'
    )
    
    # Enforce the Strict Order
    grade_order = ['A', 'B', 'C', 'D', 'F']
    section_order = sorted(grades_analysis['Section'].unique())
    
    # Use reindex() to sort the rows/cols to match the lists.
    # If a section had 0 'F's (missing from crosstab), reindex will add the row with 0.0.
    
    heatmap_data = heatmap_data.reindex(index=grade_order, columns=section_order, fill_value=0.0)
    
    # Create the Heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Section", y="Letter Grade", color="Proportion"),
        title="Distribution of Letter Grades by Section",
        color_continuous_scale='Blues'
    )
    
    return fig
