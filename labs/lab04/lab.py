# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    login['Time'] = pd.to_datetime(login['Time'])
    hours = login['Time'].dt.hour
    
    is_prime_time = (hours >= 16) & (hours < 20)
    
    result_series = is_prime_time.groupby(login['Login Id']).sum()
    
    result_df = result_series.to_frame(name='Time')
    
    return result_df.astype(int)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    login['Time'] = pd.to_datetime(login['Time'])
    
    current_time = pd.Timestamp('2024-01-31 23:59:00')

    def calculate_user_freq(user_times):
        total_logins = len(user_times)
        
        first_login = user_times.min()
        
        duration = current_time - first_login
        
        days_member = duration.days
        
        if days_member == 0:
            return total_logins
            
        return total_logins / days_member

    frequency_series = login.groupby('Login Id')['Time'].agg(calculate_user_freq)
    
    return frequency_series
    
    current_time = pd.Timestamp('2024-01-31 23:59:00')

    def calculate_user_freq(user_times):
        total_logins = len(user_times)
        
        first_login = user_times.min()
        
        duration = current_time - first_login
        
        days_member = duration.days
        
        if days_member == 0:
            return total_logins 
            
        return total_logins / days_member
    frequency_series = login.groupby('Login Id')['Time'].agg(calculate_user_freq)
    
    return frequency_series


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [1, 2]
                         
def cookies_p_value(N):
    n_cookies = 250
    prob_burnt = 0.04
    observed_burnt = 15
    
    simulated_stats = np.random.binomial(n=n_cookies, p=prob_burnt, size=N)
    
    is_extreme = simulated_stats >= observed_burnt
    
    p_value = np.mean(is_extreme)
    
    return p_value


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [1, 4]

def car_alt_hypothesis():
    return [2, 6]

def car_test_statistic():
    return [1, 4]

def car_p_value():
    return 4


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1, 2]


def bhbe_col(heroes):
    is_blond = heroes['Hair color'].str.contains('blond', case=False, na=False)
    is_blue = heroes['Eye color'].str.contains('blue', case=False, na=False)
    
    return is_blond & is_blue

def superheroes_observed_statistic(heroes):
    bhbe_mask = bhbe_col(heroes)
    bhbe_df = heroes[bhbe_mask]
    
    observed_prop = (bhbe_df['Alignment'] == 'good').mean()
    
    return observed_prop


def simulate_bhbe_null(heroes, N):
    """
    Simulates the null hypothesis N times.
    Returns an array of N simulated proportions.
    """
    # The Null assumes the BHBE group is just a random sample from the full population.
    # Population 'good' rate:
    prob_good_pop = (heroes['Alignment'] == 'good').mean()
    
    # Sample size (Size of the BHBE group):
    n_bhbe = bhbe_col(heroes).sum()
    
    # Vectorized Simulation:
    # Flip 'n_bhbe' coins, 'N' times, with P(Head) = prob_good_pop
    simulated_counts = np.random.binomial(n=n_bhbe, p=prob_good_pop, size=N)
    
    # Convert counts to proportions
    simulated_props = simulated_counts / n_bhbe
    
    return simulated_props

def superheroes_p_value(heroes):
    obs_stat = superheroes_observed_statistic(heroes)
    
    sim_stats = simulate_bhbe_null(heroes, 100000)
    
    p_value = np.mean(sim_stats >= obs_stat)
    
    if p_value < 0.01:
        decision = 'Reject'
    else:
        decision = 'Fail to reject'
        
    return [p_value, decision]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(skittles, col='orange'):
    means = skittles.groupby('Factory')[col].mean()
    
    diff = abs(means[0] - means[1])
    
    return diff

def simulate_null(skittles, col='orange'):
    shuffled_labels = skittles['Factory'].sample(frac=1, replace=False).values
    
    shuffled_means = skittles[col].groupby(shuffled_labels).mean()
    
    simulated_diff = abs(shuffled_means[0] - shuffled_means[1])
    
    return simulated_diff

def color_p_value(skittles, col='orange'):
    observed_diff = diff_of_means(skittles, col)
    
    n_repetitions = 1000
    simulated_stats = []
    
    for i in range(n_repetitions):
        sim_stat = simulate_null(skittles, col)
        simulated_stats.append(sim_stat)
    
    simulated_stats = np.array(simulated_stats)
    p_value = np.count_nonzero(simulated_stats >= observed_diff) / n_repetitions
    
    # # Visualization to check work
    # plt.figure(figsize=(8, 5))
    # plt.hist(simulated_stats, bins=20, density=True, alpha=0.6, color='skyblue', label='Null Distribution (Shuffled)')
    # plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2, label=f'Observed Diff ({observed_diff:.2f})')
    # plt.title(f'Permutation Test for "{col}" Skittles')
    # plt.xlabel('Absolute Difference in Means')
    # plt.legend()
    # plt.show()
    
    return p_value


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    return [
        ('yellow', 0.000), 
        ('orange', 0.035),
        ('red', 0.230),
        ('green',  0.445),
        ('purple',    0.979)
    ]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    
def same_color_distribution():
   return (0.007, 'Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'H', 'H', 'P']
