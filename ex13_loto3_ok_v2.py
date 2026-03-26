# -*- coding: utf-8 -*-
"""
Genetic Algorithm prediction Serbia Lottery 7/39
"""

import sys
sys.path.append('..')


import numpy as np
import pandas as pd
from geneticalgorithm2 import GeneticAlgorithm2 as ga

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------
csv_file = "/data/loto7_4586_k24.csv"  
# CSV sa 4586 prethodnih kombinacija, kolone: num1,num2, ... ,num7

num_numbers = 7
min_num = 1
max_num = 39
population_size = 220
iterations = 180

# -------------------------------------------------
# LOAD HISTORICAL DATA
# -------------------------------------------------
df = pd.read_csv(csv_file, header=None)
history = df.values  # shape (4586, 7)
history = history.astype(int)
history_sorted = np.sort(history, axis=1)

# -------------------------------------------------
# FITNESS FUNCTION
# -------------------------------------------------
def loto_fitness(candidate):
    """
    Fitness function: koliko kandidat lici na prethodne kombinacije.
    Ideja: kandidat koji je sto 'slicniji' proslim kombinacijama ima manju vrednost.
    Ovde koristimo sumu minimalnih udaljenosti po brojevima.
    """
    candidate_int = np.round(candidate).astype(int)
    candidate_int = np.clip(candidate_int, min_num, max_num)
    candidate_sorted = np.sort(candidate_int)

    # v2: kazna za duplikate (realna loto kombinacija ima 7 različitih brojeva)
    uniq = len(np.unique(candidate_sorted))
    duplicate_penalty = (num_numbers - uniq) * 1000
    
    # v2: brži i stabilniji obračun distance (vektorizovano)
    # history_sorted shape: (N, 7), candidate_sorted shape: (7,)
    abs_diff = np.abs(history_sorted - candidate_sorted)  # (N, 7)
    dist = np.sum(np.min(abs_diff, axis=1))
    return float(dist + duplicate_penalty)  # GA minimizuje fitness

# -------------------------------------------------
# VARIABLE BOUNDS
# -------------------------------------------------
varbound = [[min_num, max_num]] * num_numbers

# -------------------------------------------------
# CREATE GA MODEL
# -------------------------------------------------
model = ga(
    dimension=num_numbers,
    variable_type='int',
    variable_boundaries=varbound,
    algorithm_parameters={
        'max_num_iteration': iterations,
        'population_size': population_size,
        'mutation_probability': 0.18,
        'elit_ratio': 0.07,
        'parents_portion': 0.35,
        'crossover_type': 'uniform',
        'selection_type': 'roulette',
        'max_iteration_without_improv': 40
    }
)

# -------------------------------------------------
# RUN GA
# -------------------------------------------------
# Koristimo istoriju kao startnu populaciju, nema random starta
model.run(function=loto_fitness, no_plot=True, disable_printing=False,
          start_generation=(history, None))

# -------------------------------------------------
# BEST PREDICTION
# -------------------------------------------------
best_candidate = np.round(model.result.variable).astype(int)
best_candidate = np.clip(best_candidate, min_num, max_num)
best_candidate_sorted = np.sort(best_candidate)
print()
print("Predicted next loto 7/39 combination:", best_candidate_sorted)
print()
"""
First scores are made from gotten variables (by 49.90370488166809 secs, about 0.01093420352359073 for each creature)

Best score before optimization: 
2527 ____________________ 
0.4% GA is running...1 gen from 255...best ____________________ 
0.8% GA is running...2 gen from 255...best ____________________ 
1.2% GA is running...3 gen from 255...best ____________________ 
1.6% GA is running...4 gen from 255...best ____________________ 
2.0% GA is running...5 gen from 255...best ____________________ 
2.4% GA is running...6 gen from 255...best |___________________ 
2.7% GA is running...7 gen from 255...best |___________________ 
3.1% GA is running...8 gen from 255...best |___________________ 
3.5% GA is running...9 gen from 255...best |___________________ 
3.9% GA is running...10 gen from 255...best|___________________ 

...

_ 93.3% GA is running...238 gen from 255...be|||||||||||||||||||
_ 93.7% GA is running...239 gen from 255...be|||||||||||||||||||
_ 94.1% GA is running...240 gen from 255...be|||||||||||||||||||
_ 94.5% GA is running...241 gen from 255...be|||||||||||||||||||
_ 94.9% GA is running...242 gen from 255...be|||||||||||||||||||
_ 95.3% GA is running...243 gen from 255...be|||||||||||||||||||
_ 95.7% GA is running...244 gen from 255...be|||||||||||||||||||
_ 96.1% GA is running...245 gen from 255...be|||||||||||||||||||
_ 96.5% GA is running...246 gen from 255...be|||||||||||||||||||
_ 96.9% GA is running...247 gen from 255...be|||||||||||||||||||
_ 97.3% GA is running...248 gen from 255...be|||||||||||||||||||| 
97.6% GA is running...249 gen from 255...be|||||||||||||||||||| 
98.0% GA is running...250 gen from 255...be|||||||||||||||||||| 
98.4% GA is running...251 gen from 255...be|||||||||||||||||||| 
98.8% GA is running...252 gen from 255...be|||||||||||||||||||| 
99.2% GA is running...253 gen from 255...be|||||||||||||||||||| 
99.6% GA is running...254 gen from 255...be|||||||||||||||||||| 
100.0% GA is running... STOP! limit of iter
  

 The best found solution:
 [ 2.  8. x. y. z. 33. 39.]

 Objective function:
 2477.0

 Used generations: 255
 Used time: 8.74e+03 seconds

Predicted next loto 7/39 combination: [ 2  8 x y z 33 39]
"""

# -------------------------------------------------
# SAVE RESULT TO CSV
# -------------------------------------------------
pd.DataFrame([best_candidate_sorted], columns=[f'num{i+1}' for i in range(num_numbers)]) \
    .to_csv("/data/loto_prediction.csv", index=False)



"""
poboljšano u v2:

fitness je ubrzan (vektorizovan obračun distance),
dodat penal za duplikate u kandidatu (7 različitih brojeva),
GA parametri fino podešeni za brži konvergentniji rad (iterations=180, population_size=220, max_iteration_without_improv=40, itd.),
"""
