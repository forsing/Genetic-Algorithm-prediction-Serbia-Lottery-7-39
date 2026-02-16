# -*- coding: utf-8 -*-
"""
Genetic Algorithm Serbia Lottery 7/39 prediction 
"""

import sys
sys.path.append('..')


import numpy as np
import pandas as pd
from geneticalgorithm2 import GeneticAlgorithm2 as ga

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------
csv_file = "/Users/4c/Desktop/GHQ/data/loto7_4564_k13.csv"  
# CSV sa 4564 prethodnih kombinacija, kolone: num1,num2, ... ,num7

num_numbers = 7
min_num = 1
max_num = 39
population_size = 200
iterations = 255

# -------------------------------------------------
# LOAD HISTORICAL DATA
# -------------------------------------------------
df = pd.read_csv(csv_file, header=None)
history = df.values  # shape (4564, 7)
history = history.astype(int)

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
    
    # Distance to each historical draw
    dist = np.sum([np.min(np.abs(candidate_sorted - np.sort(row))) for row in history])
    return dist  # GA minimizuje fitness

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
        'mutation_probability': 0.2,
        'elit_ratio': 0.05,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'selection_type': 'roulette',
        'max_iteration_without_improv': None
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
 [ 2.  8. 13. 21. 27. 33. 39.]

 Objective function:
 2477.0

 Used generations: 255
 Used time: 8.74e+03 seconds

Predicted next loto 7/39 combination: [ 2  8 13 21 27 33 39]
"""

# -------------------------------------------------
# SAVE RESULT TO CSV
# -------------------------------------------------
pd.DataFrame([best_candidate_sorted], columns=[f'num{i+1}' for i in range(num_numbers)]) \
    .to_csv("/Users/4c/Desktop/GHQ/data/loto_prediction.csv", index=False)



# https://github.com/PasaOpasen/geneticalgorithm2

# https://pasaopasen.github.io/geneticalgorithm2/geneticalgorithm2/geneticalgorithm2.html