import json
from os.path import exists
import datetime

import pandas as pd
import pytest
from sklearn.metrics import f1_score

from scripts import modeldev
from logger_tools import logging_functions

with open('hyperparams.json') as f:
    best_params = json.load(f)

company = 'XOM'
steps = 5

m = modeldev.ModelDev(company=company, steps=steps, training_window=150, hyperparams=best_params)
m.simulation_walkthrough(1000,0)
# Report summary metrics
profit = m.ending_money-m.starting_money
f1_score_acc = f1_score(m.acc,m.preds)
f1_score_possible = m.f1_test_score
threshold = m.threshold

# Test visualisations
m.visualise_correct_incorrect_probs()
m.visualise_class_probs()
m.visualise_history()