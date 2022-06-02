import json
from os.path import exists

import pandas as pd
import pytest
from scripts import modeldev

companies = ['MSFT','COKE','AAPL','TSLA','BARC.L']
steps = [2,3,4,5,6]
path_to_hyperparams = r'hyperparams.json'         # Set this if you wish to test validate method for each model also, note this will vastly increase testing time

# Load in Hyperparams if stated else set to None
if path_to_hyperparams == None:
    best_params = None
elif exists(path_to_hyperparams):
    with open('hyperparams.json') as f:
        best_params = json.load(f)

# Store all permutations in a list
configurations = []
for a,b in zip(companies,steps):
    config = (a,b,best_params)
    configurations.append(config)

@pytest.mark.parametrize('company,steps,best_params', configurations)
def test_companies(company,steps,best_params):

    m = modeldev.ModelDev(company,steps,best_params)
    m.simulation_walkthrough(1000,0)
    profit = m.ending_money-m.starting_money

    # Test visualisations
    m.visualise_correct_incorrect_probs(test=True)
    m.visualise_class_probs(test=True)
    m.visualise_history(test=True)

    # Create/add to report.csv
    if exists('Report.csv'):
        report = pd.read_csv('Report.csv')
    else:
        report = pd.DataFrame(columns = ['Company','Steps','Profit'])
    data = {'Company':company,
                'Steps':steps,
                'Profit':profit}
    report = report.append(data,ignore_index=True)
    report.to_csv('Report.csv',index=False)

    # If not errors assume test passes
    assert True

