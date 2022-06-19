import json
from os.path import exists

import pandas as pd
import pytest
from scripts import modeldev
from logger_tools import logging_functions

config = {
    'report_name':'Report_3.csv',
    'companies': ['MSFT','COKE','AAPL','TSLA','BARC.L','GOOG','AMZN','NVDA','XOM','JPM','WMT','PFE','DIS','CSCO','INTC'],
    'path_to_hyperparams': r'hyperparams.json'
}
steps = [5]*len(config['companies'])

# Load in Hyperparams if stated else set to None
if config['path_to_hyperparams'] == None:
    best_params = None
elif exists(config['path_to_hyperparams']):
    with open('hyperparams.json') as f:
        best_params = json.load(f)

# Store all configurations in a list
configurations = []
for a,b in zip(config['companies'],steps):
    config = (a,b,best_params)
    configurations.append(config)

@pytest.mark.parametrize('company,steps,best_params', configurations)
def test_companies(company,steps,best_params):

    m = modeldev.ModelDev(company=company, steps=steps, training_window=150, hyperparams=best_params)
    m.simulation_walkthrough(1000,0)
    profit = m.ending_money-m.starting_money

    # Test visualisations
    m.visualise_correct_incorrect_probs(test=True)
    m.visualise_class_probs(test=True)
    m.visualise_history(test=True)

    # Create/add to report.csv
    try:
        if exists('Report_3.csv'):
            report = pd.read_csv('Report_3.csv')
        else:
            report = pd.DataFrame(columns = ['Company','Steps','Profit'])
        data = {'Company':company,
                    'Steps':steps,
                    'Profit':profit}
        report = report.append(data,ignore_index=True)
        report.to_csv('Report_3.csv',index=False)

        logging_functions.write_log('Report_creation','info',class_attributes={'company':company,'steps':steps})
    except Exception as e:
        logging_functions.write_log('Report_creation','error',class_attributes={'company':company,'steps':steps}, error_message=repr(e))


    # If not errors assume test passes
    assert True

