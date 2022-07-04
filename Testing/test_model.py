import json
from os.path import exists

import pytest
from scripts import model

config = {
    'report_name':'Report_3.csv',
    'companies': ['MSFT','COKE','AAPL','TSLA','BARC.L','AMZN','NVDA','XOM','JPM','WMT','PFE','DIS','CSCO','INTC'],
    'path_to_hyperparams': r'hyperparams.json'
}
steps = [5]*len(config['companies'])

# Load in Hyperparams if stated else set to None
if config['path_to_hyperparams'] == None:
    best_params = None
elif exists(config['path_to_hyperparams']):
    with open('hyperparams.json') as f:
        best_params = json.load(f)

# Store all permutations in a list
configurations = []
for a,b in zip(config['companies'],steps):
    config = (a,b,best_params)
    configurations.append(config)

@pytest.mark.parametrize('company,steps,best_params', configurations)
def test_companies(company,steps,best_params):

    m = model.Model(company=company,
                    steps=steps,
                    training_window=150,
                    hyperparams=best_params)
    outputs = m.predict()

    pred, prob = outputs[0][0], outputs[1]

    # If not errors assume test passes
    assert True

