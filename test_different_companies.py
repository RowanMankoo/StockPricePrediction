import json
from os.path import exists

import pytest
from scripts import model

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

    m = model.Model(company,steps,best_params)
    outputs = m.predict()

    pred, prob = outputs[0][0], outputs[1]

    # If not errors assume test passes
    assert True

