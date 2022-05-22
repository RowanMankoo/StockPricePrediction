from scripts import model
import json
from os.path import exists

company = str(input('4 letter company code name as listed on Yahoo finance:'))
steps = int(input('Number of days ahead you wish to predict on:'))
path_to_hyperparams = input('Local path to stored hyperparameters (if none then enter N):')

if path_to_hyperparams == 'N':
    best_params=None

elif exists(path_to_hyperparams):
    with open('hyperparams.json') as f:
        best_params = json.load(f)

else:
    raise ValueError('Please enter a correct path')

m = model.Model(company,steps,best_params)
outputs = m.predict()

pred, prob = outputs[0][0], outputs[1]

# Output to terminal
print('-'*20)
print('')
if pred==1:
    print('Stock price is forecasted to increase')
else:
    print('Stock price is forecasted to decrease')
print('')
print('Probability of stock price increasing:{}'.format(round(prob[0][1],2)))
print('Probability of stock price decreasing:{}'.format(round(prob[0][0],2)))
print('')
print('-'*20)


print(pred)