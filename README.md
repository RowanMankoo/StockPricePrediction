# Stock Reading Price Prediction

- xgboost classification which classifies momentum of stocks after n days
- previous 30 days closing prices added
- pulls in yahoo finance data
- includes a money simulation which will use walkthrough training to estimate how much money you would gain/lose after x days of trading with standard stratergy
- can stor hyper params as takes long to find them
- can show feature importance
- deales with class imbalance

'added class balancing technique along with visulisation method of predicted probabilities'

## To Do

develop feature importance functionality
only sell if probability is high
add more info to simulation info
polish final predict
Try new hyper parameter optimisation?
add hyperparam naming functionality
Venv
progress bars or print init statements
##add classification statistics
add fractional shares
test main.py
try violin plot for visualise_class_probs
neaten up config on report path/name and also report for other pytest? / merge them together??
early stoppiing for xgboost?
## validate on year prior?
## money sim on last year in report but get threshold basewd on most data avalible?
add validation to set threshold?
# Run pytest to see if changing threshold is acxc making a diff
# Why is number of predictions so small?

f1score, 1 year window