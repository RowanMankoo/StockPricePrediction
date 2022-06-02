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
fixed training window
start date
add more info to simulation info
polish final predict
Try new hyper parameter optimisation?
add hyperparam naming functionality
include last 30 days of high?
Venv
progress bars or print init statements
ADD WEEKEND INDICATOR!
add classification statistics
add fractional shares
add testing
try violin plot for visualise_class_probs
add report from pytest