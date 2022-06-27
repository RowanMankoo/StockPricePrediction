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
# Weight latte observations more

f1score, 1 year window


# Repo Structure

- **logger_tools** contains the logging functions used throughout all scripts in this repo
- **Outputs** is currents an empty folder but will be populated once either the testing scrips are ran or any of the visualise_* methods in modeldev.py
- **Testing** contains all pytest scripts which can be ran to generate reports of the algorithm peformance
- **scripts** contains the python scripts necessary to run the stock price prediction

# Time series prediction

Typically when faced with time-series forecasting you will often see models two types of models being used. The first is the classical fmaily of models which all use different variants of Moving Average, Smoothing and Autoregression. The second option is modern deep learning approaches for sequential data such as RNN's,LSTM's, Transformers and so on. Each option has advantages and disadvantages such as the large data requirment for deep learning approaches 

The task of median frequency stock price prediction is one where there is not an abundance of data avalible to train on so that automatically rules out the afore mentioned deep learning approaches. Also due to the 
extreme stochstic nature of stock prices, classical methods are not be able of capturing the complexities within the data to deliver reliable predictions. This is the motivation behind this project which aims to use a more modern machine learning algoithm (XGBoost) and apply to predict non trivial time series.

# Proposed Idea

This repo experiments with the idea of using an Xgboost classifier for time series prediction, in the context of stock price prediction. for a fixed period of time (100 days) 

data is pulled directly from Yahoo Finance and goes through data cleaning/feature engineering steps to ensure that each observation (day) has the ..., last 30 days stock high price for each day. Predicting the exact price of a stock is very difficult, again due to the stochatic nature of the stock market. To get around this problem simplfy the task and instead predict stock price momentum, whether sotck will go down or up, which in turn makes this a classification problem. 

 Only the last *n* number of days are used

 # Test test?

 To get an idea of how well the models will run in a real world scenario a walk forward testing method was engineered.

# Experiments Ran

- Thresholding o F1 score fail
- use high instead of closing price is more indicative
- trianing on a fixed window of days

