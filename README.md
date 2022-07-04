# Repo Structure

- **logger_tools** contains the logging functions used throughout all scripts in this repo
- **Outputs** is currents an empty folder but will be populated once either the testing scrips are ran or any of the visualise_* methods in modeldev.py
- **Testing** contains all pytest scripts which can be ran to generate reports of the algorithm peformance
- **scripts** contains the python scripts necessary to run the stock price prediction
- main.py is the script to be ran for a paticular company if you wish to get the prediction 
# Time series prediction

Typically when faced with time-series forecasting you will often see models two types of models being used. The first is the classical fmaily of models which all use different variants of Moving Average, Smoothing and Autoregression. The second option is modern deep learning approaches for sequential data such as RNN's,LSTM's, Transformers and so on. Each option has advantages and disadvantages such as the large data requirment for deep learning approaches 

The task of median frequency stock price prediction is one where there is not an abundance of data avalible to train on (assuming we take observations at the daily level) so that automatically rules out the afore mentioned deep learning approaches. Also due to the 
extreme stochstic nature of stock prices, classical methods are not be able of capturing the complexities within the data to deliver reliable predictions. This is the motivation behind this project which aims to use a more modern machine learning algoithm (XGBoost) and apply to predict non trivial time series.

# Proposed Idea

This repo experiments with the idea of using an Xgboost classifier for time series prediction, in the context of stock price prediction. for a fixed period of time (100 days) 

data is pulled directly from Yahoo Finance and goes through data cleaning/feature engineering steps to ensure that each observation (day) has the last 30 days stock 'high price' for each day. This is to ensure that for each observation our XGBoost classifier predicts on it has access to the last 30 days as historical data. Predicting the exact price of a stock is very difficult, again due to the stochatic nature of the stock market. To get around this problem we simplfy the task and instead predict stock price momentum, whether Stock will go down or up, which in turn makes this a classification problem. 

As market trends will chanage a fixed window was used of the last *n* number of days, to ensure our model can caputre the latest trends in the data. Once the model is trained on this window a prediction will be made, using all stock market information from today alongside the stock price high for the past 30 days, to predict whther the price will increase or decrease.
 # Backtesting

 To get an idea of how well the models will run in a real world scenario a walk forward testing method was engineered in which you can choose the amount of money to invest in a company over a 3 year period and the simulation engine would tell you how much you would profit/lose. The testing folder contains multiple pytest files which will run simulations for multiple companies and generate an output report on how well the algoirthm peforms.

# Report

We ran simulations on 15 different companies investing £1000 into each one over a 3 year period and the table below shows the profits made by doing this
# Experiments Ran

- Thresholding o F1 score fail
- use high instead of closing price is more indicative
- trianing on a fixed window of days

