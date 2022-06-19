import datetime
import time

import numpy as np
import pandas as pd

from logger_tools import logging_functions

class Data:
    
    def __init__(self, company, steps):
        """Initialises data class

        Args:
            company (str): capital 4 letter company name as listed on Yahoo finance
            steps (int): The number of days ahead our model will be trying to predict 
        """
        self.company = company
        self.steps = steps

        self.__pull_yahoo_data()
        self.__feature_engineering()
        self.__column_names()
    
    @logging_functions.logging_decorator
    def __pull_yahoo_data(self):
        """Pulls in historical Yahoo finance data
        """

        start_date = int(time.mktime(datetime.datetime(2019,6,13).timetuple())) # automate this to a 3 month period 
        end_date = int(time.time())
        interval = '1d' #1d 1wk 1m
        query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{self.company}?period1={start_date}&period2={end_date}&interval={interval}&events=history&includeAdjustedClose=true'
        print(query_string)
        self.df = pd.read_csv(query_string)

    @logging_functions.logging_decorator
    def __feature_engineering(self):
        """Adds useful feature's for time series prediction
        """

        # Create columns of previous days closing price
        for i in range(1,31):
            self.df[str(i)]=np.nan
        # Fill in columns
        for i,_ in self.df.iterrows():
            if i<30:
                continue
            else:
                for j in range(1,31):
                    self.df.loc[i,str(j)] = self.df.loc[i-j,'High']
        self.df = self.df.iloc[30:,:] # lose first 30 days like this but oh well

        # datetime.datetime.strptime('2019-06-12','%Y-%m-%d').weekday()
        self.df['Weekday_indicator'] = self.df['Date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').weekday())
        # Drop the date
        self.df = self.df.drop(columns='Date')

        # Add 30DayStockPriceSum
        self.df['30DayStockPriceSum'] = self.df.iloc[:,6:].sum(axis=1)

    @logging_functions.logging_decorator
    def __column_names(self):
        """Return the column names in order

        Returns:
            _type_: _description_
        """

        self.column_names = self.df.columns
    
    @logging_functions.logging_decorator
    def X_Y_dataset_creation(self):
        """Generates a typical XY inputs and targets 

        Returns:
            X (numpy.ndarray): input to model to train/evaluate on
            X_current (numpy.ndarray): latest day's observation for which model will make final prediction on
            Y (list): binary array indicating whether closing prices go up (1) or down (0) in 'self.steps' number of days
            Y_closing_prices (numpy.ndarray): Actual closing prices which are 'self.steps' number of days ahead
        """

        X = np.array(self.df[:-self.steps])
        X_current = np.array(self.df.iloc[-1,:])
        Y_closing_prices = np.array(self.df['Close'].shift(periods=-self.steps)[:-self.steps])

        Y = []
        
        for start,finish in zip(X[:,3],Y_closing_prices):
            if finish>start:
                Y.append(1)
            else:
                Y.append(0)

        return X, Y, Y_closing_prices, X_current
