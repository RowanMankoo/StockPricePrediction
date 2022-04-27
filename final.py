import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgboost
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

# add feature importance 
# only sell if probability is high
# fixed training window
# invest fixed amount of money
# start date
# load validation weights in
# add more info to simulation info
# add final predict
# Try new hyper parameter optimisation?
# double check money simulation walkthrough
# include last 30 days of high?

class Model_Dev:
    def __init__(self, X, Y, Y_closing_prices, X_current, train_test_split=0.8, hyperparams=None):
        self.prediction_window = 5 # fix this!
        self.X = X
        self.Y = Y
        self.Y_closing_prices = Y_closing_prices
        self.X_current = X_current.reshape(-1,X_current.shape[0])
        self.n = int(self.X.shape[0]*train_test_split)

        self.stocks = 5
        self.money = 0

        # If there is no hyper parameter's given we use walkthrough validation to find optimal hyper parameters
        if hyperparams:
            self.best_params = hyperparams
        else:
            self.__validate()

    def __walkthrough_train_test_split(self, i):
        # Return X_train, Y_train, X_test, Y_test, Y_closing_pries_test
        return self.X[:self.n+i,:], self.Y[:self.n+i], self.X[self.n+i:,:], self.Y[self.n+i:], self.Y_closing_prices[self.n+i:]

    def __validate(self):
        # Hyperparameter tunning
        # chnage this to do all but last 5?
        X_train, Y_train, _,_,_ = self.__walkthrough_train_test_split(0)         
        params = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
                }

        xgb1 = xgboost.XGBClassifier()
        xgb_grid = GridSearchCV(xgb1,
                                params,
                                cv = TimeSeriesSplit(),
                                n_jobs = -1,
                                verbose=True)
        xgb_grid.fit(X_train,Y_train)
        self.best_params = xgb_grid.best_params_

        with open('hyperparams.json','w') as fp:
            json.dump(self.best_params,fp)
    
    def __train_and_predict_step(self, X_train, Y_train, X_test, Y_test, Y_closing_prices_test):
        
        model = XGBClassifier(**self.best_params)
        model.fit(X_train, Y_train)

        BinaryPredicted = model.predict(X_test[0].reshape(-1,X_test.shape[1]))
        BinaryActual = Y_test[0]  
        ActualStockClosingPrice = Y_closing_prices_test[0]

        return BinaryPredicted, BinaryActual, ActualStockClosingPrice
    
    def money_simulation(self, BinaryPredicted, ActualStockClosingPrice):

        if BinaryPredicted==0:
            if self.stocks!=0:
                # If stocks are predicted to go down then sell
                self.money += ActualStockClosingPrice*self.stocks
                self.stocks = 0
            else:
                pass
        else:
            # If stocks are predicted to go up then buy
            self.stocks += self.money//ActualStockClosingPrice
            self.money -= (self.money//ActualStockClosingPrice)*ActualStockClosingPrice
    
    def feature_importance(self,columns):

        # CHANGE THIS TO TRY CATCH STATEMENT
        if self.__predict_has_been_called__:
            return sorted(zip(self.model.feature_importances_,columns),reverse=True)
        else:
             raise ValueError('predict method has not been called yet')


    def visualise(self):
        
        plt.plot(list(pred)*(prediction_window+1),color='red',label='Predictions')
        plt.plot([starting_price]+list(Y_closing_prices[max(i,0):max(i,0)+prediction_window]),color='blue',label='Actual')
        plt.legend()
        plt.show()
    
    def print_money_gain():
        print('f')
    
    def print_accuracy(self):

        #preds = list(map(lambda x: 1 if x>0.5 else 0, self.preds))
        print(accuracy_score(self.acc,self.preds))

    def simulation_walkthrough(self):

        self.starting_money = self.money + self.stocks*self.X[self.n,3]

        self.preds = []
        self.acc = []
        for i in range(0,self.X.shape[0]-self.n,self.prediction_window):

            X_train, Y_train, X_test, Y_test, Y_closing_pries_test = self.__walkthrough_train_test_split(i)
            BinaryPredicted, BinaryActual, ActualStockClosingPrice = self.__train_and_predict_step(X_train, Y_train, X_test, Y_test, Y_closing_pries_test)
            # classification error:
            self.preds.append(BinaryPredicted)
            self.acc.append(BinaryActual)

            self.money_simulation(BinaryPredicted, ActualStockClosingPrice)

        self.ending_money = self.money + self.stocks*self.Y_closing_prices[-1]

        print(f'The starting Money was {self.starting_money} and the ending money was {self.ending_money} so the total profit would have been: £{self.ending_money-self.starting_money}')
        self.print_accuracy()

    def predict(self):

        self.__predict_has_been_called__ = True

        self.model = XGBClassifier(**self.best_params)
        self.model.fit(self.X, self.Y)

        pred = self.model.predict(self.X_current)
        
        return pred




