import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgboost
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

import data

# develop feature importance functionality
# only sell if probability is high
# fixed training window
# invest fixed amount of money
# start date
# add more info to simulation info
# polish final predict
# Try new hyper parameter optimisation?
# double check money simulation walkthrough
### include last 30 days of high?
# Venv
# progress bars or print init statements

class ModelDev(data.Data):
    def __init__(self, company, steps, train_test_split=0.8, hyperparams=None):

        # Initialise data class and prepare data for model
        self.steps = steps
        super().__init__(company, steps)
        self.X, self.Y, self.Y_closing_prices, self.X_current = self.X_Y_dataset_creation()
        self.X_current = self.X_current.reshape(-1,self.X_current.shape[0])
        self.n = int(self.X.shape[0]*train_test_split)
       
        # Used for class balancing 
        counter = Counter(self.Y)
        self.weighting = counter[0]/counter[1]

        # Used for money simulation
        # MOVE THIS TO MONEY SIM METHOD
        self.stocks = 5
        self.money = 0

        # If there is no hyper parameter's given we use walkthrough validation to find optimal hyper parameters
        if hyperparams:
            self.best_params = hyperparams
        else:
            self.__validate()

        # Fix this to be more neet
        self.__simulation_walkthrough_has_been_called__ = False

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

        xgb1 = xgboost.XGBClassifier(scale_pos_weight=self.weighting)
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
        
        model = XGBClassifier(**self.best_params, scale_pos_weight=self.weighting)
        model.fit(X_train, Y_train)

        BinaryPredicted = model.predict(X_test[0].reshape(-1,X_test.shape[1]))
        prob = model.predict_proba(X_test[0].reshape(-1,X_test.shape[1]))
        BinaryActual = Y_test[0]  
        ActualStockClosingPrice = Y_closing_prices_test[0]

        return BinaryPredicted, BinaryActual, ActualStockClosingPrice, prob
    
    def __money_simulation(self, BinaryPredicted, ActualStockClosingPrice):

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

    def simulation_walkthrough(self):

        self.starting_money = self.money + self.stocks*self.X[self.n,3]

        self.preds = []
        self.acc = []
        self.probs = []
        for i in range(0,self.X.shape[0]-self.n,self.steps):

            X_train, Y_train, X_test, Y_test, Y_closing_pries_test = self.__walkthrough_train_test_split(i)
            BinaryPredicted, BinaryActual, ActualStockClosingPrice, prob = self.__train_and_predict_step(X_train, Y_train, X_test, Y_test, Y_closing_pries_test)
            # classification error:
            self.preds.append(BinaryPredicted)
            self.acc.append(BinaryActual)
            self.probs.append(prob)

            self.__money_simulation(BinaryPredicted, ActualStockClosingPrice)

        self.ending_money = self.money + self.stocks*self.Y_closing_prices[-1]

        print(f'The starting Money was {self.starting_money} and the ending money was {self.ending_money} so the total profit would have been: £{self.ending_money-self.starting_money}')
        print(accuracy_score(self.acc,self.preds))

        self.__simulation_walkthrough_has_been_called__ = True

    def visualise_correct_incorrect_probs(self):

        if self.__simulation_walkthrough_has_been_called__:
            pass
        else:
            raise ValueError('simulation_walkthrough method has not been called yet')

        correct = []
        incorrect = []
        for acc,probs in zip(self.acc,self.probs):
            if acc==1 and probs[0,1]>0.5:
                correct.append(probs[0,1])
            elif acc==0 and probs[0,1]<0.5:
                correct.append(probs[0,1])
            else:
                incorrect.append(probs[0,1])

        plt.style.use('seaborn')
        plt.boxplot(correct+incorrect, showfliers=False)

        data = [correct,incorrect]
        for dataset in data:
            if dataset == correct:
                color = 'blue'
                label = 'Correct'
            else:
                color = 'red'
                label = 'Incorrect'
            y = dataset
            x = np.random.normal(1, 0.03, len(y))
            plt.scatter(x,y,color=color, label=label)

        plt.xticks([1],[''])
        plt.title('Distribution of both correct/incorrect predicted probabilities')
        plt.legend()
        plt.ylabel('Predicted Probability of increasing')
        plt.show()

    def visualise_class_probs(self):

        if self.__simulation_walkthrough_has_been_called__:
            pass
        else:
            raise ValueError('simulation_walkthrough method has not been called yet')

        increased = []
        decreased = []
        for acc,probs in zip(self.acc,self.probs):
            if acc==1:
                increased.append(probs[0,1])
            else:
                decreased.append(probs[0,1])

        plt.style.use('seaborn')
        data = [increased,decreased]
        plt.boxplot(data, showfliers=False)

        for i,dataset in enumerate(data):
            y = dataset
            x = np.random.normal(i + 1, 0.04, len(y))
            plt.scatter(x,y)

        plt.xticks([1,2],['Increasing','Decreasing'])
        plt.title('Predicted probability distributions of each class')
        plt.xlabel('Actual Class')
        plt.ylabel('Predicted Probability of increasing')
        plt.show()

    def predict(self):

        self.__predict_has_been_called__ = True

        self.model = XGBClassifier(**self.best_params, scale_pos_weight=self.weighting)
        self.model.fit(self.X, self.Y)

        pred = self.model.predict(self.X_current)
        
        return pred




