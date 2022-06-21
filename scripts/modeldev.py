import json
from collections import Counter
import os
import inspect
import datetime

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgboost
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from scripts import data
from logger_tools import logging_functions


class ModelDev(data.Data):
    def __init__(self, company, steps, training_window, hyperparams=None):
        """Model class used for error analysis and so on

        Args:
            company (str): capital 4 letter company name as listed on Yahoo finance
            steps (int): The number of days ahead our model will be trying to predict
            training_window (float): How many observations to include in trinaing window
            hyperparams (dict): previously saved hyperparameters of model
        """
        # Store all arguments used to initalise class for later use when logging and throughout class methods
        self.company = company
        self.steps = steps
        self.training_window = training_window

        super().__init__(company, steps)
        self.X, self.Y, self.Y_closing_prices, self.X_current = self.X_Y_dataset_creation()
        self.X_current = self.X_current.reshape(-1,self.X_current.shape[0])
       
        # Used for class balancing 
        counter = Counter(self.Y)
        self.weighting = counter[0]/counter[1]

        # If there is no hyper parameter's given we use walkthrough validation to find optimal hyper parameters
        if hyperparams:
            self.hyperparams = hyperparams
        else:
            self.__validate()

        # Fix this to be more neet
        self.__simulation_walkthrough_has_been_called__ = False

        # Make path to store 'visualise_*' plots
        if not os.path.exists('Outputs/'+self.company):
            os.mkdir('Outputs/'+self.company)

        plt.style.use('seaborn')

    @logging_functions.logging_decorator
    def __walkthrough_train_test_split(self, i):
        """Splits dataset according to current step in walkthrough training programme

        Returns:
            X_train, Y_train, X_test, Y_test, Y_closing_pries_test
        """
        return self.X[i:self.training_window+i,:], self.Y[i:self.training_window+i], self.X[self.training_window+i:,:], self.Y[self.training_window+i:], self.Y_closing_prices[self.training_window+i:]

    @logging_functions.logging_decorator
    def __validate(self):
        """Finds optimal hyperparameters and saves them in file called 'hyperparams.json'
        """
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
        self.hyperparams = xgb_grid.best_params_

        with open('hyperparams.json','w') as fp:
            json.dump(self.hyperparams,fp)

    @logging_functions.logging_decorator   
    def __train_and_predict_step(self, X_train, Y_train, X_test, Y_test, Y_closing_prices_test):
        """Trains an instance of our XGBoostClassifier at a current stage in our walkthrough training and then outputs prediction
        """

        model = XGBClassifier(**self.hyperparams, scale_pos_weight=self.weighting)
        model.fit(X_train, Y_train)

        BinaryPredicted = model.predict(X_test[0].reshape(-1,X_test.shape[1]))
        prob = model.predict_proba(X_test[0].reshape(-1,X_test.shape[1]))
        BinaryActual = Y_test[0]  
        ActualStockClosingPrice = Y_closing_prices_test[0]

        return BinaryPredicted, BinaryActual, ActualStockClosingPrice, prob
    
    @logging_functions.logging_decorator
    def __money_simulation(self, BinaryPredicted, ActualStockClosingPrice):
        """Calculates money loss/gain at a current walkthrough step assuming basic trading strategy of selling of stock is predicted to decrease and buying if stock is predicted to increase
        """

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


    @logging_functions.logging_decorator
    def simulation_walkthrough(self, money, stocks):
        """Runs a walkthrough simulation over x amount of days to see profits/losses along with model accuracy

        Args:
            money (int): Starting amount of money you are willing to invest 
            stocks (int): Starting amount of stocks you currently have
        """
        self.money = money
        self.stocks = stocks
        self.starting_money = self.money + self.stocks*self.X[0,3]

        self.preds = []
        self.acc = []
        self.probs = []
        for i in range(0,self.X.shape[0]-self.training_window,self.steps):

            X_train, Y_train, X_test, Y_test, Y_closing_pries_test = self.__walkthrough_train_test_split(i)
            BinaryPredicted, BinaryActual, ActualStockClosingPrice, prob = self.__train_and_predict_step(X_train, Y_train, X_test, Y_test, Y_closing_pries_test)
            # classification error:
            self.preds.append(BinaryPredicted)
            self.acc.append(BinaryActual)
            self.probs.append(prob)

            self.__money_simulation(BinaryPredicted, ActualStockClosingPrice)

        self.ending_money = self.money + self.stocks*self.Y_closing_prices[-1]

        print(f'The starting Money was {self.starting_money} and the ending money was {self.ending_money} so the total profit would have been: £{self.ending_money-self.starting_money}')
        print(f'The accuracy of the model is: {accuracy_score(self.acc,self.preds)}')

        self.__simulation_walkthrough_has_been_called__ = True
        self.__calculate_threshold()

    
    @logging_functions.logging_decorator
    def __calculate_threshold(self):
        """Calculates the threshold to be used for final model based on maximising the F1-score on the simulation_walkthrough tests
        """
        thresholds = np.arange(0, 1, 0.001)
        positive_probs = np.array([x[0][1] for x in self.probs])

        # apply threshold to positive probabilities to create labels
        def to_labels(threshold):
            return (positive_probs >= threshold).astype('int')

        # evaluate each threshold
        scores = [f1_score(self.acc, to_labels(t)) for t in thresholds]
        # get best threshold
        indx = np.argmax(scores)

        if thresholds[indx]==0 or  thresholds[indx]==1:
            # If thresholds being set to 0 or 1, set threshold to standard 0.5
            self.threshold = 0.5
            self.f1_test_score = f1_score(self.acc,self.preds)
        else:
            self.threshold = thresholds[indx]
            self.f1_test_score = scores[indx]



    @logging_functions.logging_decorator
    def visualise_history(self):
        """Visualises history of the stock in question for last n days up to todays date
        """

        # fix to plot actual dates on x
        plt.plot(self.df['Close'])
        n = len(self.df['Close'])
        plt.title(f'History of closing prices for last {n} days', fontsize=20)
        plt.xlabel('Day', fontsize=15)
        plt.ylabel('Closing price', fontsize=15)

        # save plot to Outputs
        save_path = os.path.join('Outputs/'+self.company, inspect.stack()[0][3]+'-'+datetime.date.today().strftime('%Y_%m_%d')+'.png')
        plt.savefig(save_path)
        plt.close()


    @logging_functions.logging_decorator
    def visualise_correct_incorrect_probs(self):
        """Visualises the distribution of the predicted probabilities, can use to see whether to set a threshold for probability
        """

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
        plt.title('Distribution of both correct/incorrect predicted probabilities', fontsize=20)
        plt.legend()
        plt.ylabel('Predicted Probability of increasing', fontsize=15)
        
        # save plot to Outputs
        save_path = os.path.join('Outputs/'+self.company, inspect.stack()[0][3]+'-'+datetime.date.today().strftime('%Y_%m_%d')+'.png')
        plt.savefig(save_path)
        plt.close()


    @logging_functions.logging_decorator
    def visualise_class_probs(self):
        """Visualises the predicted probability distributions for each class, use to see whether one class is more accuracte than the other
        """

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

        data = [increased,decreased]
        plt.boxplot(data, showfliers=False)

        for i,dataset in enumerate(data):
            y = dataset
            x = np.random.normal(i + 1, 0.04, len(y))
            plt.scatter(x,y)

        plt.xticks([1,2],['Increasing','Decreasing'])
        plt.title('Predicted probability distributions of each class', fontsize=20)
        plt.xlabel('Actual Class', fontsize=15)
        plt.ylabel('Predicted Probability of increasing', fontsize=15)
        
        # save plot to Outputs
        save_path = os.path.join('Outputs/'+self.company, inspect.stack()[0][3]+'-'+datetime.date.today().strftime('%Y_%m_%d')+'.png')
        plt.savefig(save_path)
        plt.close()


    @logging_functions.logging_decorator
    def feature_importance(self,columns):

        if self.__predict_has_been_called__:
            return sorted(zip(self.model.feature_importances_,columns),reverse=True)
        else:
             raise ValueError('predict method has not been called yet')




