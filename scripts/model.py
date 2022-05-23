from xgboost import XGBClassifier

from scripts import modeldev
from logger_tools import logging_functions

class Model(modeldev.ModelDev):
    def __init__(self, company, steps, hyperparams=None, train_test_split=0.8):
        """Model class used for error analysis and so on

        Args:
            company (str): capital 4 letter company name as listed on Yahoo finance
            steps (int): The number of days ahead our model will be trying to predict
            train_test_split (float): How much of data to start training on before we start walkthrough prediction (ONLY USED FOR MODELDEV)
            hyperparams (dict): previously saved hyperparameters of model
        """
        super().__init__(company, steps, train_test_split=train_test_split, hyperparams=hyperparams)
    
    @logging_functions.logging_decorator
    def predict(self):
        """Trains on all data and predicts self.steps days ahead of today

        Returns:
            pred: Prediction of whether stocks will increase (1) or decrease (0) in 'self.step's days time
            prob: Probabilities of stock decreasing/increasing, respectively
        """

        self.model = XGBClassifier(**self.hyperparams, scale_pos_weight=self.weighting)
        self.model.fit(self.X, self.Y)

        pred = self.model.predict(self.X_current)
        prob = self.model.predict_proba(self.X_current)
        return pred, prob
