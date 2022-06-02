Each .py file contains a specific class to be used in the final model. Note that each of the classes inherent from eachother in the order: **Data** -> **ModelDev** -> **Model**.

# Data

The data class contains multiple methods such as; pulling the Yahoo finance data, cleaning the raw data, feature engineering, splitting of targets and inputs.

# ModelDev

This class contains all methods related to testing/evaluating the peformance your said model before you use it for real time prediction. There are two main functionalitys of this class:

1. Run a simulation walkthrough to see how much money you would gain if you used the trained model. This method utalises walkthrough forward validation to retrain the model everytime new data becomes avalible, this ensure we will have an accurate idea of how the model can peform. 

2. Multiple Visualise funcitions can be used to see how well the model is peforming and what type of errors the model tends to skew towards.

# Model

This is the very final class for the model and only this would need to be ran to generate predictions. Note that it also inherits from the ModelDev class allowing you to evaluate your current model strategy also.  