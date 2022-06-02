note for logger to work all classes need to store variables used to initialise class as attributes in the init stage (for Model this is done via super().__init__)

note variables used for init must be labelled same thing acrosss child classes for logging to work

depending on how detailed you want logs you can delete some of the logging decorators from each class method which will be repeated multiple times

 