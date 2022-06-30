If you plan on building upon this repo then you must ensure two things for the logging capabilities to continue to work:

1. All classes need to store variables used to initialise class as attributes in the init stage (for Model this is done via super().__init__)
2. Varibles used to initalise a parent class must share the same name across child classes

 