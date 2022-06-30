All files in this repository contain pytest code to test the model along with generating a test report. To get an idea of why any of the test cases may have failed you can look in the output.log file which are generated when the script is run. Each error log should tell you which class has failed, and specifically which method caused the failure along with the parmeters passed to it.

# test_model

This is a very simple test script which will test the model to see if it can output a prediction for multiple different companies with varying step sizes. If the model fails to generate a prediction for whatever reason it is considered to have failed. This is the faster testing script as the simulation engine is not ran for each company.

# test_dev_model

This script will run tests on all of the evaluation methods methods of the the ModelDev class. Whilst doing this a report will be generated for each company used in the test cases.
