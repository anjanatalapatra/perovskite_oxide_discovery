# -*- coding: utf-8 -*-
  
#Anjana Talapatra 2020

__version__ = '0.1.0'
"""
ML module for perovskite oxide discovery based on Random Forests

This package is organized as follows:
- main.py : driver script.
- 'main.py runall predict analyze save' will run the whole ML chain of commands and execute all four models, and save prediction results after every model ( formability, stability, insulator, bandgap regression). Omitting save will only save the final regression results.
- 'main.py XXXX' for XXXX in [ formability, stability, insulator, bg_regression] will run each model separately
- 'main.py XXXX analyze' will run each model separately and create the performance figures ( roc curves, learning curves)
- 'main.py runall predict_training' will run all the models, and then predict for the training data
- 'main.py core_energy' will run all the models, and then generate the core_energy dataset with predictions.

Results are stored in:
- ./formability_results
- ./stability_results
- ./insulator_results
- ./bg_regression_results
- ./predicted_candidates - saved candidate predictions based on arguments chosed
- ./training_candidates - saved predictions on training dataset
- ./core_energy_dataset 
Subpackages:

- datasets: pickled datasets for element data, training compounds, candidate compounds
- ML_utilities: all functionality to deal with regression and classification models

- data_utilities: all functionality to deal with generating datasets with requisite features

- ML_models: all functionality dealing with actual models. To be converted into classes at some point

Modules:

- tools: 

- utils: functions that you probably won't need, but that subpackages use

- version: holds the current API version

- exceptions: defines our custom exception classes

"""
