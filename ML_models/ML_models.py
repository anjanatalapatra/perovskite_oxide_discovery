import os, shutil, sys
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from ML_utilities.RFR_functions import fit_rfr, evaluate, predict_regression, plot_learning_curve, plot_confidence_intervals, plot_regression,pred_ints
from ML_utilities.RFC_functions import RFC_performance
from ML_utilities.ML_plotting import plot_feature_importance, plot_roc_curves, plot_pdp_plots
from data_utilities.generate_data import generate_feature_labels
#from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc
from sklearn import preprocessing

rc('font',**{'family':'serif','serif':['Helvetica']})
rc('axes', axisbelow=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_perovskite_formability(perovskite_training_data):  
    cwd = os.getcwd()
    feature_list = ['A_HOMO_diff', 'A_HOMO_sum', 'A_IE_diff', 'A_IE_sum', 'A_LUMO_diff', 'A_LUMO_sum', 'A_X_diff', 'A_X_sum',
        'A_Z_radii_diff', 'A_Z_radii_sum', 'A_e_affin_diff', 'A_e_affin_sum', 'B_HOMO_diff', 'B_HOMO_sum', 'B_IE_diff', 'B_IE_sum',
        'B_LUMO_diff', 'B_LUMO_sum', 'B_X_diff', 'B_X_sum', 'B_Z_radii_diff', 'B_Z_radii_sum', 'B_e_affin_diff', 'B_e_affin_sum',
        'tau','mu', 'mu_A_bar','mu_B_bar']
    feature_labels = generate_feature_labels(feature_list)
    features, labels = create_training_data(perovskite_training_data, feature_list,target_label='Perovskite',model_type='RFC')
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 12)
    clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators =100, max_depth = 22, max_features='auto',min_samples_split = 2, min_samples_leaf =1,class_weight='balanced')
    # Feature importances
    y_score = clf.fit(train_features, train_labels)
    plot_feature_importance(clf, feature_labels, filename='formability_feature_importance.png', n_features=20, palette="Blues_d", dir_name='formability_results')
    # Classifier performance
    pred = clf.predict(test_features) 
    RFC_performance(clf,test_labels,pred,train_features,train_labels,labels,features,c_type='Perovskite formability classification')   
     # ROC and Precision Recall curves
    if 'analyze' in sys.argv:
        dst = 'formability_results'
        plot_roc_curves(test_features,test_labels, clf,f_name=dst)	
    return feature_list,clf
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_perovskite_stability(stability_training_data):
    cwd = os.getcwd()
    feature_list = ['A_HOMO_diff', 'A_HOMO_sum', 'A_IE_diff', 'A_IE_sum', 'A_LUMO_diff', 'A_LUMO_sum', 'A_X_diff', 'A_X_sum',
        'A_Z_radii_diff', 'A_Z_radii_sum', 'A_e_affin_diff', 'A_e_affin_sum', 'B_HOMO_diff', 'B_HOMO_sum', 'B_IE_diff', 'B_IE_sum',
        'B_LUMO_diff', 'B_LUMO_sum', 'B_X_diff', 'B_X_sum', 'B_Z_radii_diff', 'B_Z_radii_sum', 'B_e_affin_diff', 'B_e_affin_sum',
        'tau', 'mu','mu_A_bar','mu_B_bar']
    feature_labels = generate_feature_labels(feature_list)
    features, labels = create_training_data(stability_training_data, feature_list,target_label='stable',model_type='RFC')
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 12)
    clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators =100, max_depth =23 , max_features='auto',min_samples_split = 2, min_samples_leaf =1, class_weight='balanced')
    y_score = clf.fit(train_features, train_labels)
    # Classifier performance
    pred = clf.predict(test_features)
    RFC_performance(clf,test_labels,pred,train_features,train_labels,labels,features,c_type='Perovskite stability classification')
    # Feature importances
    plot_feature_importance(clf, feature_labels, filename='stability_feature_importance.png', n_features=20, palette="Greens_d", dir_name='stability_results')        
    # ROC curve
    if 'analyze' in sys.argv:
        dst = 'stability_results'
        plot_roc_curves(test_features,test_labels, clf,f_name=dst)  
    return feature_list, clf
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_insulator_classification(stability_training_data):
    cwd = os.getcwd()
    feature_list = ['A_HOMO_diff', 'A_HOMO_sum', 'A_IE_sum', 'A_LUMO_diff', 'A_LUMO_sum', 'A_X_diff', 'A_X_sum',
        'A_Z_radii_sum',  'A_e_affin_sum', 'B_HOMO_diff', 'B_HOMO_sum', 'B_IE_diff', 'B_IE_sum',
        'B_LUMO_diff', 'B_LUMO_sum', 'B_X_diff', 'B_X_sum', 'B_Z_radii_diff', 'B_Z_radii_sum', 'B_e_affin_diff', 'B_e_affin_sum',
        'tau', 'mu','mu_B_bar']
    feature_labels = generate_feature_labels(feature_list)
    features, labels = create_training_data(stability_training_data, feature_list,target_label='Insulator',model_type='RFC')
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 12, shuffle=True)
    clf = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators = 200,  max_depth = 25 , max_features= 'auto', min_samples_split = 5, min_samples_leaf =1) 
    # Classifier performance
    y_score = clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    RFC_performance(clf,test_labels,pred,train_features,train_labels,labels,features,c_type='Insulator classification')
    # Feature importances
    plot_feature_importance(clf, feature_labels,filename='insulator_feature_importance.png', n_features=20, palette="YlGnBu_r", dir_name='insulator_results')      
    # ROC curve
    #dst = 'insulator_results'
    #plot_roc_curves(test_features,test_labels, clf,f_name=dst)
    return feature_list,  test_features,test_labels, clf
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_bandgap_regression(bandgap_training_data):
    cwd = os.getcwd()
    feature_list = ['A_HOMO_diff', 'A_HOMO_sum', 'A_IE_sum', 'A_LUMO_diff', 'A_LUMO_sum', 'A_X_diff', 'A_X_sum',
        'A_Z_radii_sum', 'A_e_affin_sum', 'B_HOMO_diff', 'B_HOMO_sum', 'B_IE_diff', 'B_IE_sum',
        'B_LUMO_diff', 'B_LUMO_sum', 'B_X_diff', 'B_X_sum', 'B_Z_radii_diff', 'B_Z_radii_sum', 'B_e_affin_diff', 'B_e_affin_sum',
        'tau', 'mu', 'mu_B_bar']
    feature_labels = generate_feature_labels(feature_list)
    features, labels = create_training_data(bandgap_training_data, feature_list,target_label='PBE_band_gap',model_type='RFR')
    # run regressor
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 12)
    train_labels = np.ravel(train_labels)
    best_params = {"n_estimators":2000,
            "max_depth":52,"random_state":12,"min_samples_leaf":1,"min_samples_split":2, "max_features":'auto',"bootstrap":'True'}
    best_estimator = fit_rfr(best_params,train_features,train_labels, feature_labels)
    #Feature importance
    ranked_features = plot_feature_importance(best_estimator, feature_labels, filename='bg_regression_feature_importance.png', n_features=20, palette="Greens_d", dir_name='bg_regression_results') 
    mse_test, mse_train, R_square_test, R_square_train, R2_test, predictions,pred = evaluate(best_estimator, test_features, test_labels, train_features, train_labels, feature_list, r_type='Band gap regression')
    plot_data = []
    # for i in range(len(train_labels)):
    #     dat = dict()
    #     dat['calc'] = train_labels[i]
    #     dat['pred'] = pred[i]
    #     dat['datatype'] = 'train'
    #     dat['MAE'] = np.absolute(pred[i]-train_labels[i])
    #     plot_data.append(dat)
    # for i in range(len(test_labels)):
    #     dat = dict()
    #     dat['calc'] = test_labels[i]
    #     dat['pred'] = predictions[i]
    #     dat['datatype'] = 'test'
    #     dat['MAE'] = np.absolute(predictions[i]-test_labels[i])
    #     plot_data.append(dat)
    # with open('regression_plot_data.json', 'w') as outfile: 
    #     json.dump(plot_data, outfile, sort_keys=True,indent = 4)
    plot_regression(train_labels, pred, test_labels, predictions, figname='bandgap_regression.png')   
    plot_confidence_intervals(best_estimator,train_features,test_features, test_labels, predictions, figname='regression_confidence',dir_name='bg_regression_results')
    	# err_down, err_up = pred_ints(best_estimator, test_features, percentile=90)
    	# correct = 0.
    	# for i, val in enumerate(test_labels):
    	# 	if err_down[i] <= val <= err_up[i]:
     #    		correct += 1
    	# print(correct/len(train_labels))

    if 'analyze' in sys.argv:
        plot_learning_curve(best_estimator,  train_features, train_labels, cv=5, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))

    if 'pdp' in sys.argv:
        importances = list(best_estimator.feature_importances_)
        feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        ranked_features = []
        for i in range(5):
            ranked_features.append(feature_importances[i][0])	
        print(ranked_features)
        dataset = pd.read_json('datasets/bandgap_training_data.json')
        ranked_labels = generate_feature_labels(ranked_features)
        plot_pdp_plots(best_estimator,dataset,feature_list,ranked_features,ranked_labels)
    return feature_list, best_estimator
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def create_training_data(training_raw_data, feature_list,target_label,model_type=None):
    all_data = pd.DataFrame(training_raw_data)
    features = all_data[feature_list]
    labels = np.array(all_data[target_label])
    if model_type == 'RFC':
        labels = labels.astype('int')
    return features, labels
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
