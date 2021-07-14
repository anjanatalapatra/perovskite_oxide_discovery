import pandas as pd
import json
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn import preprocessing

def classify_data(candidate_data, feature_list, clf, pred_label, data_type, model_type):
    print("Classify candidates")
    print("Number of compounds:", len(candidate_data))
    candidate_data = pd.DataFrame(candidate_data)
    features = candidate_data[feature_list]
    test_prediction = clf.predict(features) 
    p_list = clf.predict_proba(features)[:,1]
    candidate_data[pred_label] = test_prediction
    extra_keys = ['A1','A2','B1','B2','functional_group','type','A1_OS','A2_OS','B1_OS','B2_OS']
    if model_type == 'formability':
        candidate_data['p_f_value'] = p_list
        keep_col = ['p_f_value',pred_label] + feature_list + extra_keys
    elif model_type == 'stability':
        candidate_data['p_s_value'] = p_list
        keep_col = ['p_s_value', pred_label] + feature_list + extra_keys
    elif model_type == 'formability_stability':
        candidate_data['p_s_value'] = p_list
        keep_col = ['p_s_value','p_f_value',pred_label, 'predicted_Perovskite'] + feature_list + extra_keys
    elif model_type == 'insulator':
        candidate_data['p_i_value'] = p_list
        #keep_col = ['p_s_value','p_f_value','p_i_value','predicted_Perovskite','predicted_stable',pred_label] + feature_list + extra_keys
        keep_col = [pred_label] + feature_list + extra_keys     
    new_data = candidate_data[keep_col]
    if not data_type =='training':
    	classified_data = new_data[new_data[pred_label] ==1]
    else:
        classified_data = new_data
    print("Number of classified True compounds:", len(classified_data))
    return classified_data


def RFC_performance(clf,test_labels,pred,train_features,train_labels,labels,features,c_type=None):
    print("------------------------------------------------------------------------")
    print(c_type,"Model Performance")
    print("------------------------------------------------------------------------")
    test_precision = precision_score(test_labels, pred)
    test_recall = recall_score(test_labels, pred)
    test_accuracy = accuracy_score(test_labels, pred)
    print("RFC Test Precision:", test_precision)
    print("RFC Test Recall:", test_recall)
    print("RFC Test Accuracy:", test_accuracy)
    test_confusion_matrix = confusion_matrix(test_labels, pred)
    train_pred = clf.predict(train_features)
    train_confusion_matrix = confusion_matrix(train_labels, train_pred)
    print("Test confusion matrix")
    print(test_confusion_matrix)
    print("Train confusion matrix")
    print(train_confusion_matrix)
    all_data_pred = clf.predict(features)
    all_confusion_matrix = confusion_matrix(labels, all_data_pred)
    print(" All data confusion matrix")
    print(all_confusion_matrix)
    print("------------------------------------------------------------------------")
