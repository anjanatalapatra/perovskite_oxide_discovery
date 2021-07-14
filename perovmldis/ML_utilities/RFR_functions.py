import sys
import json
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc, savefig, rcParams
from scipy.stats import gaussian_kde
from sklearn.model_selection import learning_curve
import forestci as fci
import seaborn as sns

def evaluate(model, test_features, test_labels, train_features, train_labels, feature_labels,r_type=None):
    predictions = model.predict(test_features)
    mse_test = mean_squared_error(test_labels,predictions,squared=False)
    pred = model.predict(train_features)
    mse_train = mean_squared_error(train_labels,pred)
    mae_train = mean_absolute_error(train_labels,pred)
    R_square_test = model.score(test_features, test_labels)
    R_square_train = model.score(train_features, train_labels)
    R2_test = r2_score(test_labels,predictions)
    MSE_test = mean_squared_error(test_labels,predictions,squared=False)
    MAE_test = mean_absolute_error(test_labels,predictions)
    importances = list(model.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_labels, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    print("------------------------------------------------------------------------")
    print(r_type, 'Model Performance')
    print("------------------------------------------------------------------------")
    print(r_type, 'Training')
    print('MSE:', round(mse_train, 4))
    print('MAE:',round(mae_train,4))
    print('R2_coefficient:',round(R_square_train,4))
    print("------------------------------------------------------------------------")
    print(r_type, 'Test')
    print('MSE:', round(mse_test, 4))
    print('MAE:',round(MAE_test,4))
    print('R2_coefficient:',round(R_square_test,4))
    print("------------------------------------------------------------------------")   
    return mse_test, mse_train, R_square_test, R_square_train, R2_test, predictions, pred

def plot_regression(train_labels, pred, test_labels, predictions, figname):
    #rcParams['figure.figsize'] = 6, 6
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    all_labels = list(test_labels) + list(train_labels)
    all_pred = list(predictions) + list(pred)
    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)
    xmin = test_labels.min()
    xmax = test_labels.max()
    ymin = predictions.min()
    ymax = predictions.max()
    #ax.axis([xmin, xmax, ymin, ymax])
    sns.scatterplot(train_labels, pred, s=30, color='green', ax=ax[0])
    ax[0].set_title('Training',fontsize=12)
    ax[0].plot([0,4.5],[0,4.5],'k--')
    sns.scatterplot(test_labels, predictions, color='red',ax=ax[1])
    ax[1].set_title('Test',fontsize=12)
    r2 = np.round(r2_score(test_labels,predictions), 3)
    ax[0].set_xlim(0,4.5)
    ax[0].set_ylim(0,4.5)
    ax[1].set_xlim(0,3.75)
    ax[1].set_ylim(0,3.75)
    ax[1].plot([0,3.75],[0,3.75],'k--')
    plt.tight_layout()
    #savefig(figname, dpi=300,format='png')

def plot_fancy_train_regression(train_labels, pred):
    MAE = np.round(mean_absolute_error(train_labels,pred),2)
    MSE = np.round(mean_squared_error(train_labels, pred),2)
    r2 = np.round(r2_score(train_labels, pred),2)
    fig, ax = plt.subplots(figsize=(3.5,3))
    plt.grid(True)
    train_xy = np.vstack([train_labels, pred])
    w = gaussian_kde(train_xy)(train_xy)
    idx = w.argsort()
    x, y, w = train_labels[idx], pred[idx], w[idx]
    p = ax.scatter(x,y,c=w,s=3,edgecolor='', cmap='plasma')
    plt.xlabel('Calculated Bandgap (eV)')
    plt.ylabel('Predicted Bandgap (eV)')
    plt.tight_layout()
    r2 = np.round(r2_score(train_labels,pred),3)
    z = np.linspace(0,4.5,20)
    plt.plot(z, z, label="$r^2=$" + str(r2), c="k", linestyle='--',linewidth=0.2)
    plt.xlim(0,4.5)
    plt.ylim(0,4.5)
    t = 'MAE = ' + str(MAE) +' eV' +'\n' + 'MSE = ' + str(MSE) +' eV' +'\n' + r'R$^2$ coefficient = '+ str(r2)
    plt.text(0.25, 3.3, t, bbox=dict(facecolor='bisque', alpha=0.5), fontsize=8, wrap=True)
    plt.savefig('training_regression', dpi=500, fmt='eps')

def fit_rfr(params,train_features, train_labels,feature_labels):
    rfr = RandomForestRegressor()
    rfr.set_params(**params)
    train_labels = np.ravel(train_labels)
    rfr.fit(train_features, train_labels)
    return rfr

def predict_regression(prediction_data, feature_list, best_estimator, pred_label):
    test_data = pd.DataFrame(prediction_data)
    print("Number of compounds:", len(test_data))
    extra_keys = ['A1','A2','B1','B2','functional_group','type','A1_OS','A2_OS','B1_OS','B2_OS']

    features = test_data[feature_list]
    predictions = best_estimator.predict(features)
    #keep_col = ['predicted_Insulator','p_i_value'] + feature_list + [pred_label] + extra_keys
    keep_col = feature_list + [pred_label] + extra_keys
    test_data[pred_label] = predictions
    sorted_data = test_data.sort_values(by=[pred_label],ascending=False)
    new_data = sorted_data[keep_col] 
    regression_results = new_data[new_data[pred_label] >=0.5]
    print("Number of wide bandgap compounds:", len(regression_results))
    return regression_results


def predict_regression_arbitary(prediction_data, feature_list, best_estimator, pred_label):
    test_data = pd.DataFrame(prediction_data)
    print("Number of compounds:", len(test_data))
    extra_keys = ['A1','A2','B1','B2','functional_group','type','A1_OS','A2_OS','B1_OS','B2_OS','PBE_band_gap','HSE_band_gap']
    if len(test_data) > 0:
        features = test_data[feature_list]
        predictions = best_estimator.predict(features)
        keep_col = feature_list + [pred_label] + extra_keys
        test_data[pred_label] = predictions
        sorted_data = test_data.sort_values(by=[pred_label],ascending=False)
        new_data = sorted_data[keep_col]
        regression_results = new_data
        print("Number of wide bandgap compounds:", len(regression_results))
    return regression_results


def plot_learning_curve(estimator,  features, labels, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    rc('font',**{'family':'serif','serif':['Helevtica']})
    #rc('axes', axisbelow=True)
    rcParams['figure.figsize'] = 4,3
    scoring_matrix = ['r2','neg_mean_absolute_error']
    label_matrix = [r'R$^2$','Mean Absolute error (eV)']
    scoring_matrix_2 = ['neg_mean_squared_error','neg_mean_squared_log_error']
    label_matrix_2 = ['Root Mean Square Error','Root Mean Square Log Error']
    for i in range(len(scoring_matrix)):
        plt.figure(figsize=(4,3))
        plt.xlabel("Training examples",fontsize=10)
        print(len(labels))
        train_sizes, train_scores, test_scores = learning_curve(estimator,features,labels, cv=cv, n_jobs=n_jobs,
                scoring=scoring_matrix[i],train_sizes=train_sizes)
        print(train_sizes)
        test_scores = np.sqrt(np.absolute(test_scores))
        train_scores = np.sqrt(np.absolute(train_scores))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        print(train_scores_std)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        print(test_scores_std)
        #plt.grid(True)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, '--', color="r",
                 linewidth=1,markersize=5, label="Training score")
        plt.plot(train_sizes, test_scores_mean, '--', color="g",
               linewidth=1,markersize=5, label="Test score")

        plt.legend(loc="best")
        plt.ylabel(label_matrix[i],fontsize=10)
        #plt.ylabel('Mean absolute error',fontsize=12)
        figname = 'bg_regression_results/Learning_curve_RFR_M0'+ '_'+ scoring_matrix[i]
        plt.tight_layout()
        plt.xticks(fontsize=12)
        savefig(figname, dpi=300,format='pdf')

def plot_confidence_intervals(best_estimator, train_features, test_features, test_labels, predictions, figname, dir_name):
    rc('text', usetex=False)
    bg_unbiased_test = fci.random_forest_error(best_estimator, train_features, test_features)
    fig, ax = plt.subplots(figsize=(6,6))
    #fig.patch.set_facecolor('white')
    #plt.axes(frameon=True)
    #ax.set_facecolor("white")
    ax.grid(linestyle='--', linewidth='0.5', color='lightgray')
    ax.grid(b=True)
    plt.errorbar(test_labels,predictions,yerr=np.sqrt(bg_unbiased_test), fmt='o',markersize=6, color='red',
                    ecolor='darkgray', elinewidth=1, capsize=2)
    plt.tick_params(labelsize=8)
    plt.plot([0,4.5], [0, 4.5], 'k--',linewidth=1)
    plt.xlim(0,4.5)
    plt.ylim(0,4.5)
    plt.title('Confidence Intervals')
    plt.xlabel('Calculated band gap (eV)',fontsize=12)
    plt.ylabel('Predicted band gap (eV)',fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.yaxis.set_ticks((0,1,2,3,4))
    plt.tight_layout()
    # savefig(figname, dpi=600, facecolor=fig.get_facecolor(), format='png')
    # cwd = os.getcwd()
    # if not os.path.exists(dir_name):
    #         os.mkdir(dir_name)
    # dst = cwd + '/' + dir_name
    # shutil.move(os.path.join(cwd,figname),os.path.join(dst,figname))

def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            #print(x)
            #print(X)
            preds.append(pred.predict([X.iloc[x]]))
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up
