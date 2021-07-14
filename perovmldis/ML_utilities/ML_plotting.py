import matplotlib.pyplot as plt
from matplotlib.pyplot import annotate, savefig, rcParams, rc, tick_params, xlabel, ylabel, xlim, ylim, legend, title
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from .performance_functions import BinaryClassification
from sklearn.tree import export_graphviz
from subprocess import call
import os, shutil
import seaborn as sns
from itertools import combinations
import numpy as np
from pdpbox import pdp

def plot_feature_importance(clf, feature_labels=None, filename=None, n_features=None,palette=None, dir_name=None):
    cwd = os.getcwd()
    importances = list(clf.feature_importances_)
    feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_labels, importances)] 
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True) 
    x_values = list(range(n_features))
    xvalues = [x+1 for x in x_values]
    importances =[]
    feature_list = []
    for i in range(n_features):
        importances.append(feature_importances[i][1])
        feature_list.append(feature_importances[i][0])
    rc('font',**{'family':'serif','serif':['Helvetica']})
    fig, ax = plt.subplots(figsize=(4,5))
    sns.barplot(y=x_values, x= importances,palette=palette,orient='h')
    ax.grid(True,color='grey', linestyle='--', linewidth=0.5)
    plt.yticks(x_values, feature_list,fontsize=10)#, rotation='vertical')
    plt.xlabel('Importance',fontsize=14);
    plt.title('Feature Importance',fontsize=14)
    #plt.tight_layout()
    #plt.savefig(filename, dpi=300,format='png')
    #plt.show()
    #plot_feature_importance(feature_importances,filename,n_features,palette=palette)
    #if not os.path.exists(dir_name):
    #        os.mkdir(dir_name)
    #dst = cwd + '/' + dir_name
    #shutil.move(os.path.join(cwd,filename),os.path.join(dst,filename))      
    return feature_list

def plot_roc_curves(test_features,test_labels, clf):
    labels=clf.classes_
    y_test = label_binarize(test_labels, classes = [0,1])
    ypred = clf.predict_proba(test_features)[:,1]
    bc = BinaryClassification(y_test, ypred, labels)
    plt.figure(figsize=(4,4))
    bc.plot_roc_curve()
    plt.figure(figsize=(4,4))   
    bc.plot_precision_recall_curve()
    #fname = f_name+'/precision_recall_curve'
    #plt.savefig(fname,dpi=600,fmt='png')
    #plt.show()
    bc.print_report() 

def plot_roc(fpr,tpr):
    rc('font',**{'family':'serif','serif':['Helvetica']})
    #rc('text', usetex=True)
    rc('axes', axisbelow=True)
    rcParams['figure.figsize'] = 6,4
    plt.figure()
    roc_auc = auc(fpr,tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area= %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('ROC.pdf', dpi=300,format='pdf')
  
def plot_pdp_plots(model,dataset,feature_list,ranked_features,ranked_labels):
    feature_pairs = np.array(list(combinations(ranked_features,2)))
    feature_labels = np.array(list(combinations(ranked_labels,2)))
    for i in range(len(feature_pairs)):
        inter1 = pdp.pdp_interact(model=model,dataset=dataset,model_features=feature_list,features=feature_pairs[i],num_grid_points=[10, 10],percentile_ranges=[(5, 95), (5, 95)])
        fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1,feature_names=feature_labels[i],plot_type='contour',x_quantile=False,plot_pdp=False)
        fname = feature_pairs[i][0]+'_'+feature_pairs[i][1] +'.pdf'
        plt.tight_layout()
        #plt.savefig(fname, dpi=500,format='pdf')
