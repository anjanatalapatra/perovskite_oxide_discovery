import pickle
import pandas as pd
from .data_utilities.generate_data import prepare_data
from .ML_models.ML_models import run_perovskite_formability, run_perovskite_stability, run_insulator_classification, run_bandgap_regression
from .ML_utilities.RFC_functions import classify_data
from .ML_utilities.RFR_functions import predict_regression,predict_regression_arbitary
import sys, os, shutil
import json
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_perovskite_formability_training_data(training_compounds, ele_data):
	perovskite_training_compounds = [ i for i in training_compounds if 'Perovskite' in i.keys()]
	perovskite_training_data = prepare_data(perovskite_training_compounds,ele_data)
	return perovskite_training_data

def create_perovskite_stability_training_data(training_compounds, ele_data):
	stability_training_compounds = [ i for i in training_compounds if 'PBE_band_gap' in i.keys()]
	stability_training_data = prepare_data(stability_training_compounds,ele_data)
	return stability_training_data

def create_bandgap_regression_training_data(training_compounds, ele_data):
	stability_training_data = create_perovskite_stability_training_data(training_compounds, ele_data)
	bandgap_training_data = [ i for i in stability_training_data if float(i['Insulator']) == 1]
	save_predicted_data(bandgap_training_data,dir_name='datasets', fname = 'bandgap_training_data.pkl')
	return bandgap_training_data

def create_candidate_data_for_prediction(all_candidate_data,ele_data):
	all_candidate_data = prepare_data(all_candidate_data,ele_data)
	return all_candidate_data 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def save_predicted_data(predicted_data, dir_name=None, fname=None):
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
	fname = dir_name + '/'+ fname
	with open(fname, 'wb') as f:
		pickle.dump(predicted_data, f)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def predict_using_ML_models(all_candidate_data, formability_feature_list, formability_clf,stability_feature_list, stability_clf,insulator_feature_list, insulator_clf,regression_feature_list, best_estimator,data_type=None, dir_name=None):
	formable_data = classify_data(all_candidate_data, formability_feature_list, formability_clf, pred_label='predicted_Perovskite',data_type=data_type, model_type='formability')
	stable_formable_data = classify_data(formable_data, stability_feature_list, stability_clf, pred_label='predicted_stable', data_type=data_type, model_type='formability_stability')
	stable_formable_insulator_data = classify_data(stable_formable_data, insulator_feature_list, insulator_clf, pred_label='predicted_Insulator', data_type=data_type, model_type='insulator')
	wide_bandgap_data = predict_regression(stable_formable_insulator_data, regression_feature_list, best_estimator, pred_label='Predicted_band_gap')
	save_predicted_data(wide_bandgap_data,dir_name, fname = 'wide_band_gap_candidates.pkl')
	if 'saveall' in sys.argv:
		save_predicted_data(formable_data,dir_name='predicted_candidates', fname = 'formable_candidates.pkl')
		save_predicted_data(stable_formable_data,dir_name='predicted_candidates', fname = 'stable_formable_candidates.pkl')
		save_predicted_data(stable_formable_insulator_data,dir_name='predicted_candidates', fname = 'stable_formable_insulator_candidates.pkl')
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def predict_arbitary_regression(candidate_data, feature_list, best_estimator, pred_label='Predicted_band_gap'):
    bandgap_data = predict_regression_arbitary(candidate_data, feature_list, best_estimator, pred_label='Predicted_band_gap')
    save_predicted_data(bandgap_data,dir_name='predicted_candidates', fname = 'test_band_gap_candidates.pkl')   

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def create_core_energy_dataset(training_compounds, ele_data):
	stability_training_data = create_perovskite_stability_training_data(training_compounds, ele_data)
	core_energy_dataset = prepare_data(stability_training_data, ele_data)
	core_energy_dataset = pd.DataFrame(core_energy_dataset)
	core_energy_dataset = core_energy_dataset[core_energy_dataset.O_cl_eigenenergy.notnull()]
	return core_energy_dataset
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def run_all_models(training_compounds, ele_data):
	perovskite_training_data = create_perovskite_formability_training_data(training_compounds, ele_data)
	formability_feature_list, formability_clf = run_perovskite_formability(perovskite_training_data) 
	stability_training_data = create_perovskite_stability_training_data(training_compounds, ele_data)
	stability_feature_list, stability_clf = run_perovskite_stability(stability_training_data)
	insulator_training_data = stability_training_data
	insulator_feature_list, insulator_clf = run_insulator_classification(insulator_training_data)
	bandgap_training_data = create_bandgap_regression_training_data(training_compounds, ele_data)
	regression_feature_list, best_estimator = run_bandgap_regression(bandgap_training_data)
	return formability_feature_list, formability_clf, stability_feature_list, stability_clf, insulator_feature_list, insulator_clf, regression_feature_list, best_estimator
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


if __name__ == "__main__": main()
