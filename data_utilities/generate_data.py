import numpy as np
from math import sqrt
import pickle
import unicodecsv

def get_elements(compound):
    A1 = compound['A1']
    A2 = compound['A2']
    B1 = compound['B1']
    B2 = compound['B2']
    ctype = compound['type']
    ele_list = [A1,A2,B1,B2]
    return ele_list, ctype

def get_compound_ele_data(compound,ele_data):
    A1, A2, B1, B2, ctype = get_elements(compound)
    A1_data = ele_data[A1]
    A2_data = ele_data[A2]
    B1_data = ele_data[B1] 
    B2_data = ele_data[B2]
    return A1_data, A2_data, B1_data, B2_data

def populate_features(compound,feature_list,feature_value_list):
    for i in range(len(feature_list)):
        compound[feature_list[i]] = feature_value_list[i]
    return compound 
    
def generate_features(compound,ele_data,feature_list=None):
    if feature_list == None:
        feature_list = ['HOMO','LUMO','IE','X','e_affin','Z_radii']
    else:
        feature_list = feature_list
    site_list = ['A1','A2','B1','B2']
    f_name_list = [i+'_'+ j for i in site_list for j in feature_list]
    ele_list, ctype = get_elements(compound)
    
    f_list = [ele_data[i][j] for i in ele_list for j in feature_list]
    new_compound = populate_features(compound,f_name_list,f_list)

    OS_list = [str(new_compound[i+'_OS']) for i in site_list]
    CN_list = ['12','12','6','6']
    IR_name_list = [i+'_IR' for i in site_list]
    IR_list = [ele_data[i]['shannon_IR'][j][k] for i,j,k in zip(ele_list,OS_list,CN_list)]
    IR_data = populate_features(new_compound, IR_name_list, IR_list)
    structure_factors_list = calculate_structure_factors(IR_list)
    structure_factors = ['tau','mu','mu_A_bar','mu_B_bar']
    structure_data = populate_features(IR_data, structure_factors, structure_factors_list)
    return IR_data

def calculate_structure_factors(IR_list):
    comp_list = [0.5,0.5,0.5,0.5]
    weighted_IR = [IR_list[i]*comp_list[i] for i in range(len(IR_list))]
    radii_list = [weighted_IR[0]+weighted_IR[1],weighted_IR[2]+weighted_IR[3]]
    tau = round(float(radii_list[0]+ 1.4)/(sqrt(2)*(radii_list[1]+1.4)),5)
    mu = round(radii_list[1]/1.4,5)
    mu_A_bar = round(abs(weighted_IR[1]-weighted_IR[0])/1.4,5)
    mu_B_bar = round(abs(weighted_IR[3]-weighted_IR[2])/1.4,5)
    structure_factors = [tau, mu,mu_A_bar,mu_B_bar]
    return structure_factors

def generate_compound_features(new_compound,feature_list=None):
    if feature_list == None:
        feature_list = ['HOMO','LUMO','IE','X','e_affin','Z_radii']
    else:
        feature_list = feature_list
    c_sites = ['A','B']
    for i in c_sites:
        for j in feature_list:
            cf_sum = i+'_'+j+'_sum'
            new_compound[cf_sum] = (float(new_compound[i+'1_'+j])+ float(new_compound[i+'2_'+j]))/2.
            cf_diff = i+'_'+j+'_diff'
            new_compound[cf_diff] = np.absolute(float(new_compound[i+'1_'+j]) - float(new_compound[i+'2_'+j]))/2.
    return new_compound

def prepare_data(data,ele_data):#,fname):
    new_data = []
    for i in data:
        new_compound = generate_features(i,ele_data,feature_list=None)
        new_compound = generate_compound_features(new_compound,feature_list=None)
        new_data.append(new_compound)
    return new_data

def generate_feature_labels(feature_list):
    f_list = ['A_HOMO_diff','A_HOMO_sum','A_IE_diff','A_IE_sum','A_LUMO_diff','A_LUMO_sum','A_X_diff','A_X_sum','A_Z_radii_diff','A_Z_radii_sum','A_e_affin_diff', \
        'A_e_affin_sum','B_HOMO_diff','B_HOMO_sum','B_IE_diff','B_IE_sum','B_LUMO_diff','B_LUMO_sum','B_X_diff','B_X_sum','B_Z_radii_diff','B_Z_radii_sum','B_e_affin_diff', \
        'B_e_affin_sum','mu','mu_A_bar','mu_B_bar','tau']
    label_list = ['HOMO$^{A-}$', 'HOMO$^{A+}$', 'Ionization energy$^{A-}$ ', 'Ionization energy$^{A+}$ ', 'LUMO$^{A-}$', 'LUMO$^{A+}$', 'X$^{A-}$', 'X$^{A+}$', \
        'Z radius$^{A-}$', 'Z radius$^{A+}$', 'Electron affinity$^{A-}$', 'Electron affinity$^{A+}$', 'HOMO$^{B-}$', 'HOMO$^{B+}$', 'Ionization energy$^{B-}$ ', \
        'Ionization energy$^{B+}$ ', 'LUMO$^{B-}$', 'LUMO$^{B+}$', 'X$^{B-}$', 'X$^{B+}$', 'Z radius$^{A-}$', 'Z radius$^{A+}$', 'Electron affinity$^{B-}$', \
        'Electron affinity$^{B+}$', 'Octahedral factor ($\mu$)','Mismatch factor $\\bar{\\mu}_A$', 'Mismatch factor $\\bar{\\mu}_B$', 'Tolerance Factor ($t$)']
    fn_list = [f_list.index(i) for i in feature_list]
    feature_labels = [label_list[i] for i in fn_list] 
    return feature_labels    

def csv_to_json(csvfile):
    reader = unicodecsv.DictReader(open(csvfile,'rb'),encoding='utf-8-sig')
    dict_list = []
    for line in reader:
         dict_list.append(line)
    return dict_list