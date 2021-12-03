# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import train_test_split
from  multiprocessing import cpu_count
from utils_plot import rmse, plot_comparisons, plot_feature_importances
import predictor
import optimizer
from itertools import combinations
from copy import deepcopy
import csv

# %matplotlib inline

seed = 100
np.random.seed(seed)

"""# 1. Cross-Validation Tuning and Train

## Import data and preprocess

This dataset contains all the literature data + 87 experimental datapoints
"""
with open('./impute_NCM_MICE.csv', 'r') as file:
    data = pd.read_csv(file)

"""## CV Tuning with XGBoost"""

inputs = np.asarray(data.loc[:,'Ni':'total'])
labels = np.asarray(data['Dis_cap'])
"""## CV Tuning with XGBoost"""

scaler_regressor = StandardScaler()
inputs = scaler_regressor.fit_transform(inputs)

inputs_train, inputs_valid, labels_train, labels_valid = train_test_split(
    inputs, labels, test_size=0.1, random_state=seed)

XGB_Options = { 
                'cv':               10,    ##################################chi hao -9april2020 change to 10
                'scoring':          'neg_mean_squared_error',
                'seed':             seed, 
                'max_depth':        np.arange(2,14,2),
                'min_child_weight': np.arange(1,8,1),
                'n_estimators':     np.arange(10,80,5),
                'gamma':            np.arange(0.05,0.45,0.05),
                'colsample_bytree': np.arange(0.60, 0.95, 0.05),
                'subsample':        np.arange(0.60, 0.95, 0.05),
                'reg_alpha':        [1e-5, 1e-2, 0.1, 0.5, 1, 5, 10], #alpha 
                'reg_lambda':       [1e-5, 1e-2, 0.1, 0.5, 1, 5, 10],#lambda
                'learning_rate':    np.arange(0.025,0.150,0.025),
                'scaler':           scaler_regressor,
                'n_jobs':           -1,
                'verbose':          1             
              }

trained_regressor =  predictor.XGB_Regressor(options=XGB_Options) #########################
trained_regressor.fit(inputs_train, labels_train)
tuned_regressor = trained_regressor.regressor

predicted_train_ = trained_regressor.predict(inputs_train)   #this part called the regressor from trained xgb regressor
from itertools import zip_longest
d = [labels_train, predicted_train_] 
act_pred_ = zip_longest(*d, fillvalue = '')
with open("./Train_MICE_2.csv", "w", newline='') as file2:
    writer = csv.writer(file2)
    writer.writerows(act_pred_)
    
predicted_valid_ = trained_regressor.predict(inputs_valid) 
dl = [labels_valid, predicted_valid_] 
act_valid_ = zip_longest(*dl, fillvalue = '')
with open("./Valid_MICE_2.csv", "w", newline='') as file3:
    writer = csv.writer(file3)
    writer.writerows(act_valid_)

print('passsss1')
"""### Best Parameters"""

for key, value in tuned_regressor.get_params().items():    #chi hao: this is to use all the tuned parameters
    print('{}: {}'.format(key, value))


"""# 2. Train NN as Secondary Validation"""

import tensorflow as tf  #original just use tensorflow -chi hao
#tf.disable_v2_behavior()

tf.get_default_graph()  #tf.reset_default_graph() chi hao
np.random.seed(seed)
tf.set_random_seed(seed)   #tf.set_random_seed(seed)-original chi hao
sess = tf.Session()  #tf.Session() -original chi hao
 
NUM_NN = 10  ##################################chi hao -9april2020 change to 10
regressors = []

for n in range(NUM_NN):
    options = {'input_shape': [11, ], 'target_shape': [1, ],
               'n_layers': 2, 'layer_widths': [64, 128], 
               'name': 'nn_{}'.format(n), 'session': sess}
    regressors.append(predictor.NeuralNetRegressor(options=options))

print('@2++++++++++++regressor++++++++++++++\n', regressors)   #chi hao: telling the type of regressors
print('@@@@@@@@3 scaler regressor:', scaler_regressor)
ensb_regressor = predictor.EnsembleRegressor(regressors, scaler_regressor)

options={'monitor': False,
         'reinitialize': True,
         'n_iter': 10000}
list_of_options = [options,]*NUM_NN
ensb_regressor.fit(inputs_train, labels_train.reshape(-1, 1), list_of_options)

predicted_train = ensb_regressor.predict_mean(inputs_train)
predicted_validation = ensb_regressor.predict_mean(inputs_valid)

train_rmse = rmse(predicted_train, labels_train)
validation_rmse = rmse(predicted_validation, labels_valid)    #using ensemble nn for prediction and compare the accuracy

print('Train RMSE: ', train_rmse, 'validation RMSE: ', validation_rmse)

"""### Train all data"""

ensb_regressor.fit(inputs, labels.reshape(-1,1), list_of_options)
predicted_nn = ensb_regressor.predict_mean(inputs)

#print('@@@@@@@@@@@@@ 4 input:', inputs)
std_nn = np.sqrt(ensb_regressor.predict_covariance(inputs))
rmse_nn = rmse(predicted_nn, labels)

"""# 3. Inverse Design

## Train model with tuned parameters on the full dataset
"""
print('@@@@@@@@@@ 5 trained regressor :', trained_regressor)  # trained regressor = predictor.XGB_Regressor object at 0x7fbafd359e90
trained_regressor.regressor.fit(inputs, labels)
predicted = trained_regressor.predict(inputs)   #this part called the regressor from trained xgb regressor
print('All RMSE: ', rmse(predicted, labels))
#combine actual and prediction data then save into file
#print(y_predict)
#print(Y_train)
  

    
#print('@@@@@@@@@@ 6 regressor type and input:', regressor) #  this regressor is not predefined

"""## Constraint and Target Specifications

We solve the problem
$$
    \min_{x} \mathrm{objective}(x, CP)
$$
subject to
$$
    \mathrm{constraints}(x) \geq 0
$$

Some soft constraints are written into the objective function as a regularizer, and some hard constraints are moved to the selection criterion function, which must evaluates to true before we consider the solution admissiable. This is to maximize efficiency of the particle swarm method.
"""

lb = np.array([0.3, 0, 0, 0.5, 450, 1, 650, 5, 4, 0.05, 0.99]) #minimum condition
ub = np.array([1, 1, 1, 20, 650, 12, 1000, 24, 4.9, 0.2, 1.02])  #maximum condition

def objective(x, target_CP, regressor):
    '''
    loss function:
        The loss function contains 2 parts
            1. 0.5*(predicted_CP(x) - target_CP)**2
            2. coef * regularizer
    '''
    
    def regularizer(x):
        '''
        Regularizer (or soft constraints):
            We want to drive the system towards
            binary systems involving A,B,C,D
            Here, we use the simple penalizer
                \sum_{(i,j,k) \in (A,B,C,D)} i*j*k

            For task 1:
                We also want to minimize A
        '''
        reg = 0.0
        for c in combinations(x[:3], 3):  #composition restriction
            #print("#######>>", x)
            reg += np.prod(c)**(1.0/3.0)

        reg += x[1]  # minimize Co 
        return reg

    
    return 0.5 * (regressor.predict_transform(x) - target_CP)**2 \
        + 1e-1 * regularizer(x)


def constraints(x, *args):
    ''' hard constraints '''
    Final_M    = np.sum(x[:3]) #  constraint compositions, summing the 1st 3 components 
    #print('>>> x ', x[:3])
    #print('>>>FInal ', Final_M)
    cons_Mp_ub = [Final_M*1.02 - x[10]]   #restriction of the summation of those components
    cons_Mp_lb = [x[10] - Final_M*0.99]
    #print('###final:',Final_M)
    #print('###up:', cons_Mp_ub)  
    #print('###low:', cons_Mp_lb)   
    return np.asarray(cons_Mp_ub + cons_Mp_lb) 

def selection_criteria(x):
    """
    Selection criteria:
        Desired solution x should evaluates to all
        non-negative
            1. Minimum non-zero component must be at least 10%
               of the max component. (note: non-zero is defined
               to be <= ZERO_CUTOFF, purely for numerical stability)
            2. Allow for at most ternary systems
    """
    ZERO_CUTOFF    = 0.1    

    # Criterion 1
    xcomps         = x[:3]
    x_non_zero     = xcomps[xcomps > ZERO_CUTOFF]
    x_max          = xcomps.max()
    x_min          = x_non_zero.min()
    comp_condition = [x_min - 0.1*x_max]

    # Criterion 2
    num_non_zero      = np.sum(x[:3] > ZERO_CUTOFF)
    nonzero_condition = [3.1 - num_non_zero]
    # return np.asarray([1.0])
        
    return all([c >= 0 for c in comp_condition + nonzero_condition])

"""## Optimze via Particle Swarm Optimizer

The optimisation options are found using a systematic parameter study that we omit here for brevity
"""
print('@@@@@@ 7 trained regressor:', trained_regressor)
opt = optimizer.PSO_Optimizer(
    regressor=trained_regressor,
    scaler=scaler_regressor,
    objective=objective,
    constraints=constraints,
    selection_criteria=selection_criteria
)

optimisation_options = { 'criteria': 0.1,
                        'max_rounds': 200,
                        'n_solutions': 50,  #chi hao 50
                        'lb':lb, 'ub':ub, 
                        'opt_vars': [ 'Ni', 'Co', 'Mn', 'Size', 'First_sin', 'FS_time', 'Sec_sin', 'SS_time', 'Cutoff', 'C_rate', 'total'],
                        'swarmsize': 400, 'omega':0.1, 'phip':0.9, 'phig': 0.8, 'maxiter': 100, 
                        'debug_flag': False,
                        'nprocessors': 12
                       }

"""### 1. Target """

optimisation_options['target_CP'] = 150
results1, runtime  =  opt.optimisation_parallel(optimisation_options)
print('150 mAh/g:\n', results1)
results1.to_excel("result150mice.xlsx", sheet_name='Sheet_name_1')

"""### 2. Target """

optimisation_options['target_CP'] = 175
results2, runtime  =  opt.optimisation_parallel(optimisation_options)
print('175 mAh/g:\n', results2)
results2.to_excel("result175mice.xlsx", sheet_name='Sheet_name_1')



"""### 3. Target """

optimisation_options['target_CP'] = 200
results3, runtime  =  opt.optimisation_parallel(optimisation_options)
print('200 mAh/g:\n', results3)
results3.to_excel("result200mice.xlsx", sheet_name='Sheet_name_1')

"""### 4. Target"""

optimisation_options['target_CP'] = 225
results4, runtime  =  opt.optimisation_parallel(optimisation_options)
print('225 mAh/g:\n', results4)
results4.to_excel("result225mice.xlsx", sheet_name='Sheet_name_1')


"""## NN validation of the XGB results"""
def NN_Mean_Std(input_df, scaler, ensb_reg):
    
    data = deepcopy(input_df)
    inputs   = scaler_regressor.transform(data.iloc[:, :11 ].values)
    
    mean = ensb_regressor.predict_mean(inputs)
    cov  = ensb_regressor.predict_covariance(inputs)
    std  = np.sqrt(cov.squeeze())
    
    data['Mean_Pred_NN'] = mean
    data['Std_Error_NN']  = std
    
    data['%Error_Mean_Pred_NN'] = 100*(data.Mean_Pred_NN - data.target_CP)/data.target_CP
    
    
    return data

NN_validation1 = NN_Mean_Std(results1, scaler_regressor, ensb_regressor)
print('test validation:\n', NN_validation1)
NN_validation1.to_excel("validation150mice.xlsx", sheet_name='Sheet_name_1')

NN_validation2 = NN_Mean_Std(results2, scaler_regressor, ensb_regressor)
print('test validation:\n', NN_validation2)
NN_validation2.to_excel("validation175mice.xlsx", sheet_name='Sheet_name_1')

NN_validation3 = NN_Mean_Std(results3, scaler_regressor, ensb_regressor)
print('test validation:\n', NN_validation3)
NN_validation3.to_excel("validation200mice.xlsx", sheet_name='Sheet_name_1')

NN_validation4 = NN_Mean_Std(results4, scaler_regressor, ensb_regressor)
print('test validation:\n', NN_validation4)
NN_validation4.to_excel("validation225mice.xlsx", sheet_name='Sheet_name_1')

