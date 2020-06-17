## Script written by Jose` Y. Villafan.
## Last edited on 17/01/2020.
## BOSS-V algorithm.

import warnings, argparse, time, datetime, os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configparser import SafeConfigParser
from structure1_utils import proper_round, relative_error, MAPE, MPE
from structure2_datapreparation import CDataPreparation
from structure3_bayesianoptimization import CB, PI, EI, EIC, SEI
from sklearn.gaussian_process.kernels import Matern, RBF
from structure_bossv import CBOSSV as BOSS, compute_error_on_config, plot_cumulative

start = time.time()

seed = 0
np.random.seed(seed)


# 1. Read configuration file and dataset paths
#####################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--config','-c',metavar = 'config_file',type = str,help = 'config file path',required = True)
parser.add_argument('--output','-o',metavar = 'output_file',type = str,help = 'output path',required = True)
# parser.add_argument('--input1','-i1',metavar = 'input_1',type = str,help = 'input 1 path',required = True)
args = parser.parse_args()

config_parser = SafeConfigParser()
if os.path.isfile(args.config):
    config_parser.read(args.config)
    folder_path = str(config_parser.get('Folderpath','folder_path'))
else:
    print("Config file not found")


# 2. Prepare datasets X and Y for training and testing
#####################################################################################################################

##- Define price per unit time P(x), function of the configuration
def price_function(nCore):
    return nCore

data_preparation = CDataPreparation(args.config,verbose=False)
X_train, X_test, y_train, y_test, T_train, T_test, P_train, P_test = data_preparation.generate_XY_for_BO(price_function=price_function,apply_log_y=False)
features = data_preparation.model_regressors


# 3. Start full run SFS+BO on Performance models
#####################################################################################################################
max_iter = 30
max_features = 10

K_dict = {"Matern52": Matern(length_scale=[1.0],nu=2.5)}
AF_dict = {"EIC": EIC}

threshold1 = 2.2e6 if data_preparation.dataset_kind == "Kmeans" else 1.2e6
threshold2 = 600000
threshold3 = 500000
threshold4 = 400000
threshold5 = 200000
thresholds = list(map(lambda x: x/3600000, [threshold1, threshold2, threshold3, threshold4, threshold5]))
thresholds = [thresholds[0]]      # highest deadline
# thresholds = [thresholds[1]]      # 600000 ms
# thresholds = [thresholds[2]]      # 500000 ms
# thresholds = [thresholds[3]]      # 400000 ms
# thresholds = [thresholds[4]]      # 200000 ms

# model_features = []                                     # SFFS starts from an empty set
# model_features = 'all'                                  # SFFS starts from a full set => pure BO

##- Gray box models start with both nContainers and 1/nContainers features
##- On the contrary, black box models start with all their 6 features
if data_preparation.box_model == "gray":
    nContainer_index = 106 if data_preparation.dataset_kind == "Kmeans" else 51
    model_features = [nContainer_index+1]
    nC_feature = [nContainer_index]
else:
    nContainer_index = 1
    model_features = 'all'
    nC_feature = [nContainer_index]

##- Set these parameters if you want to have 1D optimization on nContainers dimension
# max_features = 1
# model_features = []
#--------------------------------------------------------------------------------------------------------------------

Splits, Cases = data_preparation.get_splits_and_cases_dict()
iSplit = 2 # Use only this split, otherwise cumulative costs will not reflect the outcomes
for iCase in Cases.keys():
# for iCase in [1]:
    print("***** Exe : {} - {} - {}".format(data_preparation.dataset_kind,data_preparation.box_model,data_preparation.split))
    print("***** Case: {}".format(iCase))

    for K_key in K_dict.keys():
        for AF_key in AF_dict.keys():
            for level in thresholds:
                print("***** Run      : {} - {}".format(K_key,AF_key))
                print("***** Threshold: {}".format(level))

                data_preparation.print_info_for_BO(X_train,X_test,iSplit,iCase)

                exe_file = open(os.path.join("./",str(args.output)), 'a')
                exe_file.write("***** Exe : {} - {} - {}".format(data_preparation.dataset_kind,data_preparation.box_model,data_preparation.split) + "\n")
                exe_file.write("***** Case: {}".format(iCase) + "\n")
                exe_file.write("***** Run : {} - {}".format(K_key,AF_key) + "\n")
                exe_file.write("      Threshold = " + str(level) + "\n")
                exe_file.close()

                ##- Execution
                run_start_time = time.time()

                X_add, y_add, T_add = [], [], []
                number_selected_observations = 0
                ##- Bayesian optimization on training datasets
                for idataSize in X_train[iSplit][iCase].keys():
                    n_train_obs = X_train[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_train[iSplit][iCase][idataSize]["train"],X_train[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_train[iSplit][iCase][idataSize]["train"],y_train[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_train[iSplit][iCase][idataSize]["train"],T_train[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_train[iSplit][iCase][idataSize]["train"],P_train[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,
                                        initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=False,mix_models=False,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict

                    ##- Save important output between BOs
                    X_add.append(gridBO_X[resObs,:])
                    y_add.append(gridBO_Y[resObs,:])
                    T_add.append(Times[resObs,:])
                    number_selected_observations += len(resObs)
                    print("chosen observations: {}\n".format(resObs))

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    model.plot_PREevaluation(title = path + subpath)
                    model.plot_POSTevaluation(title = path + subpath)
                    model.plot_convergence(title = path + subpath)
                    model.plot_regret(title = path + subpath)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.close()


                ##- Collect information from preparatory runs
                X_add = np.array(X_add).squeeze().reshape(number_selected_observations,-1)
                y_add = np.array(y_add).squeeze().reshape(number_selected_observations,1)
                T_add = np.array(T_add).squeeze().reshape(number_selected_observations,1)

                ##- Create empty lists for plotting and comparing cumulative costs
                no_data_cum_costs, avec_data_cum_costs = [], []
                no_data_stopping_iter, avec_data_stopping_iter = [], []

                #####################################
                ##- Variant A without additional data
                for idataSize in X_test[iSplit][iCase].keys():
                    n_train_obs = X_test[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_test[iSplit][iCase][idataSize]["train"],X_test[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_test[iSplit][iCase][idataSize]["train"],y_test[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_test[iSplit][iCase][idataSize]["train"],T_test[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_test[iSplit][iCase][idataSize]["train"],P_test[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=False,mix_models=False,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        X_add=None,Y_add=None,Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    no_data_cum_costs.append(cumulative_costs)
                    no_data_stopping_iter.append(stopping_iter)

                    ##- Evalute performance prediction via MAPE
                    X_pp, Y_pp = gridBO_X[sObs,:], gridBO_Y[sObs,:]
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_PRE, MPE_eval_PRE = MAPE(rel_err), MPE(rel_err)

                    ##- Evalute performance prediction via MAPE after BO
                    PPobs = list(filter(lambda el: el < n_train_obs, resObs))
                    n_SampledTrain_iter = len(PPobs)
                    X_pp, Y_pp = gridBO_X[PPobs,:], gridBO_Y[PPobs,:]
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_POST, MPE_eval_POST = MAPE(rel_err), MPE(rel_err)

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantA_noDATA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write(" - Variant A w/out data - \n")
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.write("                        = " + "\n")
                    exe_file.write("   MAPE_eval_PRE        = " + str(MAPE_eval_PRE) + "\n")
                    exe_file.write("   MPE_eval_PRE         = " + str(MPE_eval_PRE) + "\n")
                    exe_file.write("   n_SampledTrain_iter  = " + str(n_SampledTrain_iter) + "\n")
                    exe_file.write("   MAPE_eval_POST       = " + str(MAPE_eval_POST) + "\n")
                    exe_file.write("   MPE_eval_POST        = " + str(MPE_eval_POST) + "\n")
                    exe_file.close()

                #####################################
                ##- Variant A with additional data
                for idataSize in X_test[iSplit][iCase].keys():
                    n_train_obs = X_test[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_test[iSplit][iCase][idataSize]["train"],X_test[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_test[iSplit][iCase][idataSize]["train"],y_test[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_test[iSplit][iCase][idataSize]["train"],T_test[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_test[iSplit][iCase][idataSize]["train"],P_test[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=False,mix_models=False,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        X_add=X_add,Y_add=T_add,Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    avec_data_cum_costs.append(cumulative_costs)
                    avec_data_stopping_iter.append(stopping_iter)

                    ##- Evalute performance prediction via MAPE
                    X_pp, Y_pp = np.vstack((X_add,gridBO_X[sObs,:])), np.vstack((y_add,gridBO_Y[sObs,:]))
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_PRE, MPE_eval_PRE = MAPE(rel_err), MPE(rel_err)

                    ##- Evalute performance prediction via MAPE after BO
                    PPobs = list(filter(lambda el: el < n_train_obs, resObs))
                    n_SampledTrain_iter = len(PPobs)
                    X_pp, Y_pp = np.vstack((X_add,gridBO_X[PPobs,:])), np.vstack((y_add,gridBO_Y[PPobs,:]))
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_POST, MPE_eval_POST = MAPE(rel_err), MPE(rel_err)

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantA_avecDATA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write(" - Variant A with data  - \n")
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.write("                        = " + "\n")
                    exe_file.write("   MAPE_eval_PRE        = " + str(MAPE_eval_PRE) + "\n")
                    exe_file.write("   MPE_eval_PRE         = " + str(MPE_eval_PRE) + "\n")
                    exe_file.write("   n_SampledTrain_iter  = " + str(n_SampledTrain_iter) + "\n")
                    exe_file.write("   MAPE_eval_POST       = " + str(MAPE_eval_POST) + "\n")
                    exe_file.write("   MPE_eval_POST        = " + str(MPE_eval_POST) + "\n")
                    exe_file.close()

                #####################################
                ##- Variant B without additional data
                for idataSize in X_test[iSplit][iCase].keys():
                    n_train_obs = X_test[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_test[iSplit][iCase][idataSize]["train"],X_test[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_test[iSplit][iCase][idataSize]["train"],y_test[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_test[iSplit][iCase][idataSize]["train"],T_test[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_test[iSplit][iCase][idataSize]["train"],P_test[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=False,mix_models=True,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        X_add=None,Y_add=None,Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    no_data_cum_costs.append(cumulative_costs)
                    no_data_stopping_iter.append(stopping_iter)

                    ##- Evalute performance prediction via MAPE
                    X_pp, Y_pp = gridBO_X[sObs,:], gridBO_Y[sObs,:]
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_PRE, MPE_eval_PRE = MAPE(rel_err), MPE(rel_err)

                    ##- Evalute performance prediction via MAPE after BO
                    PPobs = list(filter(lambda el: el < n_train_obs, resObs))
                    n_SampledTrain_iter = len(PPobs)
                    X_pp, Y_pp = gridBO_X[PPobs,:], gridBO_Y[PPobs,:]
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_POST, MPE_eval_POST = MAPE(rel_err), MPE(rel_err)

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantB_noDATA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write(" - Variant B w/out data - \n")
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.write("                        = " + "\n")
                    exe_file.write("   MAPE_eval_PRE        = " + str(MAPE_eval_PRE) + "\n")
                    exe_file.write("   MPE_eval_PRE         = " + str(MPE_eval_PRE) + "\n")
                    exe_file.write("   n_SampledTrain_iter  = " + str(n_SampledTrain_iter) + "\n")
                    exe_file.write("   MAPE_eval_POST       = " + str(MAPE_eval_POST) + "\n")
                    exe_file.write("   MPE_eval_POST        = " + str(MPE_eval_POST) + "\n")
                    exe_file.close()

                #####################################
                ##- Variant B with additional data
                for idataSize in X_test[iSplit][iCase].keys():
                    n_train_obs = X_test[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_test[iSplit][iCase][idataSize]["train"],X_test[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_test[iSplit][iCase][idataSize]["train"],y_test[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_test[iSplit][iCase][idataSize]["train"],T_test[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_test[iSplit][iCase][idataSize]["train"],P_test[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=False,mix_models=True,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        X_add=X_add,Y_add=T_add,Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    avec_data_cum_costs.append(cumulative_costs)
                    avec_data_stopping_iter.append(stopping_iter)

                    ##- Evalute performance prediction via MAPE
                    X_pp, Y_pp = np.vstack((X_add,gridBO_X[sObs,:])), np.vstack((y_add,gridBO_Y[sObs,:]))
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_PRE, MPE_eval_PRE = MAPE(rel_err), MPE(rel_err)

                    ##- Evalute performance prediction via MAPE after BO
                    PPobs = list(filter(lambda el: el < n_train_obs, resObs))
                    n_SampledTrain_iter = len(PPobs)
                    X_pp, Y_pp = np.vstack((X_add,gridBO_X[PPobs,:])), np.vstack((y_add,gridBO_Y[PPobs,:]))
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_POST, MPE_eval_POST = MAPE(rel_err), MPE(rel_err)

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantB_avecDATA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write(" - Variant B with data  - \n")
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.write("                        = " + "\n")
                    exe_file.write("   MAPE_eval_PRE        = " + str(MAPE_eval_PRE) + "\n")
                    exe_file.write("   MPE_eval_PRE         = " + str(MPE_eval_PRE) + "\n")
                    exe_file.write("   n_SampledTrain_iter  = " + str(n_SampledTrain_iter) + "\n")
                    exe_file.write("   MAPE_eval_POST       = " + str(MAPE_eval_POST) + "\n")
                    exe_file.write("   MPE_eval_POST        = " + str(MPE_eval_POST) + "\n")
                    exe_file.close()

                #####################################
                ##- Variant C without additional data
                for idataSize in X_test[iSplit][iCase].keys():
                    n_train_obs = X_test[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_test[iSplit][iCase][idataSize]["train"],X_test[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_test[iSplit][iCase][idataSize]["train"],y_test[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_test[iSplit][iCase][idataSize]["train"],T_test[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_test[iSplit][iCase][idataSize]["train"],P_test[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=True,mix_models=False,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        X_add=None,Y_add=None,Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    no_data_cum_costs.append(cumulative_costs)
                    no_data_stopping_iter.append(stopping_iter)

                    ##- Evalute performance prediction via MAPE
                    X_pp, Y_pp = gridBO_X[sObs,:], gridBO_Y[sObs,:]
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_PRE, MPE_eval_PRE = MAPE(rel_err), MPE(rel_err)

                    ##- Evalute performance prediction via MAPE after BO
                    PPobs = list(filter(lambda el: el < n_train_obs, resObs))
                    n_SampledTrain_iter = len(PPobs)
                    X_pp, Y_pp = gridBO_X[PPobs,:], gridBO_Y[PPobs,:]
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_POST, MPE_eval_POST = MAPE(rel_err), MPE(rel_err)

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantC_noDATA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write(" - Variant C w/out data - \n")
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.write("                        = " + "\n")
                    exe_file.write("   MAPE_eval_PRE        = " + str(MAPE_eval_PRE) + "\n")
                    exe_file.write("   MPE_eval_PRE         = " + str(MPE_eval_PRE) + "\n")
                    exe_file.write("   n_SampledTrain_iter  = " + str(n_SampledTrain_iter) + "\n")
                    exe_file.write("   MAPE_eval_POST       = " + str(MAPE_eval_POST) + "\n")
                    exe_file.write("   MPE_eval_POST        = " + str(MPE_eval_POST) + "\n")
                    exe_file.close()

                #####################################
                ##- Variant C with additional data
                for idataSize in X_test[iSplit][iCase].keys():
                    n_train_obs = X_test[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_test[iSplit][iCase][idataSize]["train"],X_test[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_test[iSplit][iCase][idataSize]["train"],y_test[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_test[iSplit][iCase][idataSize]["train"],T_test[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_test[iSplit][iCase][idataSize]["train"],P_test[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=True,mix_models=False,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        X_add=X_add,Y_add=T_add,Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    avec_data_cum_costs.append(cumulative_costs)
                    avec_data_stopping_iter.append(stopping_iter)

                    ##- Evalute performance prediction via MAPE
                    X_pp, Y_pp = np.vstack((X_add,gridBO_X[sObs,:])), np.vstack((y_add,gridBO_Y[sObs,:]))
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_PRE, MPE_eval_PRE = MAPE(rel_err), MPE(rel_err)

                    ##- Evalute performance prediction via MAPE after BO
                    PPobs = list(filter(lambda el: el < n_train_obs, resObs))
                    n_SampledTrain_iter = len(PPobs)
                    X_pp, Y_pp = np.vstack((X_add,gridBO_X[PPobs,:])), np.vstack((y_add,gridBO_Y[PPobs,:]))
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_POST, MPE_eval_POST = MAPE(rel_err), MPE(rel_err)

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantC_avecDATA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write(" - Variant C with data  - \n")
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.write("                        = " + "\n")
                    exe_file.write("   MAPE_eval_PRE        = " + str(MAPE_eval_PRE) + "\n")
                    exe_file.write("   MPE_eval_PRE         = " + str(MPE_eval_PRE) + "\n")
                    exe_file.write("   n_SampledTrain_iter  = " + str(n_SampledTrain_iter) + "\n")
                    exe_file.write("   MAPE_eval_POST       = " + str(MAPE_eval_POST) + "\n")
                    exe_file.write("   MPE_eval_POST        = " + str(MPE_eval_POST) + "\n")
                    exe_file.close()

                #####################################
                ##- Variant D without additional data
                for idataSize in X_test[iSplit][iCase].keys():
                    n_train_obs = X_test[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_test[iSplit][iCase][idataSize]["train"],X_test[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_test[iSplit][iCase][idataSize]["train"],y_test[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_test[iSplit][iCase][idataSize]["train"],T_test[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_test[iSplit][iCase][idataSize]["train"],P_test[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=True,mix_models=True,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        X_add=None,Y_add=None,Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    no_data_cum_costs.append(cumulative_costs)
                    no_data_stopping_iter.append(stopping_iter)

                    ##- Evalute performance prediction via MAPE
                    X_pp, Y_pp = gridBO_X[sObs,:], gridBO_Y[sObs,:]
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_PRE, MPE_eval_PRE = MAPE(rel_err), MPE(rel_err)

                    ##- Evalute performance prediction via MAPE after BO
                    PPobs = list(filter(lambda el: el < n_train_obs, resObs))
                    n_SampledTrain_iter = len(PPobs)
                    X_pp, Y_pp = gridBO_X[PPobs,:], gridBO_Y[PPobs,:]
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_POST, MPE_eval_POST = MAPE(rel_err), MPE(rel_err)

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantD_noDATA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write(" - Variant D w/out data - \n")
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.write("                        = " + "\n")
                    exe_file.write("   MAPE_eval_PRE        = " + str(MAPE_eval_PRE) + "\n")
                    exe_file.write("   MPE_eval_PRE         = " + str(MPE_eval_PRE) + "\n")
                    exe_file.write("   n_SampledTrain_iter  = " + str(n_SampledTrain_iter) + "\n")
                    exe_file.write("   MAPE_eval_POST       = " + str(MAPE_eval_POST) + "\n")
                    exe_file.write("   MPE_eval_POST        = " + str(MPE_eval_POST) + "\n")
                    exe_file.close()

                #####################################
                ##- Variant D with additional data
                for idataSize in X_test[iSplit][iCase].keys():
                    n_train_obs = X_test[iSplit][iCase][idataSize]["train"].shape[0]
                    gridBO_X = np.vstack((X_test[iSplit][iCase][idataSize]["train"],X_test[iSplit][iCase][idataSize]["test"]))
                    gridBO_Y = np.vstack((y_test[iSplit][iCase][idataSize]["train"],y_test[iSplit][iCase][idataSize]["test"]))
                    Times    = np.vstack((T_test[iSplit][iCase][idataSize]["train"],T_test[iSplit][iCase][idataSize]["test"]))
                    Price    = np.vstack((P_test[iSplit][iCase][idataSize]["train"],P_test[iSplit][iCase][idataSize]["test"]))

                    # sObs = random.sample(range(0,n_train_obs),k=3)      # random configurations
                    sObs = np.linspace(0,n_train_obs,3)                 # equidistributed configurations
                    # sObs = [0,11,n_train_obs]                           # fixed different configurations, n0 = 3
                    sObs = list(map(lambda  x: int(proper_round(x,0)), sObs))

                    bounds = np.full((gridBO_X.shape[1],2),0.0)
                    for i in range(0,gridBO_X.shape[1]):
                        bounds[i,:] = [np.min(gridBO_X[:,i]), np.max(gridBO_X[:,i])]
                    bounds = np.array(bounds)

                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=True,mix_models=True,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        X_add=X_add,Y_add=T_add,Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    avec_data_cum_costs.append(cumulative_costs)
                    avec_data_stopping_iter.append(stopping_iter)

                    ##- Evalute performance prediction via MAPE
                    X_pp, Y_pp = np.vstack((X_add,gridBO_X[sObs,:])), np.vstack((y_add,gridBO_Y[sObs,:]))
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_PRE, MPE_eval_PRE = MAPE(rel_err), MPE(rel_err)

                    ##- Evalute performance prediction via MAPE after BO
                    PPobs = list(filter(lambda el: el < n_train_obs, resObs))
                    n_SampledTrain_iter = len(PPobs)
                    X_pp, Y_pp = np.vstack((X_add,gridBO_X[PPobs,:])), np.vstack((y_add,gridBO_Y[PPobs,:]))
                    fitted_lm = model.fo.refit_regression(X_pp,Y_pp)[1]
                    rel_err = relative_error(y_test[iSplit][iCase][idataSize]["test"],fitted_lm.predict(X_test[iSplit][iCase][idataSize]["test"]))
                    MAPE_eval_POST, MPE_eval_POST = MAPE(rel_err), MPE(rel_err)

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantD_avecDATA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write(" - Variant D with data  - \n")
                    exe_file.write(" ------------------------ " + "\n")
                    exe_file.write("   idataSize            = " + str(idataSize) + "\n")
                    exe_file.write("   features    : " + str(output_dict["features"]) + "\n")
                    exe_file.write("   observations: " + str(output_dict["observations"]) + "\n")
                    exe_file.write("   Time bounds : " + str(min(Times)) + " - " + str(max(Times)) + "\n")
                    exe_file.write("   Cost bounds : " + str(min(gridBO_Y)) + " - " + str(max(gridBO_Y)) + "\n")
                    exe_file.write("   Best cost   : " + str(output_dict["Y"]) + "\n")
                    exe_file.write("   TERM_features        = " + str(term_features) + "\n")
                    exe_file.write("   TERM_observations    = " + str(term_obs) + "\n")
                    exe_file.write("   TERM_cumulative_cost = " + str(cum_cost_at_stop) + "\n")
                    exe_file.write("   over_the_top_counter = " + str(over_the_top_counter) + "\n")
                    exe_file.write("   n_iter               = " + str(n_iter) + "\n")
                    exe_file.write("   nCPE                 = " + str(output_dict["nC_error"]) + "\n")
                    exe_file.write("   nContainer selected  = " + str(output_dict["nContainer_sel"]) + "\n")
                    exe_file.write("   nContainer minimum   = " + str(output_dict["nContainer_min"]) + "\n")
                    exe_file.write("                        = " + "\n")
                    exe_file.write("   MAPE_eval_PRE        = " + str(MAPE_eval_PRE) + "\n")
                    exe_file.write("   MPE_eval_PRE         = " + str(MPE_eval_PRE) + "\n")
                    exe_file.write("   n_SampledTrain_iter  = " + str(n_SampledTrain_iter) + "\n")
                    exe_file.write("   MAPE_eval_POST       = " + str(MAPE_eval_POST) + "\n")
                    exe_file.write("   MPE_eval_POST        = " + str(MPE_eval_POST) + "\n")
                    exe_file.close()

                ##- Plot comulative costs for comparison across the different algorithm variants
                path = folder_path + \
                       data_preparation.dataset_kind + '_' + \
                       data_preparation.box_model + '_' + \
                       data_preparation.split + '/'
                subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                variant = '_allVariants'
                plot_cumulative(no_data_cum_costs,no_data_stopping_iter,avec_data_cum_costs,avec_data_stopping_iter,title = path + subpath + variant)


                #####################################
                ##- Write results to output file

                exe_file = open(os.path.join("./",str(args.output)), 'a')
                run_finish_time = time.time()
                run_time = str(run_finish_time-run_start_time)
                exe_file.write("  Execution time: " + str(run_time) + "\n\n")
                exe_file.close()


# Tests finished
#####################################################################################################################

end = time.time()

execution_time = str(end-start)
print("Execution Time: " + execution_time)

exe_time_file = open(os.path.join("./","output.txt"), 'a')
exe_time_file.write(str(datetime.datetime.today())+": "+str(execution_time)+"\n")
exe_time_file.close()
