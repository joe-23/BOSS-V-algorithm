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

# K_dict = {"RBF": RBF(length_scale=[1.0]),
#           "Matern52": Matern(length_scale=[1.0],nu=2.5)}
K_dict = {"Matern52": Matern(length_scale=[1.0],nu=2.5)}

# AF_dict = {"CB": CB, "PI": PI, "EI": EI, "EIC": EIC, "SEI": SEI}
AF_dict = {"EIC": EIC}

threshold1 = 2.2e6 if data_preparation.dataset_kind == "Kmeans" else 1.2e6
threshold2 = 600000
threshold3 = 500000
threshold4 = 400000
threshold5 = 200000
thresholds = list(map(lambda x: x/3600000, [threshold1, threshold2, threshold3, threshold4, threshold5]))
# thresholds = [thresholds[3], thresholds[1]]      # 400000 ms, 600000 ms
thresholds = [thresholds[2], thresholds[0]]      # 500000 ms, highest deadline

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
iCase = 1 # Use only this case, they all have the same data if running interpolation analysis
for iSplit in Splits.keys():
    print("***** Exe : {} - {} - {}".format(data_preparation.dataset_kind,data_preparation.box_model,data_preparation.split))

    for K_key in K_dict.keys():
        for AF_key in AF_dict.keys():
            for idataSize in X_train[iSplit][iCase].keys():

                exe_file = open(os.path.join("./",str(args.output)), 'a')
                exe_file.write("***** Exe : {} - {} - {}".format(data_preparation.dataset_kind,data_preparation.box_model,data_preparation.split) + "\n")
                exe_file.write("***** Case: {}".format(iSplit) + "\n")
                exe_file.write("***** Run : {} - {} - {}".format(K_key,AF_key,idataSize) + "\n")
                exe_file.close()

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

                ##- Execution
                run_start_time = time.time()

                ##- Bayesian optimization on training datasets
                for level in thresholds:
                    print("***** Case     : {}".format(iCase))
                    print("***** Run      : {} - {} - {}".format(K_key,AF_key,idataSize))
                    print("***** Threshold: {}".format(level))

                    ##- Create empty lists for plotting and comparing cumulative costs across variants and thresholds
                    cum_costs_list, stopping_iter_list = [], []

                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write("      Threshold = " + str(level) + "\n")
                    exe_file.close()

                    ##- Variant A
                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,
                                        initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=False,mix_models=False,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    cum_costs_list.append(cumulative_costs)
                    stopping_iter_list.append(stopping_iter)

                    print("chosen observations: {}\n".format(resObs))
                    print("chosen features: {}\n".format(output_dict["features"]))
                    print("observations at termination: {}\n".format(term_obs))
                    print("features at termination: {}\n".format(term_features))

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantA'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" -- Variant A ----------- " + "\n")
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

                    ##- Variant B
                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,
                                        initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=False,mix_models=True,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    cum_costs_list.append(cumulative_costs)
                    stopping_iter_list.append(stopping_iter)

                    print("chosen observations: {}\n".format(resObs))
                    print("chosen features: {}\n".format(output_dict["features"]))
                    print("observations at termination: {}\n".format(term_obs))
                    print("features at termination: {}\n".format(term_features))

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantB'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" -- Variant B ----------- " + "\n")
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

                    ##- Variant C
                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,
                                        initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=True,mix_models=False,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    cum_costs_list.append(cumulative_costs)
                    stopping_iter_list.append(stopping_iter)

                    print("chosen observations: {}\n".format(resObs))
                    print("chosen features: {}\n".format(output_dict["features"]))
                    print("observations at termination: {}\n".format(term_obs))
                    print("features at termination: {}\n".format(term_features))

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantC'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" -- Variant C ----------- " + "\n")
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

                    ##- Variant D
                    model = BOSS(AF_dict[AF_key],K_dict[K_key],target_features=max_features,greater_is_better=False,random_state=seed)
                    output_dict = model(gridBO_X,gridBO_Y,bounds,
                                        initial_observations=sObs,initial_features=model_features,mandatory_features=nC_feature,
                                        threshold=level,enforce_threshold=True,mix_models=True,return_prob_bound=True,
                                        max_iter=max_iter,nContainer_index=nContainer_index,
                                        Times=Times,Price=Price,price_function=price_function)
                    [resObs, over_the_top_counter, n_iter, output_dict, term_obs, term_features, cumulative_costs, best_cumulative_costs, stopping_iter, cum_cost_at_stop, best_cum_cost_at_stop] = output_dict
                    cum_costs_list.append(cumulative_costs)
                    stopping_iter_list.append(stopping_iter)

                    print("chosen observations: {}\n".format(resObs))
                    print("chosen features: {}\n".format(output_dict["features"]))
                    print("observations at termination: {}\n".format(term_obs))
                    print("features at termination: {}\n".format(term_features))

                    ##- Save output graphics
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_VariantD'
                    model.plot_PREevaluation(title = path + subpath + variant)
                    model.plot_POSTevaluation(title = path + subpath + variant)
                    model.plot_convergence(title = path + subpath + variant)
                    model.plot_regret(title = path + subpath + variant)

                    ##- Write results to output file
                    exe_file = open(os.path.join("./",str(args.output)), 'a')
                    exe_file.write(" -- Variant D ----------- " + "\n")
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

                    ##- Plot comulative costs for comparison across the different algorithm variants
                    path = folder_path + \
                        data_preparation.dataset_kind + '_' + \
                        data_preparation.box_model + '_' + \
                        data_preparation.split + '/'
                    subpath = str(int(iCase)) + '_' + AF_key + '_' + K_key + '_' + str(int(level*3600000)) + '_dS' + str(int(idataSize))
                    variant = '_allVariants'
                    plot_cumulative(cum_costs_list,stopping_iter_list,None,None,title = path + subpath + variant)

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
