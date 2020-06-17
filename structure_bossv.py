## Script written by Jose` Y. Villafan.
## Last edited on 28/02/2020.
## BOSS-V algorithm.
# Copyright (c) 2020, the BOSS-V author (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE)

import warnings, random
import numpy as np
import matplotlib.pyplot as plt
from pylab import grid
from sklearn.preprocessing import StandardScaler
from structure1_utils import WrongCallToMethod, find_indices, min_dist, relative_error, MAPE, MPE
from structure5_featureoptimization import CFeatureOpt

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 12}
import matplotlib
matplotlib.rc('font', **font)

debug = False                   # Print information at debug level (e.g. model parameters and CV score)


# 1. Define BOSS-V algorithm class
#####################################################################################################################
class CBOSSV(object):
    """Callable class that performs Sequential Forward Feature Selection (SFS) and Bayesian optimization (BO) on a specified Performance model (see [Villafan (2020); Chapter 3.2] for details)."""

    def __init__(self,af,GPkernel,floating=True,target_features=None,scaling=True,greater_is_better=True,random_state=None,verbose=True):
        self.fo         = CFeatureOpt(af,GPkernel,floating,target_features,greater_is_better,random_state,verbose)
        if scaling:
            self.scaler = StandardScaler()                          # Standardization will be performed on X matrix
        self.verbose    = verbose


    def __call__(self,X_train,Y_train,bounds,nContainer_index,threshold,
                 initial_observations=[],initial_features=[],mandatory_features=[],
                 enforce_threshold=False,mix_models=False,return_prob_bound=False,
                 max_iter=20,max_iter_BO=100,
                 X_add=None,Y_add=None,Times=None,Price=None,price_function=None):
        """
            Callable method that implements an iterative algorithm to perform feature selection and Bayesian optimization on Performance models.
                WARNING: The closest observation available in X_train is taken as the acquired point,
                         rather than the output from Bayesian optimization draw() method.

            Input: X_train - numpy 2D array
                matrix of observed data
            Input: Y_train - numpy 1D array
                vector of observed evaluations of the objective, Y_train = f(X_train)
            Input: bounds - numpy 2D array
                bounds must be an array of length d of elements [lower_bound,upper_bound]
            Input: nContainer_index - int
                number of cores feature index
            Input: threshold - float
                constraint threshold value
            Input: initial_observations - list
                list of starting observations; if empty the algorithm will choose 2 randomly among X matrix
            Input: initial_features - list
                list of indices (representing features) that are to be excluded from the search
                if initial_features == 'all' then no feature selection is performed
            Input: mandatory_features - list
                list of indices (representing features) that are not to be removed and are automatically included in the initial_features list
            Input: enforce_threshold - bool
                enforce evaluations to not be above the threshold for every acquisition criteria when True and if threshold is not None
            Input: mix_models - bool
                the linear model is used to modify the prediction from the acquisition maximization when True
            Input: return_prob_bound - bool
                if using a constrained acquisition function, the probability that the constraints are met at the result of this method is returned when True
            Input: max_iter - int
                outer loop stopping criterion based on maximum number of iterations
            Input: max_iter_BO - int
                maximization of acquisition function inner loop stopping criterion based on maximum number of iterations
            Input: X_add - numpy 2D array
                matrix of additional observed data to be used in the feature selection procedure
            Input: Y_add - numpy 1D array
                vector of additional observed evaluations of the constraint function, Y_add = T(X_add)
            Input: Times - numpy 1D array
                vector of evaluations of the constraint function, T(x)
            Input: Price - numpy 1D array
                vector of evaluations of the price function, P(x), where Y = f(X) =  T(x) * P(x)
            Input: price_function - function
                price per unit time P(x), function of the configuration
            Output: output_dict - 5 key disctionary FIXME
                dictionary containing: best configuration (achieved lowest MAPE),
                                       fitted linear model with such configuration,
                                       iteration in which such configuration was found,
                                       Y value achieved by BO algorithm at such iteration,
                                       MAPE value achieved by such configuration
        """
        ##- Check whether nContainer_index is among the mandatory_features list; if not throw error
        if not (mandatory_features and nContainer_index in mandatory_features):
            raise ValueError("nContainers feature must be among the mandatory features.")

        ##- Compute nContainers
        self.nContainers = X_train[:,nContainer_index].reshape(-1,1)

        ##- Set threshold level used by the methods in this class
        self.threshold = threshold

        ##- Scale data and bounds
        if hasattr(self,"scaler"):
            self.scaler = self.scaler.fit(X_train)
            self.X = self.scaler.transform(X_train,copy=True)
            self.bounds = self.scaler.transform(bounds.T,copy=True).T
        else:
            self.X = X_train
            self.bounds = bounds

        self.d          = self.X.shape[1]                           # Dimensionality of the observations
        self.nObs       = self.X.shape[0]                           # Number of total observations in X matrix
        self.features   = initial_features
        self.max_iter   = max_iter
        self.Y          = Y_train
        self.Times      = Times                                     # Data which is used for feature selection instead of costs Y_train
        self.Price      = Price                                     # For plotting purposes

        ##- Print some info regarding current dataset
        self.nC_upper_bound, self.nC_lower_bound = max(self.nContainers), min(self.nContainers)
        self.C_upper_bound,  self.C_lower_bound  = max(self.Y), min(self.Y)
        self.T_upper_bound,  self.T_lower_bound  = max(self.Times), min(self.Times)
        self.P_upper_bound,  self.P_lower_bound  = max(self.Price), min(self.Price)
        if self.verbose:
            print("Cost bounds: {} - {}".format(self.C_lower_bound,self.C_upper_bound))
            print("Time bounds: {} - {}".format(self.T_lower_bound,self.T_upper_bound))

        ##- Compute minimum configuration
        best_idx = self.__get_minimum_configuration()
        minimum_configuration = X_train[best_idx,:]
        self.threshold_nContainer = minimum_configuration[nContainer_index]
        self.threshold_Cost = self.Y[best_idx]
        self.threshold_Time = self.Times[best_idx]
        self.threshold_Price = self.Price[best_idx]

        ##- Set parameters used by EIC acquisition function
        lb = 0.0
        ub = self.threshold
        mean_transform, sigma_transform = self.get_transformations(nContainer_index,price_function)
        self.fo.set_EIC_params(lb = lb,ub = ub,
                                compute_mean = mean_transform,
                                compute_sigma = sigma_transform)
        if self.verbose:
            print("AF bounds: {} - {}".format(lb,ub))

        ##- Set threshold to be used by Variants C or D of the algorithm (see [Villafan (2020) Chapter 4.2])
        if enforce_threshold:
            self.fo.set_threshold(threshold=self.threshold)

        ##- Choose initial observations randomly if initial_observations is an empty list
        if not initial_observations:
            initial_observations = random.sample(range(0,self.nObs),k=2)
        self.sObs       = list(initial_observations)                # list is passed by copy (not reference) to avoid overwriting it

        ##- Perform SFS + BO
        output_dict = self.__create_empty_dict_with_keys()
        self.nC_list, self.nCPE_list = [], []
        self.Y_output, self.features_amount = [], []
        self.T_output, self.P_output = [], []
        self.best_nC, self.best_Y, self.best_T, self.best_P = [], [], [], []
        self.predicted_time_list, self.predicted_cost_list = [], []
        cumulative_costs, best_cumulative_costs = [], []
        cumulative_costs_var, best_cumulative_costs_var = 0.0, 0.0

        PRE_rel_error, POST_rel_error = [], []
        self.PRE_MAPE_list, self.PRE_MPE_list, self.PRE_APE_list, self.PRE_PE_list = [], [], [], []
        self.POST_MAPE_list, self.POST_MPE_list, self.POST_APE_list, self.POST_PE_list = [], [], [], []

        Y_TO_BEAT = max(self.Y[self.sObs])
        self.check_Toutput_admissible_list = []
        self.check_Tbest_admissible_list = []
        over_the_top_counter = 0
        stop_updating = False
        for iter in range(0,self.max_iter):
            if self.verbose:
                print("***** Iteration {} *****".format(iter+1))

            ##- Perform feature selection and Bayesian optimization via CFeatureOpt call() method
            result = self.fo(self.X[self.sObs,:],self.Y[self.sObs],self.bounds,
                             initial_features=self.features,mandatory_features=mandatory_features,
                             max_iter_BO=max_iter_BO,mix_models=mix_models,return_prob_bound=return_prob_bound,
                             LR_Y=self.Times[self.sObs],X_add=X_add,LR_Y_add=Y_add)
            configuration, AF_value, self.features, fitted_lm = result[0], result[1], result[2], result[3]
            prob_bound = result[4] if return_prob_bound else float('nan')

            self.features_amount.append(len(self.features))
            if self.verbose:
                print("Features: ", self.features)
            if debug and self.verbose:
                print("Linear model intercept: ", fitted_lm.steps[-1][1].intercept_)
                print("Linear model betas: ", fitted_lm.steps[-1][1].coef_)

            ##- Find closest observation with scaled data (it works as if using Mahalanobis distance on unscaled data)
            ##- If no index is found, it means we finished the available training data: exit before max_iter
            index, full_configuration = self.__find_closest(configuration,nContainer_index)
            if index is None and full_configuration is None:
                break

            self.sObs.append(index)
            if debug and self.verbose:
                print("Observations: ", self.sObs)

            if hasattr(self,"scaler"):
                full_configuration = self.scaler.inverse_transform(full_configuration)

            ##- Observe completion time, or get Y parameter from train data
            self.Y_output.append(self.Y[index])
            self.T_output.append(self.Times[index])
            self.P_output.append(self.Price[index])
            nCPE = compute_error_on_config(minimum_configuration,full_configuration,nContainer_index)
            self.nCPE_list.append(nCPE[0])
            self.nC_list.append(nCPE[2])
            if self.verbose:
                print("Y = {}".format(self.Y_output[-1]))

            predicted_time = fitted_lm.predict(self.X[self.sObs[-1],self.features].reshape(1,-1))
            self.predicted_time_list.append(predicted_time)
            self.predicted_cost_list.append(predicted_time * price_function(nCPE[2]))

            ##- Compute MAPE on LR prediction on T(x)
            PRE_rel_error.append(relative_error(self.T_output[-1],predicted_time))

            ##- Refit Ridge with the latest observation and compute MAPE on LR prediction on T(x)
            fitted_lm = self.fo.refit_regression(self.X[self.sObs,:][:,self.features],self.Times[self.sObs])[1]
            POST_rel_error.append(relative_error(self.T_output[-1],fitted_lm.predict(self.X[self.sObs[-1],self.features].reshape(1,-1))))

            ##- Store the best obtained configuration and its evaluations
            if iter == 0:
                self.best_nC.append(self.nC_list[-1])
                self.best_Y.append(self.Y_output[-1])
                self.best_T.append(self.T_output[-1])
                self.best_P.append(self.P_output[-1])
                self.counter = 0
            else:
                ##- If improving, then store the new one
                if (self.best_T[-1] > self.threshold and self.T_output[-1] > self.threshold and self.Y_output[-1] < self.best_Y[-1])\
                    or (self.best_T[-1] > self.threshold and self.T_output[-1] < self.threshold)\
                    or (self.best_T[-1] < self.threshold and self.T_output[-1] < self.threshold and self.Y_output[-1] < self.best_Y[-1]):
                    self.best_nC.append(self.nC_list[-1])
                    self.best_Y.append(self.Y_output[-1])
                    self.best_T.append(self.T_output[-1])
                    self.best_P.append(self.P_output[-1])
                    self.counter = 0
                ##- Else repeat best values found so far
                else:
                    self.best_nC.append(self.best_nC[-1])
                    self.best_Y.append(self.best_Y[-1])
                    self.best_T.append(self.best_T[-1])
                    self.best_P.append(self.best_P[-1])
                    self.counter += 1

            ##- Store the cumulative costs for comparison and plotting
            ##- By casting to float, the value is appended rather than the variable. Please fix this Python!
            cumulative_costs_var += self.Y_output[-1]
            best_cumulative_costs_var += self.best_Y[-1]
            cumulative_costs.append(float(cumulative_costs_var))
            best_cumulative_costs.append(float(best_cumulative_costs_var))

            ##-                                                                                                       ##
            ##- Here the termination criteria would decide whether the iterations stop                                ##
            ##-                                                                                                       ##

            if not stop_updating:

                if self.T_output[-1] > self.threshold:
                    over_the_top_counter += 1

                cumulative_cost_at_stop = cumulative_costs[-1]
                best_cumulative_cost_at_stop = best_cumulative_costs[-1]

                resulting_observations = self.sObs

                self.stopping_iter = iter

                ##- Store the best obtained model
                if self.T_output[-1] < self.threshold and self.Y_output[-1] < Y_TO_BEAT:
                    output_dict["observations"]         = self.sObs
                    output_dict["features"]             = self.features
                    output_dict["full_configuration"]   = full_configuration
                    output_dict["linear_model"]         = fitted_lm
                    output_dict["n_iters"]              = iter + 1
                    output_dict["over_the_top_counter"] = over_the_top_counter
                    output_dict["Y"]                    = self.Y_output[-1]
                    output_dict["AF_value"]             = AF_value
                    output_dict["prob_bound"]           = prob_bound
                    output_dict["nC_error"]             = nCPE[0]
                    output_dict["nContainer_min"]       = nCPE[1]
                    output_dict["nContainer_sel"]       = nCPE[2]
                    Y_TO_BEAT = self.Y_output[-1]

            use_prediction = enforce_threshold or mix_models
            if self.stopping_criterion(AF_value,prob_bound,iter,over_the_top_counter,use_prediction):
                stop_updating = True

            ##-                                                                                                       ##
            ##- Compute necessary information for plotting                                                            ##
            ##-                                                                                                       ##

            ##- Compute and store prediction errors
            self.PRE_MAPE_list.append(MAPE(PRE_rel_error[:iter+1]))
            self.PRE_MPE_list.append(MPE(PRE_rel_error[:iter+1]))
            self.PRE_APE_list.append(MAPE(PRE_rel_error[-1]))
            self.PRE_PE_list.append(MPE(PRE_rel_error[-1]))

            self.POST_MAPE_list.append(MAPE(POST_rel_error[:iter+1]))
            self.POST_MPE_list.append(MPE(POST_rel_error[:iter+1]))
            self.POST_APE_list.append(MAPE(POST_rel_error[-1]))
            self.POST_PE_list.append(MPE(POST_rel_error[-1]))

            check = 1 if self.T_output[-1] < self.threshold else 0
            self.check_Toutput_admissible_list.append(check)
            check_best = 1 if self.best_T[-1] < self.threshold else 0
            self.check_Tbest_admissible_list.append(check_best)

            ##- Break loop if stopping criteria are met
            # if stop_updating:
            #     break

        ##- Output: resulting_observations == output_dict["observations"], it's there for convenience
        return resulting_observations, over_the_top_counter, iter+1, output_dict,\
               self.sObs, self.features,\
               cumulative_costs, best_cumulative_costs, self.stopping_iter,\
               cumulative_cost_at_stop, best_cumulative_cost_at_stop


    def stopping_criterion(self,AF_value=float('nan'),prob_bound=float('nan'),iter=0,over_the_top_counter=0,use_prediction=False):
        """
            Termination criterion employed by CBOSSV call() method. See [Villafan (2020)] for details.

            Input: AF_value - float
                value of the acquisition function at the sampled configuration
            Input: prob_bound - float
                probability that the constraints are met at the sampled configuration
            Input: iter - int
                current iteration number
            Input: over_the_top_counter - int
                number of times the algorithm found non admissible configurations
            Input: use_prediction - bool
                the linear model prediction of the time T(x) is used the True
            Output: stopping_criterion - bool
                stopping_criterion evalutes to True when termination criterion is met
        """
        stopping_criterion = False
        ##---------------------------------------------------------------------------

        ##- Stopping criterion based on improvement in the regret
        ##- MEANING: if there was no improvement in the last 10 iterations, then stop
        if self.best_T[-1] < self.threshold and self.counter > 10:
            stopping_criterion = True

        ##- Stopping criterion based on number of times that the sample does not meet the constraints
        ##- MEANING: if the constraint is not met for 3 times, then stop
        # if self.best_T[-1] < self.threshold:
        #     if over_the_top_counter > 3:
        #         stopping_criterion = True
        #     ##- However, if prediction error is low and the last sampled configuration is admissible, then continue
        #     if use_prediction and self.PRE_MAPE_list and self.PRE_MAPE_list[-1] < 10 and self.predicted_time_list[-1] < self.threshold:
        #     # if self.PRE_MAPE_list and self.PRE_MAPE_list[-1] < 10 and self.predicted_time_list[-1] < self.threshold:
        #         stopping_criterion = False

        ##- Stopping criterion based on acquisition function value
        ##- MEANING: if there is no improvement to be had, then stop
        if not np.isnan(AF_value) and AF_value < 0.1:
            stopping_criterion = True

        ##- Stopping criterion based on the probability that the last sample meets the constraints
        ##- MEANING: if the constraint is met with high probability, then stop
        # if not np.isnan(prob_bound) and prob_bound > 0.95:
        #     stopping_criterion = True

        ##- Stopping criterion based on having found a near-optimal configuration w.r.t. threshold
        ##- MEANING: if the optimal configuration has been found, then stop
        if self.best_T[-1] <= self.threshold and self.best_T[-1] >= 0.9 * self.threshold:
            stopping_criterion = True

        ##- Stopping criterion based on number of minimum iterations
        if iter < 6:
            stopping_criterion = False

        return stopping_criterion


    def get_transformations(self,nContainer_index,price_function):
        """
            Return mu and sigma transformation functions which compute the constraint's mu and sigma from the objective's mu and sigma.
                WARNING: this method's definition and signature must be changed if the price function is not 1D anymore.

            Input: nContainer_index - int
                number of cores feature index
            Input: price_function - function
                function that returns the price per unit time with the chosen configuration
            Output: mean_transform, sigma_transform - function, function
                functions whose signature is signature f(mu,x) and signature g(sigma,x)
        """
        def mean_transform(mu,x):
            if hasattr(self,"scaler"):
                temp = np.full((1,self.d),0.0)
                temp[:,nContainer_index] = x
                unscaled_temp = self.scaler.inverse_transform(temp).reshape(1,-1)
                x = unscaled_temp[:,nContainer_index]
            return mu / price_function(x)

        def sigma_transform(sigma,x):
            if hasattr(self,"scaler"):
                temp = np.full((1,self.d),0.0)
                temp[:,nContainer_index] = x
                unscaled_temp = self.scaler.inverse_transform(temp).reshape(1,-1)
                x = unscaled_temp[:,nContainer_index]
            return sigma / price_function(x)**2

        ##- FIXME: compute transformations when using logarithms

        return mean_transform, sigma_transform


    def __find_closest(self,x,nContainer_index):
        """
            Find index of closest observation in matrix self.X to observation x.
            Look for observations with equal number of cores, then proceed to look for the closest in L2 distance.
            If data has been scaled it works as if using Mahalanobis distance.

            Input: x - numpy 1D array
                sampled configuration
            Input: nContainer_index - int
                number of cores feature index
            Output: index, observation - int, numpy 1D array
        """
        ##- Define list of observations which have not been selected yet
        all_observations = list(range(0,self.nObs))
        new_observations = list(sorted(set(all_observations) - set(self.sObs)))

        ##- Compute index of mandatory configuration features among the ones selected by the feature selector
        idx = find_indices(self.features,lambda  x_idx: x_idx == nContainer_index)[0]

        ##- Define list of observations with same number of cores as sampled configuration (they are not integer values if data has been scaled)
        equal_nContainer_observations = []
        for obs in new_observations:
            observed_value = self.X[obs,self.features[idx]]
            sampled_value = x[idx]
            if abs(observed_value - sampled_value) < 1e-3:
                equal_nContainer_observations.append(obs)

        if equal_nContainer_observations:
            temp_index = min_dist(self.X[equal_nContainer_observations,:][:,self.features],x)[1]
        else:
            temp_index = min_dist(self.X[new_observations,:][:,self.features],x)[1]

        if temp_index is None:
            return None, None
        else:
            index = new_observations[temp_index]
            return index, self.X[index,:]


    def __create_empty_dict_with_keys(self):
        empty_dict = dict()

        empty_dict["observations"]          = float('nan')
        empty_dict["features"]              = float('nan')
        empty_dict["full_configuration"]    = float('nan')
        empty_dict["linear_model"]          = float('nan')
        empty_dict["n_iters"]               = float('nan')
        empty_dict["over_the_top_counter"]  = float('nan')
        empty_dict["Y"]                     = float('nan')
        empty_dict["AF_value"]              = float('nan')
        empty_dict["prob_bound"]            = float('nan')
        empty_dict["nC_error"]              = float('nan')
        empty_dict["nContainer_min"]        = float('nan')
        empty_dict["nContainer_sel"]        = float('nan')

        return empty_dict


    def __get_minimum_configuration(self):
        """
            This is basically like solving the problem with 5 lines of code..
            However, this can be done a posteriori only. And I wrote 4300+ lines of code to solve the problem.
            Worth it!
        """
        admissible_configurations = []
        for iVar in range(0,self.nObs):
            if self.Times[iVar] < self.threshold:
                admissible_configurations.append(iVar)
        if admissible_configurations:
            idx = admissible_configurations[np.argmin(self.Y[admissible_configurations])]
        else:
            idx = np.argmin(self.Times)

        return idx

        # ##- Find lowest nContainers feature value for which the time T(x) is below the threshold
        # if hasattr(self,"threshold"):
        #     ##- Create a structured array and sort it by cores
        #     dvals = [tuple(elem) for elem in np.hstack((self.Times,self.Y,self.nContainers))]
        #     dtype = [('Time',float), ('Cost',float), ('cores',int)]
        #     vals = np.sort(np.array(dvals, dtype=dtype), order=('cores','Time'))

        #     ##- Find the tuple whose Time is below the threshold
        #     tuple_result = vals[-1]
        #     for elem in vals[::-1]:
        #         if elem['Time'] > self.threshold:
        #             break
        #         tuple_result = elem
        #     self.threshold_nContainer = tuple_result['cores']


    def plot_convergence(self,title="",show=False):
        """
            Plot number of cores, costs C(x), Times T(x) and price P(x) w.r.t. number of iterations.
                WARNING: CBOSSV() method must first be called.

            Input: title - string
                folder path where images should be stored and/or image name
        """
        ##- Check whether CBOSSV() class has been called
        if not hasattr(self, "Y_output"):
            raise WrongCallToMethod("The class CBOSSV() must first be called.")

        plt.figure(figsize=(22,5))

        plt.subplot(1, 4, 1)
        plt.plot(self.nC_list,'-bo',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.nC_list)):
            if self.check_Toutput_admissible_list[iter] == 1:
                plt.plot(iter,self.nC_list[iter],'bo',markerfacecolor='b',lw=2)
        plt.plot(np.full((len(self.nC_list),1),self.threshold_nContainer),'g-',lw=2.0)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Number of cores', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'nContainers', fontsize=14)
        plt.ylim(bottom=0.0,top=50.0)
        grid(True)

        plt.subplot(1, 4, 2)
        plt.plot(self.Y_output,'-ro',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.Y_output)):
            if self.check_Toutput_admissible_list[iter] == 1:
                plt.plot(iter,self.Y_output[iter],'ro',markerfacecolor='r',lw=2)
        plt.plot(np.full((len(self.Y_output),1),self.threshold_Cost),'g-',lw=2.0)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Application cost', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'C(x)', fontsize=14)
        plt.ylim(bottom=0.98*self.C_lower_bound,top=1.01*self.C_upper_bound)
        grid(True)

        plt.subplot(1, 4, 3)
        plt.plot(self.T_output,'-mo',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.T_output)):
            if self.check_Toutput_admissible_list[iter] == 1:
                plt.plot(iter,self.T_output[iter],'mo',markerfacecolor='m',lw=2)
        plt.plot(np.full((len(self.T_output),1),self.threshold),'r--',lw=2.0)
        plt.plot(np.full((len(self.T_output),1),self.threshold_Time),'g-',lw=2.0)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Application completion time', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'T(x)', fontsize=14)
        upper_y_lim = max(self.T_upper_bound,self.threshold)
        lower_y_lim = min(self.T_lower_bound,self.threshold)
        plt.ylim(bottom=0.98*lower_y_lim,top=1.01*upper_y_lim)
        grid(True)

        plt.subplot(1, 4, 4)
        plt.plot(self.P_output,'-yo',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.P_output)):
            if self.check_Toutput_admissible_list[iter] == 1:
                plt.plot(iter,self.P_output[iter],'yo',markerfacecolor='y',lw=2)
        plt.plot(np.full((len(self.P_output),1),self.threshold_Price),'g-',lw=2.0)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Price per unit time', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'P(x)', fontsize=14)
        plt.ylim(bottom=0.0,top=50.0)
        grid(True)

        plt.savefig(title+'_BOSSV_convergence.svg',bbox_inches='tight')
        plt.savefig(title+'_BOSSV_convergence.png',bbox_inches='tight')
        if show:
            plt.show(block=True)


    def plot_regret(self,title="",show=False):
        """
            Plot number of cores, costs C(x), Times T(x) and price P(x) w.r.t. number of iterations of best admissible configuration.
                WARNING: CBOSSV() method must first be called.

            Input: title - string
                folder path where images should be stored and/or image name
        """
        ##- Check whether CBOSSV() class has been called
        if not hasattr(self, "best_Y"):
            raise WrongCallToMethod("The class CBOSSV() must first be called.")

        plt.figure(figsize=(22,5))

        plt.subplot(1, 4, 1)
        plt.plot(self.best_nC,'-bo',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.best_nC)):
            if self.check_Tbest_admissible_list[iter] == 1:
                plt.plot(iter,self.best_nC[iter],'bo',markerfacecolor='b',lw=2)
        plt.plot(np.full((len(self.best_nC),1),self.threshold_nContainer),'g-',lw=2.0)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Number of cores', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'nContainers', fontsize=14)
        plt.ylim(bottom=0.0,top=50.0)
        grid(True)

        plt.subplot(1, 4, 2)
        plt.plot(self.best_Y,'-ro',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.best_Y)):
            if self.check_Tbest_admissible_list[iter] == 1:
                plt.plot(iter,self.best_Y[iter],'ro',markerfacecolor='r',lw=2)
        plt.plot(np.full((len(self.best_Y),1),self.threshold_Cost),'g-',lw=2.0)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Application cost', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'C(x)', fontsize=14)
        plt.ylim(bottom=0.98*self.C_lower_bound,top=1.01*self.C_upper_bound)
        grid(True)

        plt.subplot(1, 4, 3)
        plt.plot(self.best_T,'-mo',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.best_T)):
            if self.check_Tbest_admissible_list[iter] == 1:
                plt.plot(iter,self.best_T[iter],'mo',markerfacecolor='m',lw=2)
        plt.plot(np.full((len(self.best_T),1),self.threshold),'r--',lw=2.0)
        plt.plot(np.full((len(self.best_T),1),self.threshold_Time),'g-',lw=2.0)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Application completion time', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'T(x)', fontsize=14)
        upper_y_lim = max(self.T_upper_bound,self.threshold)
        lower_y_lim = min(self.T_lower_bound,self.threshold)
        plt.ylim(bottom=0.98*lower_y_lim,top=1.01*upper_y_lim)
        grid(True)

        plt.subplot(1, 4, 4)
        plt.plot(self.best_P,'-yo',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.best_P)):
            if self.check_Tbest_admissible_list[iter] == 1:
                plt.plot(iter,self.best_P[iter],'yo',markerfacecolor='y',lw=2)
        plt.plot(np.full((len(self.P_output),1),self.threshold_Price),'g-',lw=2.0)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Price per unit time', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'P(x)', fontsize=14)
        plt.ylim(bottom=0.0,top=50.0)
        grid(True)

        plt.savefig(title+'_BOSSV_regret.svg',bbox_inches='tight')
        plt.savefig(title+'_BOSSV_regret.png',bbox_inches='tight')
        if show:
            plt.show(block=True)


    def plot_PREevaluation(self,title="",show=False):
        """
            Plot Y values, MAPE, MPE and model size w.r.t. number of iterations.
                WARNING: CBOSSV() method must first be called.

            Input: title - string
                folder path where images should be stored and/or image name
        """
        ##- Check whether CBOSSV() class has been called
        if not hasattr(self, "Y_output"):
            raise WrongCallToMethod("The class CBOSSV() must first be called.")

        plt.figure(figsize=(22,5))

        plt.subplot(1, 4, 1)
        plt.plot(self.nCPE_list,'-bo',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.nCPE_list)):
            if self.check_Toutput_admissible_list[iter] == 1:
                plt.plot(iter,self.nCPE_list[iter],'bo',markerfacecolor='b',lw=2)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Number of cores error (%)', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'nCPE', fontsize=14)
        # plt.ylim((0,100))
        grid(True)

        f_opt = self.threshold_Cost
        denf = 1.0 if abs(f_opt) < 1e-7 else abs(f_opt)
        f_err = [(abs(f_opt - f) / denf * 100) for f in self.Y_output]

        plt.subplot(1, 4, 2)
        plt.plot(f_err,'-ro',markerfacecolor='none',lw=2)
        for iter in range(0,len(f_err)):
            if self.check_Toutput_admissible_list[iter] == 1:
                plt.plot(iter,f_err[iter],'ro',markerfacecolor='r',lw=2)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Application cost error (%)', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'$|C(x_{i})-C_{opt}| / |C_{opt}|  \%$', fontsize=14)
        # plt.ylim((0,100))
        grid(True)

        # plt.subplot(1, 4, 2)
        # plt.plot(self.PRE_APE_list,'-s',label=u'APE')
        # plt.plot(self.PRE_MAPE_list,'-s',label=u'MAPE')
        # if hasattr(self,"stopping_iter"):
        #     plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        # plt.title(r'Prediction absolute error (%)', fontsize=18)
        # plt.xlabel(r'iteration', fontsize=16)
        # plt.ylabel(r'MAPE', fontsize=14)
        # # plt.ylim((0,100))
        # plt.legend(loc='upper right')
        # grid(True)

        plt.subplot(1, 4, 3)
        plt.plot(self.PRE_PE_list,'-s',label=u'PE')
        plt.plot(self.PRE_MPE_list,'-s',label=u'MPE')
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Prediction error (%)', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'MPE', fontsize=14)
        # plt.ylim((-100,100))
        plt.legend(loc='upper right')
        grid(True)

        plt.subplot(1, 4, 4)
        plt.plot(self.features_amount,'-gv',lw=2)
        # plt.plot(np.full((self.max_iter,1),self.d),'r--',lw=2)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Model size', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'# features', fontsize=14)
        plt.ylim((0,30))
        grid(True)

        plt.savefig(title+'_BOSSV_PREevaluation.svg',bbox_inches='tight')
        plt.savefig(title+'_BOSSV_PREevaluation.png',bbox_inches='tight')
        if show:
            plt.show(block=True)


    def plot_POSTevaluation(self,title="",show=False):
        """
            Plot Y values, MAPE, MPE and model size w.r.t. number of iterations.
                WARNING: CBOSSV() method must first be called.

            Input: title - string
                folder path where images should be stored and/or image name
        """
        ##- Check whether CBOSSV() class has been called
        if not hasattr(self, "Y_output"):
            raise WrongCallToMethod("The class CBOSSV() must first be called.")

        plt.figure(figsize=(22,5))

        plt.subplot(1, 4, 1)
        plt.plot(self.nCPE_list,'-bo',markerfacecolor='none',lw=2)
        for iter in range(0,len(self.nCPE_list)):
            if self.check_Toutput_admissible_list[iter] == 1:
                plt.plot(iter,self.nCPE_list[iter],'bo',markerfacecolor='b',lw=2)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Number of cores error (%)', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'nCPE', fontsize=14)
        # plt.ylim((0,100))
        grid(True)

        f_opt = self.threshold_Cost
        denf = 1.0 if abs(f_opt) < 1e-7 else abs(f_opt)
        f_err = [(abs(f_opt - f) / denf * 100) for f in self.Y_output]

        plt.subplot(1, 4, 2)
        plt.plot(f_err,'-ro',markerfacecolor='none',lw=2)
        for iter in range(0,len(f_err)):
            if self.check_Toutput_admissible_list[iter] == 1:
                plt.plot(iter,f_err[iter],'ro',markerfacecolor='r',lw=2)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Application cost error (%)', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'$|C(x_{i})-C_{opt}| / |C_{opt}|  \%$', fontsize=14)
        # plt.ylim((0,100))
        grid(True)

        # plt.subplot(1, 4, 2)
        # plt.plot(self.POST_APE_list,'-s',label=u'APE')
        # plt.plot(self.POST_MAPE_list,'-s',label=u'MAPE')
        # if hasattr(self,"stopping_iter"):
        #     plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        # plt.title(r'Prediction absolute error (%)', fontsize=18)
        # plt.xlabel(r'iteration', fontsize=16)
        # plt.ylabel(r'MAPE', fontsize=14)
        # # plt.ylim((0,100))
        # plt.legend(loc='upper right')
        # grid(True)

        plt.subplot(1, 4, 3)
        plt.plot(self.POST_PE_list,'-s',label=u'PE')
        plt.plot(self.POST_MPE_list,'-s',label=u'MPE')
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Prediction error (%)', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'MPE', fontsize=14)
        # plt.ylim((-100,100))
        plt.legend(loc='upper right')
        grid(True)

        plt.subplot(1, 4, 4)
        plt.plot(self.features_amount,'-gv',lw=2)
        # plt.plot(np.full((self.max_iter,1),self.d),'r--',lw=2)
        if hasattr(self,"stopping_iter"):
            plt.axvline(x=self.stopping_iter,color='k',lw=2.5)
        plt.title(r'Model size', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'# features', fontsize=14)
        plt.ylim((0,30))
        grid(True)

        plt.savefig(title+'_BOSSV_POSTevaluation.svg',bbox_inches='tight')
        plt.savefig(title+'_BOSSV_POSTevaluation.png',bbox_inches='tight')
        if show:
            plt.show(block=True)


    def plot_modelsize(self,title="",show=False):
        """
            Plot model size w.r.t. number of iterations.
                WARNING: CBOSSV() method must first be called.

            Input: title - string
                folder path where images should be stored and/or image name
        """
        ##- Check whether CBOSSV() class has been called
        if not hasattr(self, "features_amount"):
            raise WrongCallToMethod("The class CBOSSV() must first be called.")

        plt.figure(figsize=(5,4))
        plt.plot(self.features_amount,'-go')
        plt.plot(np.full((self.max_iter,1),self.d),'r--',lw=2)
        plt.title(r'Model size', fontsize=18)
        plt.xlabel(r'iteration', fontsize=16)
        plt.ylabel(r'# features', fontsize=14)
        grid(True)

        plt.savefig(title+'_BOSSV_modelsize.png')
        if show:
            plt.show(block=True)


def compute_error_on_config(minimum_configuration,selected_configuration,nContainer_index):
    """
        Compute percentage error on nContainers feature given selected_configuration w.r.t configuration that yields the minimum application completion time.

        Input: minimum_configuration - numpy 1D array
            configuration which yields the minimum completion time
        Input: selected_configuration - numpy 1D array
            configuration of which to compute percentage error w.r.t. minimum_configuration
        Input: nContainer_index - int
            index of the nContainers feature in the considered configurations
        Output: percentage error w.r.t. nContainers of minimum configuration - float
                nContainers value of minimum configuration - int
                nContainers value of selected configuration - int
    """
    ##- Find nContainer values
    min_nContainer = minimum_configuration[nContainer_index]

    selected_nContainer = selected_configuration[nContainer_index]

    return (selected_nContainer - min_nContainer) / min_nContainer * 100, min_nContainer, selected_nContainer


def plot_cumulative(no_data_cum_costs,no_data_stopping_iter,avec_data_cum_costs=None,avec_data_stopping_iter=None,title="",show=False):
    """
        Plot cumulative costs for all 4 variants of the algorithm.
    """
    ##- Plot with no additional data
    A, B, C, D = no_data_cum_costs
    iterA, iterB, iterC, iterD = no_data_stopping_iter
    y_max = max(max(A),max(B),max(C),max(D))

    plt.figure(figsize=(22,5))

    plt.subplot(1, 4, 1)
    plt.plot(A,'-bs',lw=2)
    plt.axvline(x=iterA,color='k',lw=2.5)
    plt.title(r'Cumulative costs of Variant A', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(B,'-gs',lw=2)
    plt.axvline(x=iterB,color='k',lw=2.5)
    plt.title(r'Cumulative costs of Variant B', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(C,'-rs',lw=2)
    plt.axvline(x=iterC,color='k',lw=2.5)
    plt.title(r'Cumulative costs of Variant C', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(D,'-ms',lw=2)
    plt.axvline(x=iterD,color='k',lw=2.5)
    plt.title(r'Cumulative costs of Variant D', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    grid(True)

    plt.savefig(title+'_BOSSV_CumCosts_nodata.png',bbox_inches='tight')
    plt.savefig(title+'_BOSSV_CumCosts_nodata.svg',bbox_inches='tight')
    if show:
        plt.show(block=True)


    plt.figure(figsize=(8,5))

    plt.plot(A[:iterA+1],'-bs',lw=2,label=u'Variant A')
    plt.plot(B[:iterB+1],'-go',lw=2,label=u'Variant B')
    plt.plot(C[:iterC+1],'-rv',lw=2,label=u'Variant C')
    plt.plot(D[:iterD+1],'-md',lw=2,label=u'Variant D')
    plt.title(r'Cumulative costs comparison', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    plt.legend(loc='upper left', fontsize=16)
    grid(True)

    plt.savefig(title+'_BOSSV_CumCosts_nodata_comparison.png',bbox_inches='tight')
    plt.savefig(title+'_BOSSV_CumCosts_nodata_comparison.svg',bbox_inches='tight')
    if show:
        plt.show(block=True)


    ##- Plot with additional data
    if avec_data_cum_costs is None or avec_data_stopping_iter is None:
        return

    A, B, C, D = avec_data_cum_costs
    iterA, iterB, iterC, iterD = avec_data_stopping_iter
    y_max = max(max(A),max(B),max(C),max(D))

    plt.figure(figsize=(22,5))

    plt.subplot(1, 4, 1)
    plt.plot(A,'-bs',lw=2)
    plt.axvline(x=iterA,color='k',lw=2.5)
    plt.title(r'Cumulative costs of Variant A', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(B,'-gs',lw=2)
    plt.axvline(x=iterB,color='k',lw=2.5)
    plt.title(r'Cumulative costs of Variant B', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(C,'-rs',lw=2)
    plt.axvline(x=iterC,color='k',lw=2.5)
    plt.title(r'Cumulative costs of Variant C', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(D,'-ms',lw=2)
    plt.axvline(x=iterD,color='k',lw=2.5)
    plt.title(r'Cumulative costs of Variant D', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    grid(True)

    plt.savefig(title+'_BOSSV_CumCosts_avecdata.png',bbox_inches='tight')
    plt.savefig(title+'_BOSSV_CumCosts_avecdata.svg',bbox_inches='tight')
    if show:
        plt.show(block=True)


    plt.figure(figsize=(8,5))

    plt.plot(A[:iterA+1],'-bs',lw=2,label=u'Variant A')
    plt.plot(B[:iterB+1],'-go',lw=2,label=u'Variant B')
    plt.plot(C[:iterC+1],'-rv',lw=2,label=u'Variant C')
    plt.plot(D[:iterD+1],'-md',lw=2,label=u'Variant D')
    plt.title(r'Cumulative costs comparison', fontsize=18)
    plt.xlabel(r'iteration', fontsize=16)
    plt.ylabel(r'Sum C(x)', fontsize=14)
    plt.ylim((0,y_max))
    plt.legend(loc='upper left', fontsize=16)
    grid(True)

    plt.savefig(title+'_BOSSV_CumCosts_avecdata_comparison.png',bbox_inches='tight')
    plt.savefig(title+'_BOSSV_CumCosts_avecdata_comparison.svg',bbox_inches='tight')
    if show:
        plt.show(block=True)
