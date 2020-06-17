## Script written by Jose` Y. Villafan.
## Last edited on 21/02/2020.
## BOSS-V algorithm.
# Copyright (c) 2020, the BOSS-V author (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from itertools import product
from diversipy import lhd_matrix, transform_spread_out
from structure1_utils import WrongCallToMethod, find_indices, check_duplicate
from structure3_bayesianoptimization import CBayesOpt
from structure4_featureselection import CFeatureSelector

debug = False                   # Print information at debug level (e.g. model parameters and CV score)


# 1. Define Feature optimization class, which uses Bayesian optimization and feature selection to select the best features
#####################################################################################################################
class CFeatureOpt(CBayesOpt):
    """Callable class, based on Bayesian Optimization with a set of sensible defaults, that also performs Sequential Forward Feature Selection (SFS).
    Exposes a method, refit_regression(), for fitting the internal regression model via Cross-validation."""

    def __init__(self,af,GPkernel,floating=True,target_features=None,greater_is_better=True,random_state=None,verbose=True):
        super().__init__(af,GPkernel,greater_is_better,False,True,True,random_state,verbose)
        self.sfs        = CFeatureSelector(floating=floating,target_features=target_features,random_state=random_state)
        self.k          = 2                                         # Exponent of the minima MD function


    def fit(self,X,Y,LR_Y=None,X_add=None,LR_Y_add=None,initial_features=[],mandatory_features=[]):
        """
            Perform feature selection and fit the Bayesian optimization Gaussian process model.

            Input: X - numpy 2D array
                matrix of observed data
            Input: Y - numpy 1D array
                vector of observed evaluations of the objective function, Y = f(X)
            Input: LR_Y - numpy 1D array
                vector of observed evaluations of the constraint function, LR_Y = c(X); LR_Y will be used instead of Y
            Input: X_add - numpy 2D array
                matrix of additional observed data to be used in the feature selection procedure
            Input: LR_Y_add - numpy 1D array
                vector of additional observed evaluations of the constraint function to be used in the feature selection procedure, either LR_Y_add = c(X_add) or LR_Y_add = f(X_add)
            Input: initial_features - list
                list of indexes (representing features) that are to be excluded from the search
                if initial_features == 'all' then no feature selection is performed
            Input: mandatory_features - list
                list of indexes (representing features) that are not to be removed and are automatically included in the initial_features list
        """
        ##- Store the original dimension of the space; done once to avoid overwritings
        if not hasattr(self,"full_dim"):
            self.full_dim = X.shape[1]

        ##- Store mandatory features used by the transformation functions of the EIC acquisition function
        self.mandatory_features = mandatory_features

        ##- Perform feature selection
        if LR_Y is not None:
            self.selected_features, score, self.fitted_lm = self.sfs(X,LR_Y,X_add,LR_Y_add,initial_features=initial_features,mandatory_features=self.mandatory_features)
        else:
            self.selected_features, score, self.fitted_lm = self.sfs(X,Y,X_add,LR_Y_add,initial_features=initial_features,mandatory_features=self.mandatory_features)
        if debug and self.verbose:
            print("Features: ", self.selected_features)
            print("Linear model intercept: ", self.fitted_lm.steps[-1][1].intercept_)
            print("Linear model betas: ", self.fitted_lm.steps[-1][1].coef_)
            print("CV score = ", score)

        ##- Fit the Gaussian process model
        super().fit(X[:,self.selected_features],Y)

        return self


    def refit_regression(self,X_train,Y_train):
        """
            Re-evaluate the internal regression model by performing grid search (GridSearchCV).

            Input: X_train - numpy 2D array
                matrix of observed data
            Input: Y_train - numpy 1D array
                vector of observed evaluations of the objective, Y_train = f(X_train)
            Output: score of the best model object - float,
                    best regression model object - sklearn.pipeline.Pipeline,
                    parameters of the best model object - dictionary
        """
        best_score, best_estimator, best_params = self.sfs.compute_CV(X_train,Y_train)
        return best_score, best_estimator, best_params


    def acquisition(self,x):
        """
            Compute single-parameter acquisition function ready to be used with minimize() method, or for plotting.

            Input: x - numpy 1D array
            Output: acquisition function evaluated at x - float
        """
        x = np.array(x).reshape(1,-1)
        af = self.AF(x,self)[0]
        coeff = self.__get_coeff(x)
        return coeff * af


    def __get_coeff(self,x):
        """
            Compute multiplicative coefficient for acquisition function at x using minima distribution.
            See [Luo (2019)] and [Villafan (2020); Chapter 4.2] for details.

            Input: x - numpy 1D array
            Output: coefficient for acquisition function evaluated at x - float
        """
        ##- Set coefficient to zero to prevent the optimization method from drawing this configuration
        ##- This may affect Confidence Bound acquisition in a bad way, because its values may also be negative:
        ##-     e.g., when minimizing a positive function ( f(x)>0 for every x ) such as Performance models
        if hasattr(self,"threshold"):
            pred_val = self.fitted_lm.predict(x)
            if not self.maximize and pred_val > self.threshold:
                return 0.0
            if self.maximize and pred_val < self.threshold:
                return 0.0

        ##- Compute coefficient using minima distribution of the linear regression model
        if hasattr(self,"mix_af_with_lm"):
            return self.__tau_k(x) / self.__I
        else:
            return 1.0


    def __tau_k(self,*x):
        """
            Compute the exponential-type Tau function of the linear regression model function.

            Input: *x - list
                list of elements that will be rebuilt into a numpy 1D array
            Output: Nascent minima distribution (MD) function evaluated at x - float
        """
        x = np.array(x).reshape(1,-1)
        r = self.fitted_lm.predict(x)
        r = (r - self.__LR_min) / (self.__LR_max - self.__LR_min)
        return np.exp(-self.k*r)


    def __get_LR_range(self,bounds):
        """
            Compute range of the function which is the numerator of the nascent minima distribution function.
            Since we are computing the minima MD of a linear function, the max and min are at the vertices.

            Input: bounds - numpy 2D array
                matrix of [lower_bound,upper_bound] where indices represent the dimension (starting from 0)
            Output: max, min - [float, float]
                maximum and minimum value of tau_k() evaluated at the boundaries
        """
        bounds = np.array(bounds).reshape(-1,2)
        vertices = list(product(*zip(bounds[:,0],bounds[:,1])))
        evals = [self.fitted_lm.predict(np.array(vertex).reshape(1,-1)) for vertex in vertices]
        return max(evals), min(evals)


    def set_EIC_params(self,lb,ub,compute_mean=None,compute_sigma=None):
        """
            Set bound parameters and tranformations used by constrained EI (EIC) acquisition function.
            These parameters bound the value of the objective: lb < c(x) < ub.
            The tranformations compute the constraints mean and std from the objective's mean and std.
            The mandatory_features will be used for the computation of the transformations.

            Input: lb - float
                lower bound for constrained EI (EIC) acquisition function
            Input: ub - float
                upper bound for constrained EI (EIC) acquisition function
            Input: compute_mean - function
                function with signature f(mu,x_1,x_2,..,x_m) that transforms the objective mean 'mu' into the constraint function mean, where x_1,x_2,..,x_m are the mandatory configuration features;
                if None the constraint is computed on the objective function, and not on a function derived from it
            Input: compute_sigma - function
                function with signature g(sigma,x_1,x_2,..,x_m) that transforms the objective std 'sigma' into the constraint function std, where x_1,x_2,..,x_m are the mandatory configuration features;
                if None the constraint is computed on the objective function, and not on a function derived from it
        """
        self.EIC_params_given = True

        self.lb = lb
        self.ub = ub
        self.compute_mean = compute_mean
        self.compute_sigma = compute_sigma


    def __set_EIC_params(self):
        """
            Set EIC params, if they were given.
        """
        if hasattr(self,"EIC_params_given"):
            ##- Compute index of mandatory configuration features among the ones selected by the feature selector
            idx = []
            for feature_idx in self.mandatory_features:
                idx.append(find_indices(self.selected_features,lambda  x_idx: x_idx == feature_idx))

            if debug:
                print("**** indices = {}".format(idx))

            ##- Redefine functions in order to change their signature
            def sub_compute_mean(mu,x):
                return self.compute_mean(mu,*x[:,idx])
            def sub_compute_sigma(sigma,x):
                return self.compute_sigma(sigma,*x[:,idx])

            ##- Set parameters to be used at the current iteration
            super().set_EIC_params(self.lb,self.ub,sub_compute_mean,sub_compute_sigma)


    def set_threshold(self,threshold):
        """
            Set threshold level coefficient for acquisition function at x to be used by Variants C or D of the algorithm.
            See [Villafan (2020)] for details.

            Input: threshold - float
        """
        self.threshold = threshold


    def draw(self,bounds,n_restarts=None,return_prob_bound=False,fill_result=False,mix_models=False):
        """
            Draw next observation via Bayesian optimization.
                WARNING: fit() method must be called first.

            Input: bounds - numpy 2D array
                matrix of [lower_bound,upper_bound] where indices represent the dimension (starting from 0)
            Input: x0_list - numpy 1D array OR iterable of numpy 1D arrays
                optimization starting point
            Input: n_restarts - int
                number of points to be generated within provided bounds, i.e. number of times to run the minimizer in minimize() method
            Input: choose - str
                choose = ["lhd","random"]
                if "lhd" n_restarts samples are drawn from a space-filling latin hypercube design
                if "random" n_restarts samples are drawn from a uniform distribution on the domain
            Input: use_derivatives - bool
                use derivatives of conditional mean and variance during optimization when True
            Input: return_prob_bound - bool
                if using a constrained acquisition function, the probability that the constraints are met at the result of this method is returned when True
            Input: fill_result - bool
                the resulting configuration's irrelevant dimensions are set 0 when True;
                in this way the configuration can be evaluated by the objective function if available
            Input: mix_models - bool
                the linear model is used to modify the prediction from the acquisition maximization when True
            Output: result - numpy 1D array
                        new observation to query
                    AF_value - float
                        acquisition function evaluated at result
        """
        ##- Set EIC params, if they were given
        self.__set_EIC_params()

        ##- Compute tau_k() function normalization constant: its integral; may take long to compute a d-dimensional integral
        ##- Another way is to normalize the values of tau_k() by normalizing the values of the LR predictive function
        if mix_models:
            self.mix_af_with_lm = mix_models
            ##- To normalize tau_k(), either compute its integral
            # self.__I = nquad(self.__tau_k,bounds[self.selected_features,:])[0]
            # self.__LR_max, self.__LR_min = 1.0, 0.0
            ##- Or normalize the LR values
            self.__I = 1.0
            self.__LR_max, self.__LR_min = self.__get_LR_range(bounds[self.selected_features,:])

        ##- Draw next configuration
        if return_prob_bound:
            result, AF_value, prob_bound = super().draw(bounds[self.selected_features,:],n_restarts=n_restarts,choose="lhd",return_prob_bound=True)
        else:
            result, AF_value = super().draw(bounds[self.selected_features,:],n_restarts=n_restarts,choose="lhd",return_prob_bound=False)

        ##- Fill irrelevant dimensions of resulting observation with 0's
        if fill_result:
            temp = np.full((1,self.full_dim),0.0)
            temp[:,self.selected_features] = result
            result = temp.reshape(1,-1)[0]

        if return_prob_bound:
            return result, AF_value, prob_bound
        else:
            return result, AF_value


    def __call__(self,X,Y,bounds,initial_features=[],mandatory_features=[],max_iter_BO=None,mix_models=False,
                 return_prob_bound=False,LR_Y=None,X_add=None,LR_Y_add=None):
        """
            Callable method that performs one iteration of Feature selection and one of Bayesian optimization.

            Input: X - numpy 2D array
                matrix of observed data
            Input: Y - numpy 1D array
                vector of observed evaluations of the objective, Y = f(X)
            Input: bounds - numpy 2D array
                matrix of [lower_bound,upper_bound] where indices represent the dimension (starting from 0)
            Input: initial_features - list
                list of indexes (representing features) that are to be excluded from the search
                if initial_features == 'all' then no feature selection is performed
            Input: mandatory_features - list
                list of indexes (representing features) that are not to be removed and are automatically included in the initial_features list
            Input: max_iter_BO - int
                maximization of acquisition function stopping criterion based on maximum number of iterations
            Input: mix_models - bool
                the linear model is used to modify the prediction from the acquisition maximization when True
            Input: return_prob_bound - bool
                if using a constrained acquisition function, the probability that the constraints are met at the result of this method is returned when True
            Input: LR_Y - numpy 1D array
                vector of observed evaluations of the constraint function, LR_Y = c(X); LR_Y will be used instead of Y
            Input: X_add - numpy 2D array
                matrix of additional observed data to be used in the feature selection procedure
            Input: LR_Y_add - numpy 1D array
                vector of additional observed evaluations of the constraint function to be used in the feature selection procedure, either LR_Y_add = c(X_add) or LR_Y_add = f(X_add)
            Output: result, AF_value, selected_features, fitted_lm - [numpy 1D array, list, sklearn.linear_model]
                result is the suggested configuration
                AF_value is the evaluation of the acquisition function at suggested configuration
                selected_features are the relevant features selected by the provided SFS
                fitted_lm is the linear model that has been fitted on the subset of X with features selected_features
        """
        ##- Feature selection via fit() method
        self.fit(X,Y,LR_Y,X_add,LR_Y_add,initial_features,mandatory_features)

        ##- Bayesian optimization via draw() method
        result = self.draw(bounds,n_restarts=max_iter_BO,return_prob_bound=return_prob_bound,mix_models=mix_models)

        if return_prob_bound:
            return [result[0], result[1], self.selected_features, self.fitted_lm, result[2]]
        else:
            return [result[0], result[1], self.selected_features, self.fitted_lm]


    def optimize(self,OF,bounds,X=None,n_samples=3,max_iter=100,design="lhd",
                 threshold=None,mix_models=True,initial_features=[],
                 max_random_jumps=None,tol=1e-10,plot_acquisition=False):
        """
            Iterative algorithm that performs feature selection and Bayesian optimization.
                WARNING: Optimization happens on the boundary of the domains:
                         the values of the irrelevant features are set to 0.
                         See [Villafan (2020); Chapter 4.2] for details.

            Input: OF - function
                objective function
            Input: bounds - numpy 2D array
                bounds must be an array of length d of elements [lower_bound,upper_bound]
            Input: X - numpy 2D array
                matrix of known observations
            Input: n_samples - int
                number of starting observations in case X is not given
            Input: max_iter - int
                stopping criterion based on maximum number of iterations
            Input: design - str
                design = ["lhd","random"]
                if "lhd" n_samples are drawn from a space-filling latin hypercube design
                if "random" n_samples are drawn from a uniform distribution on the domain
            Input: threshold - float
                objective threshold value; this value is enforced on all acquisition functions so that their value is 0 at points where the threshold is not met
            Input: mix_models - bool
                the linear model is used to modify the prediction from the acquisition maximization when True
            Input: initial_features - list
                list of indexes (representing features) that are to be excluded from the search
                if initial_features == 'all' then no feature selection is performed
            Input: max_random_jumps - int
                maximum number of random jumps the algorithm can make when sampling too close to already sampled points; if None is given, a default value is chosen accordingly based on the dimensionality of OF
            Input: tol - float
                tolerance at which to make a random jump
            Output: result of the optimization procedure: observation and its evaluation - [numpy 1D array, float]
        """
        ##- Define the matrix of starting observations
        self.x_list, self.y_list = [], []
        if X is None:
            if design == "lhd":
                grid = transform_spread_out(lhd_matrix(n_samples,bounds.shape[0]))
                x0_list = bounds[:,0] + (bounds[:,1]-bounds[:,0])*grid
            elif design == "random":
                x0_list = np.random.uniform(bounds[:, 0],bounds[:, 1],(n_samples,bounds.shape[0]))
            else:
                raise ValueError("Only 'random' and 'lhd' designs are supported as of now.")
            for x in x0_list:
                self.x_list.append(x)
                self.y_list.append(OF(x))
        else:
            for x in X:
                self.x_list.append(x)
                self.y_list.append(OF(x))

        X = np.array(self.x_list).reshape(-1,bounds.shape[0])
        Y = np.array(self.y_list).reshape(-1,1)

        ##- Set parameters
        if threshold is not None:
            self.set_threshold(threshold)

        ##- Instantiate check_duplicate class
        is_duplicate = check_duplicate()

        ##- Define the list of sampled configurations x and observations; used by plot_convergence() method
        self.x_output_list , self.y_output_list = [], []

        ##- Iterative optimization of the objective function
        self.selected_features = initial_features
        for i in range(0,max_iter):
            ##- selected_features is updated everytime fit() method is called
            self.fit(X,Y,initial_features=self.selected_features)

            ##- Select the point where to evaluate OF next via maximization of AF
            new_Obs = self.draw(bounds,fill_result=True,mix_models=mix_models)[0]
            eval_newObs = OF(new_Obs)
            self.x_output_list.append(new_Obs)
            self.y_output_list.append(eval_newObs)

            ##- Select another point randomly in case of duplicate and print to screen
            new_Obs, duplicate = is_duplicate(new_Obs,X,bounds,max_random_jumps,tol)
            if duplicate and self.verbose:
                print("Iteration {}: configuration chosen ramdomly!".format(i+1))
                eval_newObs = OF(new_Obs)

            ##- Print to screen
            if self.verbose:
                print("Iteration {}: x = {}, f(x) = {}".format(i+1,new_Obs,eval_newObs))

            ##- Add the newly acquired observation to appropriate lists for plotting
            self.x_list.append(new_Obs)
            self.y_list.append(eval_newObs)
            X = np.array(self.x_list).reshape(-1,bounds.shape[0])
            Y = np.array(self.y_list).reshape(-1,1)

            ##- Plot acquisition function
            if plot_acquisition:
                self.plot_acquisition(bounds,OF)

        best_index = np.argmax(self.y_list) if self.maximize else np.argmin(self.y_list)
        best_x = self.x_list[best_index]
        best_y = self.y_list[best_index]

        return best_x, best_y


    def plot_acquisition(self,bounds,OF=None,plot_regression=True):
        """
            Plot objective and acquisition functions.
                WARNING: fit() method must first be called.

            Input: bounds - numpy 2D array
                bounds must be an array of length d of elements [lower_bound,upper_bound]
            Input: OF - function
                objective function
        """
        ##- Check whether fit() method has been called
        if not hasattr(self, "X"):
            raise WrongCallToMethod("The method fit() must first be called.")

        if self.d == 1:
            n_points = 1000
            xc = np.linspace(bounds[:,0],bounds[:,1],n_points).reshape(n_points,1)
            mu, sigma = self.gp.predict(xc,return_std=True)
            mu = mu.reshape(n_points,1) + self.Y_mean
            sigma = sigma.reshape(n_points,1)

            plt.figure(figsize=(10,7))

            plt.subplot(2, 1, 1)
            if OF is not None:
                yc = np.array(list(map(OF,xc))).reshape(n_points,1)
                plt.plot(xc,yc,'b-',lw=2)
            if plot_regression:
                lr_pred = self.fitted_lm.predict(xc)
                plt.plot(xc,lr_pred,'k-',lw=2)
            plt.plot(self.X,self.Y + self.Y_mean,'ko',markersize=5,label=u'Observations')
            plt.plot(xc,mu,'g-',lw=1.5,label=u'Posterior mean')
            plt.fill(np.concatenate([xc,xc[::-1]]),\
                     np.concatenate([mu - 1.96 * sigma, (mu + 1.96 * sigma)[::-1]]),\
                     alpha=0.5,fc='g',ec='None',label='95% P.I.')
            plt.axvline(x=self.x_list[-1],color='r',lw=2.5)
            plt.xlim(*bounds)
            plt.ylabel(r'Objective function', fontsize=16)
            plt.legend(loc='upper left', fontsize=16)

            plt.subplot(2, 1, 2)
            yc = np.array(list(map(lambda x: -x, list(map(self.acquisition,xc)))))
            yc_normalized = (yc - min(yc))/(max(yc - min(yc))) # normalize acquisition
            yc_normalized = yc_normalized.reshape(n_points,1)
            plt.plot(xc,yc_normalized,'r-',lw=1.5)
            plt.axvline(x=self.x_list[-1],color='r',lw=2.5)
            plt.xlim(*bounds)
            plt.xlabel(r'x', fontsize=16)
            plt.ylabel(r'Acquisition function', fontsize=16)

            plt.show(block=True)

        else:
            print("Plot of acquisition function is {}-dimensional. Thus, it will not be shown.".format(self.d))


    # def plot_acquisition(self,bounds,OF=None,plot_regression=True):
    #     """
    #         Plot objective and acquisition functions.
    #             WARNING: fit() method must first be called.

    #         Input: bounds - numpy 2D array
    #             bounds must be an array of length d of elements [lower_bound,upper_bound]
    #         Input: OF - function
    #             objective function
    #     """
    #     ##- Check whether fit() method has been called
    #     if not hasattr(self, "X"):
    #         raise WrongCallToMethod("The method fit() must first be called.")

    #     if self.d == 1:
    #         n_points = 1000
    #         xc = np.linspace(bounds[:,0],bounds[:,1],n_points).reshape(n_points,1)
    #         mu, sigma = self.gp.predict(xc,return_std=True)
    #         mu = mu.reshape(n_points,1) + self.Y_mean
    #         sigma = sigma.reshape(n_points,1)

    #         plt.figure(figsize=(10,7))

    #         plt.subplot(2, 1, 1)
    #         if OF is not None:
    #             yc = np.array(list(map(OF,xc))).reshape(n_points,1)
    #             plt.plot(xc,yc,'b-',lw=2)
    #         # if plot_regression:
    #         #     lr_pred = self.fitted_lm.predict(xc)
    #         #     plt.plot(xc,lr_pred,'k-',lw=2)
    #         plt.plot(self.X,self.Y + self.Y_mean,'ko',markersize=5,label=u'Observations')
    #         plt.plot(xc,mu,'g-',lw=1.5,label=u'Posterior mean')
    #         plt.fill(np.concatenate([xc,xc[::-1]]),\
    #                  np.concatenate([mu - 1.96 * sigma, (mu + 1.96 * sigma)[::-1]]),\
    #                  alpha=0.5,fc='g',ec='None',label='95% P.I.')
    #         # plt.axvline(x=self.x_list[-1],color='r',lw=2.5)
    #         plt.xlim(*bounds)
    #         plt.ylabel(r'Objective function', fontsize=16)
    #         plt.legend(loc='upper left', fontsize=16)

    #         plt.subplot(2, 1, 2)
    #         yc = np.array(list(map(lambda x: -x, list(map(self.acquisition,xc)))))
    #         yc_normalized = (yc - min(yc))/(max(yc - min(yc))) # normalize acquisition
    #         yc_normalized = yc_normalized.reshape(n_points,1)

    #         from structure3_bayesianoptimization import CB, SEI
    #         yc2, yc3 = [], []
    #         for xxx in xc:
    #             yc2.append(-1.0 * CB(xxx,self)[0])
    #             yc3.append(-1.0 * SEI(xxx,self)[0])
    #         yc2 = np.array(yc2).reshape(n_points,1)
    #         yc3 = np.array(yc3).reshape(n_points,1)
    #         yc2_normalized = (yc2 - min(yc2))/(max(yc2 - min(yc2))) # normalize acquisition
    #         yc3_normalized = (yc3 - min(yc3))/(max(yc3 - min(yc3))) # normalize acquisition

    #         plt.plot(xc,yc_normalized,'r-',lw=1.5)
    #         # plt.axvline(x=self.x_list[-1],color='r',lw=2.5)
    #         plt.plot(xc,yc2_normalized,'b-',lw=1.5)
    #         plt.plot(xc,yc3_normalized,'g-',lw=1.5)
    #         plt.xlim(*bounds)
    #         plt.xlabel(r'x', fontsize=16)
    #         plt.ylabel(r'Acquisition functions', fontsize=16)

    #         plt.show(block=True)

    #     else:
    #         print("Plot of acquisition function is {}-dimensional. Thus, it will not be shown.".format(self.d))
