## Script written by Jose` Y. Villafan.
## Last edited on 11/02/2020.
## BOSS-V algorithm.
# Copyright (c) 2020, the BOSS-V author (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE)

import warnings
import numpy as np
import matplotlib.pyplot as plt
from pylab import grid
from sklearn.gaussian_process import GaussianProcessRegressor as gp
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel as W
from diversipy import lhd_matrix, transform_spread_out
from mystic.scipy_optimize import fmin
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve#, cholesky
from scipy.stats import norm
from structure1_utils import WrongCallToMethod, min_dist, check_duplicate, compute_simpleregret
warnings.filterwarnings('ignore')

debug = False                   # Print information at debug level (e.g. parameters, condition numbers and determinants)


# 1. Define Bayesian optimization class, which uses Gaussian Processes to optimize the objective function
#####################################################################################################################
class CBayesOpt(object):
    """Class that implements Bayesian optimization with noise and constraints on the objective function.
    Its main methods are draw() which draws the next sample, and optimize() which iterates until a termination criterion is met.
    It exposes the plot_acquisition() method for plotting the acquisition function."""

    def __init__(self,af,GPkernel,greater_is_better=True,normalize_y=False,adaptive=True,noise=True,random_state=None,verbose=True):
        self.AF         = af                                        # Acquisition function criterion
        self.kernel     = GPkernel                                  # Gaussian Process kernel: sklearn.gaussian_process.kernels
        self.maximize   = greater_is_better                         # Boolean indicating whether to maximize the Objective function
        self.norm_y     = normalize_y                               # Boolean indicating whether to remove mean from evaluations
        self.adaptive   = adaptive                                  # Boolean indicating whether an adaptive method is chosen
        self.noise      = noise                                     # Boolean indicating whether evaluations are noisy
        self.rng_seed   = random_state                              # RandomState instance
        self.verbose    = verbose

        ##- Add Constant kernel to estimate variance magnitude and White kernel for noise level
        if isinstance(self.kernel,Matern) or isinstance(self.kernel,RBF):
            self.GPk = C(constant_value=2.0) * self.kernel
            if self.noise:
                self.GPk += W(noise_level=0.1,noise_level_bounds=(5e-6,1e-2))
        else:
            self.GPk = self.kernel

        ##- Define Gaussian Process regressor estimator
        self.alpha = 1e-10
        optimizer = 'fmin_l_bfgs_b'
        n_restarts_optimizer = 10
        self.gp = CGP(self.GPk,self.alpha,optimizer,n_restarts_optimizer,normalize_y=False,random_state=self.rng_seed)

        ##- Set acquisition function parameters
        self.set_AF_tradeoff(None)
        self.set_EIC_params(-20.0,20.0)

        ##- Print Gaussian Process initial parameters
        if debug:
            self.debug = True
            self.K_cond = []
            self.K_det = []
            if self.verbose:
                print("\n     Gaussian Process kernel parameters:\n{}".format(self.GPk.get_params()))


    def fit(self,X,Y,bounds=None):
        """
            Fit the internal Gaussian Process and compute prior mean and variance.

            Input: X - numpy 2D array
                matrix of observed data
            Input: Y - numpy 1D array
                vector of observed evaluations of the objective, Y = f(X)
            Input: bounds - numpy 2D array
                matrix of [lower_bound,upper_bound] where indices represent the dimension (starting from 0)
        """
        self.X          = X                                         # Matrix of known observations
        self.Y_mean     = Y.mean() if self.norm_y else 0.0          # Mean of evaluations of the objective f at X
        self.Y          = Y - self.Y_mean                           # Vector of evaluations of the objective f at X
        self.nObs       = self.X.shape[0]                           # Number of observations
        self.d          = self.X.shape[1]                           # Dimensionality of the observations

        ##- Reset Gaussian Process optimizer for non adaptive models, forward tuned hyperparameters
        if not self.adaptive:
            if not hasattr(self,"first_time_fit"):
                self.first_time_fit = True
            else:
                self.gp.optimizer = None
                self.gp.n_restarts_optimizer = 0
                self.gp.kernel.set_params( **self.gp.kernel_.get_params() )

        ##- Fit the Gaussian Process regressor
        self.gp.fit(self.X,self.Y)

        ##- Count the number of predicted observations
        # if not hasattr(self,"__counter"):
        #     self.__counter = 0
        # else:
        #     self.__counter += 1

        ##- Compute incumbent, only when required by the acquisition function
        if self.AF in AF_with_incumbent:
            self.incumbent = self.compute_incumbent(bounds)

        ##- Print Gaussian Process current/updated parameters
        if hasattr(self,"debug"):
            self.K = self.gp.L_.dot(self.gp.L_.T)
            self.K_cond.append(np.linalg.cond(self.K,p=2))
            self.K_det.append(np.linalg.det(self.K))
            if debug and self.verbose:
                print("     Gaussian Process updated kernel parameters:\n{}".format(self.gp.kernel_.get_params()))
                print("     Amount of jitter in the covariance computation = ", self.alpha)
                # print("\n     Gaussian Process covariance matrix:\n{}".format(self.K))
                print("     Gaussian Process covariance matrix condition number = ", self.K_cond[-1])
                print("     Gaussian Process covariance matrix determinant      = ", self.K_det[-1])

        return self


    def compute_incumbent(self,bounds=None):
        """
            Compute incumbent used by Improvement based acquisition functions.
            Takes into consideration whether observations are noisy or not.

            Output: incumbent - float
        """
        ##- Check whether fit() method has been called
        if not hasattr(self, "Y"):
            raise WrongCallToMethod("The method fit() must first be called on a non-empty vector Y.")

        ##- Check whether bounds.shape[0] is equal to the dimension of the space
        if bounds is not None:
            if not bounds.shape[0] == self.d:
                raise ValueError("Bounds must be a dx2 array where d is the dimension of observed points X.")

        ##- Compute incumbent as optimum of observations
        incumbent = np.max(self.Y) if self.maximize else np.min(self.Y)

        ##- Compute incumbent as optimum of conditional mean evaluated at available configurations
        # loss = np.array([self.__GPmu(self.X[i,:]) for i in range(0,self.nObs)]) if self.noise else self.Y
        # incumbent = np.max(loss) if self.maximize else np.min(loss)

        ##- Compute incumbent as optimum of conditional mean over the whole domain
        # if self.noise:
        #     if bounds is None:
        #         loss = np.array([self.__GPmu(self.X[i,:]) for i in range(0,self.nObs)])
        #         incumbent = np.max(loss) if self.maximize else np.min(loss)
        #     else:
        #         incumbent = self.optimize_GPmu(bounds)[1]
        # else:
        #     loss = self.Y
        #     incumbent = np.max(loss) if self.maximize else np.min(loss)

        if self.verbose:
            print("Incumbent = ",incumbent)

        return incumbent


    def set_AF_tradeoff(self,top=None):
        """
            Set exploration/exploitation trade-off parameter used by acquisition functions.
            For CB acquisition, its value is computed at every iteration in optimize() method; see [Srinivas et al. (2010)] for details.
            For other AFs it represents a percentage and its value should belong to the [0,1] interval.
            When high the algorithm does more exploration and converges more slowly (takes more iterations). When low then exploitation is performed.

            Input: top - float
                exploitation/exploration trade-off parameter
        """
        if top is None:
            self.top = 7.0 if self.AF == CB else 0.0
        else:
            self.top = top


    def set_EIC_params(self,lb=-20.0,ub=20.0,compute_mean=None,compute_sigma=None):
        """
            Set bound parameters and tranformations used by constrained EI (EIC) acquisition function.
            These parameters bound the value of the objective: lb < c(x) < ub.
            The tranformations compute the constraints mean and std from the objective's mean and std.

            Input: lb - float
                lower bound for constrained EI (EIC) acquisition function
            Input: ub - float
                upper bound for constrained EI (EIC) acquisition function
            Input: compute_mean - function
                function with signature f(mu,x) that transforms the objective mean 'mu' into the constraint function mean;
                if not given the constraint is computed on the objective function, and not on a function derived from it
            Input: compute_sigma - function
                function with signature g(sigma,x) that transforms the objective std 'sigma' into the constraint function std;
                if not given the constraint is computed on the objective function, and not on a function derived from it
        """
        ##- Define identity function
        def identity(posterior_stat,x):
            return posterior_stat

        if compute_mean is None:
            self.compute_EIC_mu = identity
        else:
            self.compute_EIC_mu = compute_mean

        if compute_sigma is None:
            self.compute_EIC_sigma = identity
        else:
            self.compute_EIC_sigma = compute_sigma

        self.EIC_lb = lb
        self.EIC_ub = ub


    def __GPmu(self,x):
        """
            Compute Gaussian Process conditional mean.

            Input: x - numpy 1D array
            Output: conditional mean evaluated at x - float
        """
        x = np.array(x).reshape(1,-1)
        return self.gp.predict(x,return_std=False)[0]


    def __invertedGPmu(self,x):
        x = np.array(x).reshape(1,-1)
        return -1 * self.gp.predict(x,return_std=False)[0]


    def optimize_GPmu(self,bounds):
        """
            Optimize the Gaussian Process conditional mean method. This method is called by optimize() and KG() acquisition function.

            Input: bounds - numpy 2D array
                matrix of [lower_bound,upper_bound] where indices represent the dimension (starting from 0)
            Output: optimum of Gaussian Process conditional mean - [numpy 1D array, float]
        """
        ##- Check whether fit() method has been called
        if not hasattr(self, "X"):
            raise WrongCallToMethod("The method fit() must first be called.")

        if self.maximize:
            fun = self.__invertedGPmu
        else:
            fun = self.__GPmu

        x_result = self.__minimize(fun,bounds,n_restarts=100*self.d,choose="lhd",composite=False)

        return x_result, self.__GPmu(x_result)


    def __minimize(self,fun,bounds,dfun=None,x0_list=None,n_restarts=None,choose="random",composite=False):
        """
            Minimize the function fun using mystic.scipy_optimize.fmin with 'Nelder-Mead' heuristic.
            If 1st order derivatives are given, it uses scipy.optimize.minimize with 'L-BFGS-B' method.
            Tolerates constrained optimization.
            Unconstrained optimization is achieved by passing x0_list and setting bounds=None.

            Input: fun - function
                function to be minimized
            Input: bounds - numpy 2D array
                matrix of [lower_bound,upper_bound] where indices represent the dimension (starting from 0)
            Input: dufn - function
                derivative of the function to be minimized
            Input: x0_list - numpy 1D array OR iterable of numpy 1D arrays
                optimization starting points
            Input: n_restarts - int
                number of points to be generated within provided bounds in case x0_list is not provided
            Input: choose - str
                choose = ["lhd","random"]
                if "lhd" n_restarts samples are drawn from a space-filling latin hypercube design
                if "random" n_restarts samples are drawn from a uniform distribution on the domain
            Input: composite - bool
                the local optimizer is started from every element of x0_list when True, only from the best otherwise (faster, less precise)
            Output: new observation - numpy 1D array
        """
        if n_restarts is None:
            n_restarts = 10 * self.d

        ##- Generate a list of points, if not given, using different methods
        if x0_list is None:
            if choose == "lhd":
                design = transform_spread_out(lhd_matrix(n_restarts,self.d))
                x0_list = bounds[:,0] + (bounds[:,1]-bounds[:,0])*design
            elif choose == "random":
                x0_list = np.random.uniform(bounds[:,0],bounds[:,1],size=(n_restarts,self.d))
                # x0_list = np.linspace(bounds[:,0],bounds[:,1],n_restarts)
            else:
                raise ValueError("Only 'random' and 'lhd' designs are supported as of now.")

        ##- Minimize the function
        if composite:
            ##- ..starting from every element in x0_list
            results_x, results_y = [], []
            for x0 in x0_list:
                if dfun is None:
                    res = fmin(fun,np.array(x0),bounds=bounds,xtol=0.001,ftol=0.001,disp=0)
                else:
                    res = minimize(fun=fun,jac=dfun,x0=x0.reshape(1,-1),method='L-BFGS-B',bounds=bounds).x
                results_x.append(res)
                results_y.append(fun(res))
            best_x = results_x[np.argmin(results_y)]
        else:
            ##- ..starting from the best element in x0_list
            y0_list = list(map(fun,x0_list))
            x0_best = x0_list[np.argmin(y0_list)]
            if dfun is None:
                best_x = fmin(fun,np.array(x0_best),bounds=bounds,xtol=0.001,ftol=0.001,disp=0)
            else:
                best_x = minimize(fun=fun,jac=dfun,x0=x0_best.reshape(1,-1),method='L-BFGS-B',bounds=bounds).x

        return best_x


    def predict(self,x,return_std=False):
        """
            Return Gaussian Process conditional mean and standard deviation.

            Input: x - numpy 1D array
            Input: return_std - bool
                sigma_N, square root of the conditional variance evaluated at x, is returned when True
            Output: mu_N, conditional mean evaluated at x - float
        """
        x = np.array(x).reshape(1,-1)
        return self.gp.predict(x,return_std)


    def predict_with_derivatives(self,x):
        """
            Return Gaussian Process conditional mean, standard deviation and their gradients evaluated at x.

            Input: x - numpy 1D array
            Output: conditional mean, standard deviation and their gradients evaluated at x - float, float, float, float
        """
        x = np.array(x).reshape(1,-1)
        mu, sigma = self.gp.predict(x,return_std=True)
        if sigma < 1e-10:
            sigma = 1e-10

        dmu_dx, dsigma_dx = self.gp.predict_derivatives(x)
        dsigma_dx = dsigma_dx / ( 2 * sigma )

        return mu, sigma, dmu_dx, dsigma_dx


    def sample_gp(self,x,random_state):
        """
            Draw samples from the gaussian process and evaluate at x.

            Input: x - numpy 1D array
            Input: random_state - RandomState instance
            Output: sample of gaussian process evaluated at x - float
        """
        x = np.array(x).reshape(1,-1)
        return self.gp.sample_y(x,n_samples=1,random_state=random_state)


    def __inverted_sample_gp(self,x,random_state):
        x = np.array(x).reshape(1,-1)
        return -1 * self.gp.sample_y(x,n_samples=1,random_state=random_state)


    def optimize_sample_gp(self,bounds,random_state=None):
        """
            Optimize the Gaussian Process sample. This method is called by MES() acquisition function.

            Input: bounds - numpy 2D array
                matrix of [lower_bound,upper_bound] where indices represent the dimension (starting from 0)
            Input: random_state - RandomState instance
            Output: optimum of Gaussian Process sample - [numpy 1D array, float]
        """
        ##- Check whether fit() method has been called
        if not hasattr(self, "X"):
            raise WrongCallToMethod("The method fit() must first be called.")

        ##- Optimize the gaussian process sample
        if self.maximize:
            fun = self.__inverted_sample_gp
        else:
            fun = self.sample_gp
        
        x0 = self.optimize_GPmu(bounds)[0]

        # x_result = fmin(fun,np.array(x0),args=(random_state),bounds=bounds,xtol=0.01,ftol=0.01,disp=0)
        x_result = minimize(fun=fun,x0=x0.reshape(1,-1),args=(random_state),method='L-BFGS-B',bounds=bounds).x

        return x_result, self.sample_gp(x_result,random_state)


    def acquisition(self,x):
        """
            Compute single-parameter acquisition function ready to be used with minimize() method, or for plotting.

            Input: x - numpy 1D array
            Output: acquisition function evaluated at x - float
        """
        x = np.array(x).reshape(1,-1)
        return self.AF(x,self)[0]


    def Dacquisition(self,x):
        """
            Compute single-parameter derivative of acquisition function.

            Input: x - numpy 1D array
            Output: derivative of acquisition function evaluated at x - float
        """
        x = np.array(x).reshape(1,-1)
        return self.AF(x,self)[1]


    def draw(self,bounds,x0_list=None,n_restarts=None,choose="random",use_derivatives=False,return_prob_bound=False):
        """
            Draw next observation in the one-step Bayesian method via optimization of the acquisition function.
            Tolerates constrained optimization.
            Unconstrained optimization is achieved by passing x0_list and setting bounds=None.
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
            Output: result - numpy 1D array
                        new observation to query
                    AF_value - float
                        acquisition function evaluated at result
        """
        ##- Check whether fit() method has been called
        if not hasattr(self, "X"):
            raise WrongCallToMethod("The method fit() must first be called on a non-empty matrix X.")

        ##- Check whether bounds.shape[0] is equal to the dimension of the space
        if not bounds.shape[0] == self.d:
            raise ValueError("Bounds must be a dx2 array where d is the dimension of observed points X.")

        ##- Initialize attribute required by acquisition functions in case derivatives are used
        if use_derivatives:
            ##- FIXME: make derivatives available again:
            ##- 1. composite kernel has no length scale
            ##- 2. implement correct check in CGP.dK_dr() method
            raise ValueError("Derivatives are not available with this version of the code. See the BACKUP from February 18 2020.")
        self.use_derivatives = use_derivatives

        ##- Optimize Gaussian Process conditional mean whose output is needed by KG() acquisition function
        if self.AF == KG:
            self.mu_N = self.optimize_GPmu(bounds)[1]

        ##- Create variable where a probability value defined by an acquisition function with constraints will be stored
        self.prob_bound = float('nan')

        ##- Maximize the aquisition function; a faster method is employed for KG() acquisition function
        if n_restarts is None:
            n_restarts = 100 * self.d

        if self.use_derivatives and self.AF in AF_with_derivatives:
            result = self.__minimize(self.acquisition,bounds,self.Dacquisition,x0_list,n_restarts,choose,True)
        else:
            if self.AF == KG:
                result = self.__minimize(self.acquisition,bounds,None,x0_list,n_restarts,choose,False)
            else:
                result = self.__minimize(self.acquisition,bounds,None,x0_list,n_restarts,choose,True)

        ##- Compute acquisition function value, and update the variable prob_bound when using EIC
        AF_value = -1 * self.acquisition(result)

        if return_prob_bound:
            return result, AF_value, self.prob_bound
        else:
            return result, AF_value


    def optimize(self,OF,bounds,X=None,n_samples=3,max_iter=100,design="lhd",use_derivatives=False,
                 max_random_jumps=None,tol=1e-10,plot_acquisition=False):
        """
            Uses the Gaussian Process defined at class initialization to optimize the objective function OF.
                WARNING: method fit() need not be called.

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
            Input: use_derivatives - bool
                use derivatives of conditional mean and variance during optimization when True
            Input: max_random_jumps - int
                maximum number of random jumps the algorithm can make when sampling too close to already sampled points; if None is given, a default value is chosen accordingly based on the dimensionality of OF
            Input: tol - float
                tolerance at which to make a random jump
            Input: plot_acquisition - bool
                plot the objective and the acquisition functions when True
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

        ##- Instantiate check_duplicate class
        is_duplicate = check_duplicate()

        ##- Define the list of sampled configurations x and observations; used by plot_convergence() method
        self.x_output_list , self.y_output_list = [], []

        ##- Iterative optimization of the objective function
        for i in range(0,max_iter):
            self.fit(X,Y)#,bounds)

            ##- Set trade-off parameter for "no regret" CB acquisition, see [Srinivas et al. (2010)] for details
            if self.AF == CB:
                delta = 0.01
                diff = max(bounds[:,1]-bounds[:,0])
                term1 = 2 * np.log( 2 * i**2 * np.pi**2 / (3 * delta) )
                term2 = 2 * self.d * np.log( i**2 * self.d * diff * np.sqrt( np.log( 4 * self.d / delta ) ) )
                top = np.sqrt( term1 + term2 )
                self.set_AF_tradeoff(top=top)
                if debug and self.verbose:
                    print("trade-off parameter = ",top)

            ##- Select the point where to evaluate OF next via maximization of AF
            new_Obs = self.draw(bounds=bounds,choose="lhd",use_derivatives=use_derivatives)[0]
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


    def plot_posterior_mean(self,bounds):
        """
            Plot objective and acquisition functions.
                WARNING: fit() method must first be called.

            Input: bounds - numpy 2D array
                bounds must be an array of length d of elements [lower_bound,upper_bound]
        """
        ##- Check whether fit() method has been called
        if not hasattr(self, "X"):
            raise WrongCallToMethod("The method fit() must first be called.")

        if self.d == 1:
            n_points = 100
            xc = np.linspace(bounds[:,0],bounds[:,1],n_points).reshape(n_points,1)
            mu = self.gp.predict(xc,return_std=False).reshape(n_points,1) + self.Y_mean

            plt.plot(xc,mu,'g-',lw=1.5)
            plt.plot(self.X,self.Y + self.Y_mean,'ko',markersize=5,label=u'Observations')
            plt.xlim(*bounds)
            plt.title(r'Conditional mean')
            plt.xlabel(r'x', fontsize=12)
            plt.ylabel(r'$\mu(x)$', fontsize=12)
            plt.legend(loc='upper left')
            plt.show(block=True)

        elif self.d == 2:
            n_points = 100
            X1 = np.linspace(bounds[0][0],bounds[0][1],n_points)
            X2 = np.linspace(bounds[1][0],bounds[1][1],n_points)
            grid_x1, grid_x2 = np.meshgrid(X1,X2)
            xc = np.hstack((grid_x1.reshape(n_points*n_points,1),grid_x2.reshape(n_points*n_points,1)))
            mu = self.gp.predict(xc,return_std=False).reshape(n_points,n_points) + self.Y_mean

            plt.contourf(X1,X2,mu,100)
            plt.colorbar()
            plt.plot(self.X[:,0],self.X[:,1],'ko',markersize=5,label=u'Observations')
            plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
            plt.title(r'Conditional mean')
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.legend(loc='upper left')
            plt.show(block=True)

        else:
            print("Plot of conditional mean is {}-dimensional. Thus, it will not be shown.".format(self.d))


    def plot_acquisition(self,bounds,OF=None):
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
            n_points = 100 if self.AF == KG else 1000
            xc = np.linspace(bounds[:,0],bounds[:,1],n_points).reshape(n_points,1)
            mu, sigma = self.gp.predict(xc,return_std=True)
            mu = mu.reshape(n_points,1) + self.Y_mean
            sigma = sigma.reshape(n_points,1)

            plt.figure(figsize=(10,7))

            plt.subplot(2, 1, 1)
            if OF is not None:
                yc = np.array(list(map(OF,xc))).reshape(n_points,1)
                plt.plot(xc,yc,'b-',lw=2)
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

        elif self.d == 2:
            n_points = 25 if self.AF == KG else 100
            X1 = np.linspace(bounds[0][0],bounds[0][1],n_points)
            X2 = np.linspace(bounds[1][0],bounds[1][1],n_points)
            grid_x1, grid_x2 = np.meshgrid(X1,X2)
            xc = np.hstack((grid_x1.reshape(n_points*n_points,1),grid_x2.reshape(n_points*n_points,1)))
            mu, sigma = self.gp.predict(xc,return_std=True)
            mu = mu.reshape(n_points,n_points) + self.Y_mean
            sigma = np.sqrt(sigma.reshape(n_points,n_points))
            yc = np.array(list(map(lambda x: -x, list(map(self.acquisition,xc)))))
            yc_normalized = (yc - min(yc))/(max(yc - min(yc))) # normalize acquisition
            yc_normalized = yc_normalized.reshape(n_points,n_points)

            plt.figure(figsize=(17,5))

            plt.subplot(1, 3, 1)
            plt.contourf(X1,X2,mu,100)
            plt.colorbar()
            plt.plot(self.X[:,0],self.X[:,1],'ko',markersize=5,label=u'Observations')
            plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
            plt.title(r'Conditional mean')
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')

            plt.subplot(1, 3, 2)
            plt.contourf(X1,X2,sigma,100)
            plt.colorbar()
            plt.plot(self.X[:,0],self.X[:,1],'ko',markersize=5,alpha=0.85,label=u'Observations')
            plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
            plt.title(r'Conditional variance')
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')

            plt.subplot(1, 3, 3)
            plt.contourf(X1,X2,yc_normalized,100)
            plt.colorbar()
            plt.plot(self.x_list[-1][0],self.x_list[-1][1],'ro',markersize=6)
            plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
            plt.title(r'Acquisition function')
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')

            plt.show(block=True)

        else:
            print("Plot of acquisition function is {}-dimensional. Thus, it will not be shown.".format(self.d))


    def plot_convergence(self,f_opt,title="",show=False):
        """
            Plot convergence graphs:
                - distance between observations at each iteration;
                - evaluation percentage error at each iteration;
                - simple regret at each iteration. See [Wang and Jegelka (2017); Chapter 2.3] for details.
                WARNING: optimize() method must first be called.

            Input: f_opt - float
                exact value of the objective evaluated at the optimum
            Input: title - string
                folder path where image should be stored and/or image name
            Input: show - bool
                images is saved in the same folder where execution happens, otherwise is also shown when True
        """
        ##- Check whether optimize() method has been called
        if not hasattr(self,"x_output_list"):
            return

        iter_dist = []
        for i in range(0,len(self.x_output_list)-1):
            iter_dist.append(np.linalg.norm((self.x_output_list[i+1]-self.x_output_list[i]),2))

        denf = 1.0 if abs(f_opt) < 1e-7 else abs(f_opt)
        f_err = [(abs(f_opt - f) / denf * 100) for f in self.y_output_list]

        r = compute_simpleregret(self.y_output_list,f_opt)

        plt.figure(figsize=(17,5))

        plt.subplot(1, 3, 1)
        plt.plot(iter_dist,'-ro')
        plt.title(r'Distance between iterates', fontsize=14)
        plt.xlabel(r'iteration', fontsize=14)
        plt.ylabel(r'$||x_{i+1}-x_{i}||$', fontsize=12)
        grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(f_err,'-o')
        plt.yscale('log')
        plt.title(r'Evaluations % error', fontsize=14)
        plt.xlabel(r'iteration', fontsize=14)
        plt.ylabel(r'$|f(x_{i})-f_{opt}| / |f_{opt}|  \%$', fontsize=12)
        grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(r,'-o')
        plt.yscale('log')
        plt.title(r'Simple regret', fontsize=14)
        plt.xlabel(r'iteration', fontsize=14)
        plt.ylabel(r'$min_{i} |f_{opt}-f(x_{i})|$', fontsize=12)
        grid(True)

        plt.savefig(title+'_BayesianOptimization_convergence.png')
        if show:
            plt.show(block=True)


    def plot_convergence_x(self,x_opt,title="",show=False):
        """
            Plot convergence graph of predicted observations.
                WARNING: optimize() method must first be called.

            Input: x_opt - float
                exact value of the optimum
            Input: title - string
                folder path where image should be stored and/or image name
            Input: show - bool
                images is saved in the same folder where execution happens, otherwise is also shown when True
        """
        ##- Check whether optimize() method has been called
        if not hasattr(self,"x_output_list"):
            return

        x_err = [np.linalg.norm((x-x_opt),2) for x in self.x_output_list]

        plt.figure()
        plt.plot(x_err,'-go')
        plt.yscale('log')
        plt.title(r'Configurations error', fontsize=14)
        plt.xlabel(r'iteration', fontsize=14)
        plt.ylabel(r'$|x_{i}-x_{opt}|$', fontsize=12)
        grid(True)

        plt.savefig(title+'_BayesianOptimization_convergence_configurations.png')
        if show:
            plt.show(block=True)


def plot_regret(f_list,f_opt,title="",show=False):
    """
        Plot simple regret graph, see [Wang and Jegelka (2017); Chapter 2.3] for details.

        Input: f_list - list
            list of evaluations of the objective obtained from optimization procedure, tipically CBayesOpt.y_output_list
        Input: f_opt - float
            exact value of the objective evaluated at the optimum
        Input: title - string
            folder path where image should be stored and/or image name
        Input: show - bool
            images is saved in the same folder where execution happens, otherwise is also shown when True
    """
    r = compute_simpleregret(f_list,f_opt)

    plt.plot(r,'-o')
    plt.yscale('log')
    plt.title(r'Simple regret', fontsize=14)
    plt.xlabel(r'iteration', fontsize=14)
    plt.ylabel(r'$min_{i} |f_{opt}-f(x_{i})|$', fontsize=12)
    grid(True)

    plt.savefig(title+'_BayesianOptimization_simpleregret.png')
    if show:
        plt.show(block=True)


# 2. Extend sklearn.gaussian_process.GaussianProcessRegressor to compute derivatives of conditional mean and variance
#####################################################################################################################
class CGP(gp):
    """Wrapper class of sklearn.gaussian_process.GaussianProcessRegressor,
    that implements conditional mean and variance derivatives of the Gaussian Process model.
    Supports only RBF and Matern (nu=1.5, nu=2.5) kernels derivatives. No composite kernels are supported yet."""

    def predict_derivatives(self,x):
        """
            Predict derivatives using the Gaussian process regression model.
                WARNING: This is not the same as computing the mean and variance of the derivative of the posterior function.

            Input: x - numpy 1D array
            Output: gradients of conditional mean and standard deviation evaluated at x - float, float
        """
        if not hasattr(self,"alpha_"):
            raise WrongCallToMethod("The method predict() must first be called.")

        ##- Store kernel length scale in an easily callable variable
        ##- TODO: Create method that finds the lengthscale of a composite kernel
        self.lengthscale = self.kernel_.length_scale

        ##- Compute inverse of prior covariance
        u = cho_solve((self.L_,True),np.diag(np.ones(self.alpha_.shape[0])))
        K_inv = cho_solve((self.L_.T,True),u)

        ##- Compute mean and variance gradients
        k = self.kernel_(x,self.X_train_)
        mean_jac = self._gradients_X(self.alpha_.T,x)
        temp_vect = -2.*np.dot(k,K_inv)
        var_jac = self._gradients_X(temp_vect,x)

        return mean_jac, var_jac


    def _gradients_X(self, dL_dK, x):
        """
            Given the derivative of the latent function wrt K, compute the derivative wrt x.

            Input: dL_dK - numpy 1D array
                derivative of either conditional mean or conditional variance wrt to the prior covariance
            Input: x - numpy 1D array
                point at which prediction is performed
            Output: derivative of the latent function wrt x - numpy 1D array
        """
        invdist = self._inv_dist(x, self.X_train_)
        dK_dr = self.dK_dr(x, self.X_train_)
        dL_dr = dK_dr * dL_dK
        tmp = invdist * dL_dr

        ##- TODO: Compute dKdr using sklearn.gaussian_process.kernels call() method
        # _, dKdr = self.kernel_(self.X_train_, eval_gradient=True)

        grad = np.empty(x.shape,dtype=np.float64)
        for q in range(x.shape[1]):
            np.sum(tmp * (x[:,q][:,None]-self.X_train_[:,q][None,:]), axis=1, out=grad[:,q])

        return grad / self.lengthscale**2


    def _inv_dist(self, X, X2):
        """
            Compute the elementwise inverse of the distance matrix.
        """
        dist = self._scaled_dist(X, X2).copy()

        ##- On the diagonal the distance is zero
        return 1./np.where(dist != 0., dist, np.inf)


    def _scaled_dist(self, X, X2):
        """
            Compute the elementwise scaled distance matrix between X and X2.
        """
        l = self.lengthscale
        dists = cdist(X / l, X2 / l, metric='euclidean')
        return dists


    def dK_dr(self, X, X2):
        """
            Compute the derivative of K wrt X going through X, for several kernel choices.
        """
        r = self._scaled_dist(X, X2)

        if isinstance(self.kernel,Matern):
            if self.kernel.nu == 1.5:
                dKdr = -3.*r * np.exp( -np.sqrt(3.) * r )
            elif self.kernel.nu == 2.5:
                dKdr = ( -5./3*r - 5.*np.sqrt(5.)/3*r**2 ) * np.exp( -np.sqrt(5.) * r )
            else:
                raise NotImplementedError("Derivatives have not been implemented for this kernel yet.")
        elif isinstance(self.kernel,RBF):
            dKdr = -r * np.exp( -0.5 * r**2 )
        else:
            raise NotImplementedError("Derivatives have not been implemented for this kernel yet.")

        return dKdr


# 3. Define non adaptive Bayesian model - OBSOLETE
#####################################################################################################################
# from sklearn.base import BaseEstimator

# class CNonAdaptiveGP(BaseEstimator):
#     """Class that implements a non adaptive Gaussian Process model."""

#     def __init__(self,GPkernel,alpha=1e-10,normalize_y=False):
#         self.kernel_    = GPkernel                                  # Gaussian Process kernel: sklearn.gaussian_process.kernels
#         self.alpha      = alpha                                     # Stability/noise variable for non adaptive models
#         self.norm_y     = normalize_y                               # Boolean indicating whether to remove mean from evaluations


#     def fit(self,X,Y):
#         """
#             Fit the Gaussian Process and compute its prior mean and variance.

#             Input: X - numpy 2D array
#                 matrix of observed data
#             Input: Y - numpy 1D array
#                 vector of observed evaluations of the objective, Y = f(X)
#             Output: instance of self
#         """
#         self.X          = np.copy(X)                                # Matrix of known observations
#         self.n          = self.X.shape[0]                           # Number of observations
#         self.d          = self.X.shape[1]                           # Dimensionality of the observations

#         self.Y_mean     = np.average(Y) if self.norm_y else 0.0     # Mean of evaluations of the objective f
#         self.Y          = np.copy(Y) - self.Y_mean                  # Vector of evaluations of the objective f at X

#         ##- Compute covariance matrix K and its lower Cholesky decomposition L_
#         K = self.kernel_(self.X)
#         K[np.diag_indices_from(K)] += self.alpha
#         try:
#             self.L_ = cholesky(K)
#         except np.linalg.LinAlgError as exc:
#             exc.args = ("The kernel, %s, is not returning a positive definite matrix. "
#                         "Try gradually increasing the 'alpha' parameter of the Gaussian Process regressor estimator."
#                         % self.kernel_,) + exc.args
#             raise

#         return self


#     def predict(self,x,return_std=False):
#         """
#             Return Gaussian Process conditional mean and/or variance.

#             Input: x - numpy 1D array
#             Input: return_std - bool
#                 square root of the conditional variance (sigma_N) evaluated at x is returned when True
#             Output: conditional mean evaluated at x - float
#         """
#         ##- Check whether fit() method has been called
#         if not hasattr(self, "X"):
#             raise ValueError("The method fit() must first be called on a non-empty matrix X.")

#         k = self.kernel_(x,self.X)
#         mu = k.dot(cho_solve((self.L_,True),self.Y)) + self.Y_mean

#         if return_std:
#             sigma2 = self.kernel_(x) - k.dot(cho_solve((self.L_, True),k.T))
#             return mu, np.sqrt(sigma2)
#         else:
#             return mu


#     def get_params(self,deep=True):
#         """
#             Return Gaussian Process parameters.

#             Input: deep - bool
#                 parameters for this estimator and contained subobjects that are estimators are returned when True
#             Output: Gaussian Process parameters - dictionary
#                 parameter names mapped to their values
#         """
#         params = dict()
#         for key in self._get_param_names():
#             try:
#                 value = getattr(self, key)
#             except AttributeError:
#                 value = None
#             if deep and hasattr(value, 'get_params'):
#                 deep_items = value.get_params().items()
#                 params.update((key + '__' + k, val) for k, val in deep_items)
#             params[key] = value
#         return params


# 4. Define acquisition functions used in the Bayesian optimization procedure
#####################################################################################################################
# An acquisition function (AF) yields the observation x where the objective f should be evaluated next:
#                                   x_(n+1) = arg max AF(x),
# based on the time-n update of the Gaussian Process distribution of f.
# They can be of three types:
#  - Optimistic       : CB
#  - Improvement-based: PI, EI, EIC, SEI
#  - Information-based: KG, ES, PES, MES


def CB(x,CBayesOpt_instance):
    """
        Confidence Bound acquisition function.

        Input: x - numpy 1D array
        Input: CBayesOpt_instance - instance of class CBayesOpt
        Output: Confidence Bound evaluated at x - float
    """
    x = np.array(x).reshape(1,-1)
    sign = (-1) ** (not CBayesOpt_instance.maximize)

    if CBayesOpt_instance.use_derivatives:
        mu, sigma, dmu_dx, dsigma_dx = CBayesOpt_instance.predict_with_derivatives(x)
    else:
        mu, sigma = CBayesOpt_instance.predict(x,return_std=True)

    cb = sign * mu + CBayesOpt_instance.top * sigma

    if CBayesOpt_instance.use_derivatives:
        dcb_dx = sign * dmu_dx + CBayesOpt_instance.top * dsigma_dx
        return -1 * cb, -1 * dcb_dx
    else:
        return [-1 * cb]


def PI(x,CBayesOpt_instance):
    """
        Probability of Improvement acquisition function.

        Input: x - numpy 1D array
        Input: CBayesOpt_instance - instance of class CBayesOpt
        Output: Probability of Improvement evaluated at x - float
    """
    x = np.array(x).reshape(1,-1)
    incumbent = CBayesOpt_instance.incumbent
    sign = (-1) ** (not CBayesOpt_instance.maximize)

    if CBayesOpt_instance.use_derivatives:
        mu, sigma, dmu_dx, dsigma_dx = CBayesOpt_instance.predict_with_derivatives(x)
    else:
        mu, sigma = CBayesOpt_instance.predict(x,return_std=True)

    if sigma < 1e-10:
        sigma = 1e-10

    delta = sign * (mu - incumbent) - CBayesOpt_instance.top
    u = delta / sigma
    pi = norm.cdf(u)

    if CBayesOpt_instance.use_derivatives:
        dpi_dx = (norm.pdf(u) / sigma) * (sign * dmu_dx - dsigma_dx * u)
        return -1 * pi, -1 * dpi_dx
    else:
        return [-1 * pi]


def EI(x,CBayesOpt_instance):
    """
        Expected Improvement acquisition function.

        Input: x - numpy 1D array
        Input: CBayesOpt_instance - instance of class CBayesOpt
        Output: Expected Improvement evaluated at x - float
    """
    x = np.array(x).reshape(1,-1)
    incumbent = CBayesOpt_instance.incumbent
    sign = (-1) ** (not CBayesOpt_instance.maximize)

    if CBayesOpt_instance.use_derivatives:
        mu, sigma, dmu_dx, dsigma_dx = CBayesOpt_instance.predict_with_derivatives(x)
    else:
        mu, sigma = CBayesOpt_instance.predict(x,return_std=True)

    if sigma < 1e-10:
        sigma = 1e-10

    delta = sign * (mu - incumbent) - CBayesOpt_instance.top
    u = delta / sigma
    ei = delta * norm.cdf(u) + sigma * norm.pdf(u)

    if CBayesOpt_instance.use_derivatives:
        dei_dx = dsigma_dx * norm.pdf(u) + sign * dmu_dx * norm.cdf(u)
        return -1 * ei, -1 * dei_dx
    else:
        return [-1 * ei]


def EIC(x,CBayesOpt_instance):
    """
        Constrained Expected Improvement acquisition function.
        This acquisition function tolerates constraints on f(x) of the form: lb < f(x) < ub.

        Input: x - numpy 1D array
        Input: CBayesOpt_instance - instance of class CBayesOpt
        Output: Expected Improvement evaluated at x - float
    """
    x = np.array(x).reshape(1,-1)
    incumbent = CBayesOpt_instance.incumbent
    sign = (-1) ** (not CBayesOpt_instance.maximize)
    mu, sigma = CBayesOpt_instance.predict(x,return_std=True)

    if sigma < 1e-10:
        sigma = 1e-10

    delta = sign * (mu - incumbent) - CBayesOpt_instance.top
    u = delta / sigma
    ei = delta * norm.cdf(u) + sigma * norm.pdf(u)

    ub = CBayesOpt_instance.EIC_ub - CBayesOpt_instance.Y_mean
    lb = CBayesOpt_instance.EIC_lb - CBayesOpt_instance.Y_mean
    c_mu = CBayesOpt_instance.compute_EIC_mu(mu,x)
    c_sigma = CBayesOpt_instance.compute_EIC_sigma(sigma,x)
    prob_ub = norm.cdf( (ub - c_mu) / c_sigma )
    prob_lb = norm.cdf( (lb - c_mu) / c_sigma )
    CBayesOpt_instance.prob_bound = prob_ub - prob_lb

    return [-1 * ei * CBayesOpt_instance.prob_bound]


def SEI(x,CBayesOpt_instance):
    """
        Scaled Expected Improvement acquisition function.

        Input: x - numpy 1D array
        Input: CBayesOpt_instance - instance of class CBayesOpt
        Output: Scaled Expected Improvement evaluated at x - float
    """
    x = np.array(x).reshape(1,-1)
    incumbent = CBayesOpt_instance.incumbent
    sign = (-1) ** (not CBayesOpt_instance.maximize)
    mu, sigma = CBayesOpt_instance.predict(x,return_std=True)

    if sigma < 1e-10:
        sigma = 1e-10

    delta = sign * (mu - incumbent) - CBayesOpt_instance.top
    u = delta / sigma
    num = delta * norm.cdf(u) + sigma * norm.pdf(u)
    den = sigma**2 * ((u**2 + 1) * norm.cdf(u) + u * norm.pdf(u)) - num**2
    sei = num / np.sqrt(den) if den > 0.0 else 0.0

    return [-1 * sei]


def KG(x,CBayesOpt_instance):
    """
        Knowledge Gradient acquisition function.

        Input: x - numpy 1D array
        Input: CBayesOpt_instance - instance of class CBayesOpt
        Output: Knowledge Gradient evaluated at x - float
    """
    x = np.array(x).reshape(1,-1)
    sign = (-1) ** (not CBayesOpt_instance.maximize)
    mu, sigma = CBayesOpt_instance.predict(x,return_std=True)

    J = 30 ##- They say 30 is a big number in statistics..
    kg = 0.0
    for _ in range(0,J):
        y_sim = mu + sigma * np.random.normal()
        y_sim = np.array(y_sim).reshape(1,1)

        X = np.vstack((CBayesOpt_instance.X,x))
        Y = np.vstack((CBayesOpt_instance.Y,y_sim))

        verbose = True if debug else False
        intermediate_model = CBayesOpt(CBayesOpt_instance.AF,
                                       CBayesOpt_instance.GPk,
                                       CBayesOpt_instance.maximize,
                                       CBayesOpt_instance.norm_y,
                                       CBayesOpt_instance.adaptive,
                                       CBayesOpt_instance.rng_seed,
                                       verbose)
        intermediate_model.fit(X,Y)
        mu_Nplus1 = intermediate_model.optimize_GPmu(CBayesOpt_instance.bounds)[1]

        delta = sign * (mu_Nplus1 - CBayesOpt_instance.mu_N)
        kg += delta / J

    return [-1 * kg]


# FIXME: REMOVE
def MES(x,CBayesOpt_instance):
    """
        Max-value Entropy Search acquisition function.
            WARNING: FIXME: does not work in all cases yet

        Input: x - numpy 1D array
        Input: CBayesOpt_instance - instance of class CBayesOpt
        Output: Max-value Entropy Search evaluated at x - float
    """
    raise NotImplementedError

    # x = np.array(x).reshape(1,-1)
    # mu, sigma = CBayesOpt_instance.predict(x,return_std=True)

    # if sigma < 1e-10:
    #     sigma = 1e-10

    # J = 5
    # seeds = [1,33,97,56431,5643161]
    # mes = 0.0
    # for j in range(0,J):
    #     seed = seeds[j]
    #     _, y = CBayesOpt_instance.optimize_sample_gp(CBayesOpt_instance.bounds,seed)
    #     gamma = (y - mu) / sigma
    #     alpha = gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - np.log(norm.cdf(gamma))
    #     mes += alpha / J

    # return [mes]


##- CBayesOpt class uses the following groups of acquisition functions in its methods.
##- Acquisition functions based on the improvement variable:
AF_with_incumbent = [PI,EI,EIC,SEI]

##- Acquisition functions that also compute derivatives of the acquisition function:
AF_with_derivatives = [CB,PI,EI]
