## Script written by Jose` Y. Villafan.
## Last edited on 26/05/2020.
## BOSS-V algorithm example run.

'''
This script will show you how to use BOSS-V main function.
BOSS-V is an optimization algorithm which integrates Bayesian Optimization
and Feature Selection techniques based on a Linear Regression (Ridge) model.

The code has been written with modularity in mind, and many of its features
can be further developed and extended, such as:
    - using Lasso instead of Ridge ==> see structure4_.py line 19
    - change what Feature Selection technique to use ==> change structure4_.py class CFeatureSelector __call__() method
    - change Guassian Process Regressor module ==> see structure3_.py line 848
    - re-implement derivatives of the acquisition function ==> see structure3_.py class CGP

However, the code has been used as is to obtain the experiments shown in the Msc Thesis by [Villafan, 2020].
Those experiments can be repeated by running:
    - runBO_average.py ==> average of 10 BO runs on benchmark functions from 1D to 10D
    - runBO_single.py ==> single BO run on benchmark functions from 1D to 10D
    - runBOSS_inter.py ==> optimization of Spark applications Query26 and K-means for varying data sizes
    - runBOSS_extra.py ==> optimization of Spark applications Query26 and K-means for biggest data size with/out additional data
See [Villafan, 2020] for details.

Thus, in this script I will show you how to write a main that uses BOSS-V to optimize a 2D benchmark function.

DISCLAIMER:
This script is part of an installer, which was built to preserve the dependencies of the algorithm.
Its dependencies are:
    - sklearn
    - matplotlib
    - pandas
    - diversipy
    - mystic
I do not guarantee it will work if any of its dependencies are changed or modified.
'''


import warnings, time, datetime, os
import numpy as np
warnings.filterwarnings('ignore')


# 1. 2D benchmark function definition: Branin function
#####################################################################################################################

##- Define objective function, domain bounds, solution:

##- Alternate from of Branin function from [Forrester et al. (2008)], "http://www.sfu.ca/~ssurjano/branin.html"
##- a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π)
a, b, c = 1.0, 5.1/(4*np.pi**2), 5/np.pi
r, s, t = 6.0, 10.0, 1/(8*np.pi)
def Branin(x):
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1 - t)*np.cos(x[0]) + s + 5*x[0]

bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])

sol = np.array([-3.68928526, 13.62998805])

print("Branin OF minimum: x = {}, f(x) = {}".format(sol,Branin(sol)))


# 2. Bayesian Optimization, for details see Algorithm 2.1 from [Villafan (2020)]
#####################################################################################################################

##- I showcase an adaptive Bayesian approach to global optimization.


##- Choose prior covariance:
from sklearn.gaussian_process.kernels import Matern, RBF
# kernel = RBF(length_scale=[1.0])                            # C^inf objective
kernel = Matern(length_scale=[1.0],nu=2.5)                  # C^2 objective
# kernel = Matern(length_scale=[1.0],nu=1.5)                  # C^1 objective


##- Choose acquisition function:
##- See [Villafan (2020)] for details, correct references, proofs of convergence
import structure3_bayesianoptimization as BO
# Acquisition_function = BO.CB                                # Confidence Bound
# Acquisition_function = BO.PI                                # Probability of Improvement
Acquisition_function = BO.EI                                # Expected Improvement
# Acquisition_function = BO.SEI                               # Scaled Expected Improvement
# Acquisition_function = BO.KG                                # Knowledge Gradient
# Acquisition_function = BO.EIC                               # Expected Improvement with Constraints


##- Instantiate BO class:
##- Toggle greater_is_better flag if the objective is to be maximized
##- Toggle noise flag if observations are corrupted by noise
model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,adaptive=True,noise=True,random_state=0)


##- Set acquisition function parameters:
##- Use only with EIC
# model.set_EIC_params(lb=-20.0,ub=-10.0)


##- Perform optimization:
# start = time.time()
# result = model.optimize(Branin,bounds,n_samples=5,max_iter=50,plot_acquisition=False)[1]
# end = time.time()
# print("Result: {}".format(result))
# print("Time: {} seconds".format(end-start))


##- Save images
# full_path = ''
# model.plot_convergence(f_opt=Branin(sol),title=full_path,show=True)
# model.plot_convergence_x(sol,title=full_path,show=True)


# 3. BOSS-V, for details see Algorithm 5.1 from [Villafan (2020)]
#####################################################################################################################

##- I showcase Variant A without Feature Selection (which is equivalent to the previous Algorithm 2.1)
##- For the adaptation of BOSS-V to the Spark applications, see structure_bossv.py


##- Choose prior covariance:
from sklearn.gaussian_process.kernels import Matern, RBF
# kernel = RBF(length_scale=[1.0])                            # C^inf objective
kernel = Matern(length_scale=[1.0],nu=2.5)                  # C^2 objective
# kernel = Matern(length_scale=[1.0],nu=1.5)                  # C^1 objective


##- Choose acquisition function:
##- See [Villafan (2020)] for details, correct references, proofs of convergence
import structure3_bayesianoptimization as BO
# Acquisition_function = BO.CB                                # Confidence Bound
# Acquisition_function = BO.PI                                # Probability of Improvement
Acquisition_function = BO.EI                                # Expected Improvement
# Acquisition_function = BO.SEI                               # Scaled Expected Improvement
# Acquisition_function = BO.KG                                # Knowledge Gradient
# Acquisition_function = BO.EIC                               # Expected Improvement with Constraints


##- Instantiate BOSS-V class:
##- Toggle floating flag is Forward Selection should be Floating variant
##- Toggle greater_is_better flag if the objective is to be maximized
##- noise flag is set to True, see structure5_.py line 24 to untoggle
import structure5_featureoptimization as FO
model = FO.CFeatureOpt(Acquisition_function,kernel,floating=True,greater_is_better=False,random_state=0)


##- Set acquisition function parameters:
##- Use only with EIC
# model.set_EIC_params(lb=-20.0,ub=-10.0)


##- Perform optimization:
##- Variant B is chosen by setting threshold = float(..)
##- Variant C is chosen by setting mix_models = True
##- Variant D is chosen by setting both parameters
start = time.time()
result = model.optimize(Branin,bounds,n_samples=5,max_iter=50,threshold=None,mix_models=False,initial_features="all",plot_acquisition=False)[1]
end = time.time()
print("Result: {}".format(result))
print("Time: {} seconds".format(end-start))


##- Save images
full_path = ''
model.plot_convergence(f_opt=Branin(sol),title=full_path,show=True)
model.plot_convergence_x(sol,title=full_path,show=True)
