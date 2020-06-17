## Script written by Jose` Y. Villafan.
## Last edited on 11/02/2020.
## BOSS-V algorithm.

import warnings, time, datetime, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern, RBF
import structure3_bayesianoptimization as BO
warnings.filterwarnings('ignore')

start = time.time()


# 0. Settings
#####################################################################################################################
seed = 0
np.random.seed(seed)

# test_images_folder_path = '../Bebop images BACKUP 2020-04-16/BO_single/'
test_images_folder_path = 'BO_Ackley/' # for experiments in Ardagna's servers

BO.debug = False

Acquisition_function = BO.EI
AF_name = "EI"

K_dict = {"RBF": RBF(length_scale=[1.0]),
          "Matern52": Matern(length_scale=[1.0],nu=2.5)}
K_key = "Matern52"
kernel = K_dict[K_key]

noise = True
max_iter = 1000
show_plots = False


# 1. 1D test
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution and starting samples:
x0 = 0.33
def OF1(x):
    # return (x[0]-x0)**2 - 10
    # return np.sqrt(abs(x[0]-x0))+0.5
    return x[0]**4-2*x[0]**3-3*(x[0]-1)**2

# bounds = np.array([[-4.0, 4.0]])
bounds = np.array([[-2.0, 2.0]])

# sol = np.array([x0])
sol = np.array([-0.76328])

# X = np.array([[-5.0],[3.0],[6.0]])
X = np.array([[-1.5],[1.0]])#,[2.2]])


##- Optimize OF:
# model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,random_state=seed)
# model.set_AF_tradeoff(None)
# model.set_EIC_params(lb=-20.0,ub=0.0)
# result = model.optimize(OF1,bounds,X,max_iter=max_iter,plot_acquisition=show_plots)[1]
# print("Result: {}".format(result))

# full_path = test_images_folder_path + '1D_' + AF_name + '_' + K_key
# model.plot_convergence(f_opt=OF1(sol),title=full_path,show=True)
# model.plot_convergence_x(sol,title=full_path,show=True)


# 2. 1D test: Forrester function
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution and starting samples:
##- Forrester function from [Forrester et al. (2008)], "http://www.sfu.ca/~ssurjano/forretal08.html"
def ForresterOF(x):
    return (6*x[0]-2)**2 * np.sin(12*x[0]-4)

bounds = np.array([[0.0, 1.0]])

x0 = np.array([0.8])
from scipy.optimize import minimize
res = minimize(fun=ForresterOF,x0=x0.reshape(1,-1),method='L-BFGS-B',bounds=bounds)
print("Forrester OF minimum: x = {}, f(x) = {}".format(res.x,res.fun))
sol = res.x


##- PI tends to exploit around the best point, more than others AFs.
##- Thus, it needs a good starting configuration, either denser or "luckier".
if Acquisition_function in [BO.PI,BO.SEI,BO.MES]:
    X = np.array([[0.1],[0.5],[0.8]])
else:
    X = np.array([[0.1],[0.5],[0.9]])


##- Optimize OF:
# model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,adaptive=True,noise=True,random_state=seed)
# model.set_AF_tradeoff(None)
# model.set_EIC_params(lb=-20.0,ub=-3.0)
# result = model.optimize(ForresterOF,bounds,X,max_iter=max_iter,plot_acquisition=show_plots)[1]
# print("Result: {}".format(result))

# full_path = test_images_folder_path + 'TESTING_1D_Forrester_' + AF_name + '_' + K_key
# model.plot_convergence(f_opt=ForresterOF(sol),title=full_path,show=True)
# model.plot_convergence_x(sol,title=full_path,show=True)


# 3. 2D test: Branin function
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution:
##- Alternate from of Branin function from [Forrester et al. (2008)], "http://www.sfu.ca/~ssurjano/branin.html"
##- a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π)
a, b, c = 1.0, 5.1/(4*np.pi**2), 5/np.pi
r, s, t = 6.0, 10.0, 1/(8*np.pi)
def BraninOF(x):
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1 - t)*np.cos(x[0]) + s + 5*x[0]

bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])

x0 = np.array([-np.pi,12.275])      ##- global minimum
# x0 = np.array([np.pi,2.275])        ##- local minimum
# x0 = np.array([9.42478,2.475])      ##- local minimum
from scipy.optimize import minimize
res = minimize(fun=BraninOF,x0=x0.reshape(1,-1),method='L-BFGS-B',bounds=np.array([[-5.0, 10.0], [0.0, 15.0]]))
print("Branin OF minimum: x = {}, f(x) = {}".format(res.x,res.fun))
sol = res.x


##- Optimize OF:
# model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,adaptive=True,noise=True,random_state=seed)
# model.set_AF_tradeoff(None)
# model.set_EIC_params(lb=-20.0,ub=-10.0)
# result = model.optimize(BraninOF,bounds,n_samples=5,max_iter=max_iter,plot_acquisition=show_plots)[1]
# print("Result: {}".format(result))

# full_path = test_images_folder_path + 'TESTING_2D_Branin' + AF_name + '_' + K_key
# model.plot_convergence(f_opt=BraninOF(sol),title=full_path,show=True)
# model.plot_convergence_x(sol,title=full_path,show=True)


# 4. 2D test: Rosenbrock function
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution:
##- Rosenbrock function, "http://www.sfu.ca/~ssurjano/rosen.html"
a = 1.0
def RosenbrockOF(x):
    b = 100
    return (a-x[0])**2 + b*((x[1]-x[0]**2)**2)

bounds = np.array([[-2.048, 2.048],[-2.048, 2.048]])

sol = np.array([a, a**2])


##- Optimize OF:
# model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,random_state=seed)
# model.set_AF_tradeoff(None)
# model.set_EIC_params(lb=-20.0,ub=20.0)
# result = model.optimize(RosenbrockOF,bounds,n_samples=21,max_iter=200,plot_acquisition=show_plots)[1]
# print("Result: {}".format(result))

# full_path = test_images_folder_path + '2D_Rosenbrock_' + AF_name + '_' + K_key
# model.plot_convergence(f_opt=RosenbrockOF(sol),title=full_path,show=True)
# model.plot_convergence_x(sol,title=full_path,show=True)


# 5. 4D test
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution:
def OF4(x):
    return (x[0]-2)**2 + (x[1]-3)**2 + (x[2]-0)**2 + (x[3]-1)**2 - 10

bounds = np.full((4,2),[-1., 5.])

sol = np.array([2., 3., 0., 1.])


##- Optimize OF:
# model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,random_state=seed)
# model.set_AF_tradeoff(None)
# model.set_EIC_params(lb=-20.0,ub=20.0)
# result = model.optimize(OF4,bounds,n_samples=6,max_iter=max_iter)[1]
# print("Result: {}".format(result))

# full_path = test_images_folder_path + '4D_' + AF_name + '_' + K_key
# model.plot_convergence(f_opt=OF4(sol),title=full_path,show=True)
# model.plot_convergence_x(sol,title=full_path,show=True)


# 6. 6D test: Hartmann function
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution:
##- Rescaled from of Hartmann-6 function from [Picheny et al. (2012)], "http://www.sfu.ca/~ssurjano/hart6.html"
alpha = np.array([[1.0], [1.2], [3.0], [3.2]])
A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
              [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
              [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
              [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                     [2329, 4135, 8307, 3736, 1004, 9991],
                     [2348, 1451, 3522, 2883, 3047, 6650],
                     [4047, 8828, 8732, 5743, 1091, 381]])
def Hartmann6OF(x):
    result = []
    for i in range(0,4):
        vect = []
        for j in range(0,6):
            vect.append( A[i,j] * ( x[j] - P[i,j] )**2 )
        result.append( alpha[i] * np.exp( -1 * sum(vect) ) )
    # return -1 * sum(result)
    return -1 * ( 2.58 + sum(result) ) / 1.94

bounds = np.full((6,2),[0., 1.])

sol = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
print("Hartmann OF minimum: x = {}, f(x) = {}".format(sol,Hartmann6OF(sol)))


##- Optimize OF:
# model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,random_state=seed)
# model.set_AF_tradeoff(None)
# model.set_EIC_params(lb=-20.0,ub=-2.9)
# result = model.optimize(Hartmann6OF,bounds,n_samples=60,max_iter=max_iter)[1]
# print("Result: {}".format(result))

# full_path = test_images_folder_path + '6D_Hartmann_' + AF_name + '_' + K_key
# model.plot_convergence(f_opt=Hartmann6OF(sol),title=full_path,show=True)
# model.plot_convergence_x(sol,title=full_path,show=True)


# 7. 10D test: Ackley function
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution:
##- Ackley function, "http://www.sfu.ca/~ssurjano/ackley.html"
D = 10
##- a = 20, b = 0.2, c = 2π
a, b, c = 20, 0.2, 2*np.pi
def AckleyOF(x):
    x = np.array(x).reshape(1,D)
    term1 = np.exp( -b * np.sqrt( (x**2).sum(1) / D ) )
    term2 = np.exp( np.cos( c*x ).sum(1) / D )
    return a + np.exp(1) - a * term1 - term2

bounds = np.array([[-32.768, 32.768]*D]).reshape(-1,2)

sol = np.array([[0.0]*D]).reshape(1,D)
print("Ackley OF minimum: x = {}, f(x) = {}".format(sol,AckleyOF(sol)))


# ##- Optimize OF:
model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,random_state=seed)
model.set_AF_tradeoff(None)
model.set_EIC_params(lb=-20.0,ub=20.0)
result = model.optimize(AckleyOF,bounds,n_samples=10*D,max_iter=max_iter)[1]
print("Result: {}".format(result))

full_path = test_images_folder_path + '10D_Ackley_' + AF_name + '_' + K_key
model.plot_convergence(f_opt=AckleyOF(sol),title=full_path,show=True)
model.plot_convergence_x(sol,title=full_path,show=True)


# 8. 10D test: Dixon-Price function
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution:
##- Dixon-Price function, "http://www.sfu.ca/~ssurjano/dixonpr.html"
D = 10
def PrayOF(x):
    vect = []
    for i in range(1,D):
        vect.append( i * ( 2*x[i]**2 - x[i-1] )**2 )
    return (x[0] - 1)**2 + sum(vect)

# bounds = np.array([[-10.0, 10.0]*D]).reshape(-1,2)
bounds = np.array([[-6.0, 6.0]*D]).reshape(-1,2)

sol = []
for i in range(1,D+1):
    sol.append( 2**( -(2**i-2) / 2**i ) )
sol = np.array(sol)
print("Dixon-Price OF minimum: x = {}, f(x) = {}".format(sol,PrayOF(sol)))


# ##- Optimize OF:
# model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,random_state=seed)
# model.set_AF_tradeoff(None)
# model.set_EIC_params(lb=-20.0,ub=20.0)
# result = model.optimize(PrayOF,bounds,n_samples=10*D,max_iter=max_iter)[1]
# print("Result: {}".format(result))

# full_path = test_images_folder_path + '10D_DixonPrice_' + AF_name + '_' + K_key
# model.plot_convergence(f_opt=PrayOF(sol),title=full_path,show=True)
# model.plot_convergence_x(sol,title=full_path,show=True)


# Tests finished
#####################################################################################################################

end = time.time()

execution_time = str(end-start)
print("Execution Time: " + execution_time)

exe_time_file = open(os.path.join("./","output.txt"), 'a')
exe_time_file.write(str(datetime.datetime.today())+": "+str(execution_time)+"\n")
exe_time_file.close()
