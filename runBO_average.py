## Script written by Jose` Y. Villafan.
## Last edited on 11/02/2020.
## BOSS-V algorithm.

import warnings, time, datetime, os
import numpy as np
import matplotlib.pyplot as plt
from operator import add
from structure1_utils import find_first_index_below, compute_simpleregret, CComputeIterStats
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel as W
import structure3_bayesianoptimization as BO
warnings.filterwarnings('ignore')

start = time.time()


# 0. Settings
#####################################################################################################################
seed = 0
np.random.seed(seed)

# test_images_folder_path = '../Bebop images BACKUP 2020-04-16/BO_average/'
test_images_folder_path = 'BO_average/' # for experiments in Ardagna's servers

BO.debug = False

Acquisition_function = BO.SEI
AF_name = "SEI"

K_dict = {"RBF": RBF(length_scale=[1.0]),
          "Matern52": Matern(length_scale=[1.0],nu=2.5)}
K_key = "Matern52"
kernel = K_dict[K_key]

noise = True
max_iter = 100
show_plots = False

N_runs_to_average = 10
tols = [1e0,1e-1,1e-2,1e-3,1e-4]
CCalc = CComputeIterStats()


# 0. Define functions to plot results
#####################################################################################################################

def plot_regret(f_list,f_opt,title="",show=False):
    symbols = ['-bo','-go','-ro','-ko']
    labels = [u'CB',u'PI',u'EI',u'SEI']
    # symbols = ['-go','-ro','-ko']
    # labels = [u'PI',u'EI',u'SEI']

    plt.figure(figsize=(7,4))

    for i in range(0,len(f_list)):
        r = compute_simpleregret(f_list[i],f_opt)
        plt.plot(r,symbols[i],label=labels[i])

    plt.yscale('log')
    plt.title(r'Simple regret', fontsize=14)
    plt.xlabel(r'iteration', fontsize=14)
    plt.ylabel(r'$min_{i} |f_{opt}-f(x_{i})|$', fontsize=12)
    plt.legend(loc='upper right', fontsize=16)
    from pylab import grid
    grid(True)

    plt.savefig(title+'_BayesianOptimization_SIMPLEregret.png')
    plt.savefig(title+'_BayesianOptimization_SIMPLEregret.svg')
    if show:
        plt.show(block=True)


def plot_all_y_lists(f_list,f_opt,title="",show=False):
    symbols = ['-bo','-go','-ro','-ko']
    labels = [u'CB',u'PI',u'EI',u'SEI']
    # symbols = ['-go','-ro','-ko']
    # labels = [u'PI',u'EI',u'SEI']

    plt.figure(figsize=(7,4))

    temp_list = [0.0] * max_iter
    for i in range(0,len(f_list)):
        r = np.log(compute_simpleregret(f_list[i],f_opt))
        temp_list = list(map(add,temp_list,r))

        if ((i+1) % N_runs_to_average) == 0:
            temp_list = list(map(lambda x: x / N_runs_to_average,temp_list))
            plt.plot(temp_list,symbols[(i+1)//N_runs_to_average-1],label=labels[(i+1)//N_runs_to_average-1])

    # plt.yscale('log')
    plt.title(r'Simple regret', fontsize=14)
    plt.xlabel(r'iteration', fontsize=14)
    plt.ylabel(r'$min_{i} |f_{opt}-f(x_{i})|$', fontsize=12)
    plt.legend(loc='upper right', fontsize=16)
    from pylab import grid
    grid(True)

    plt.savefig(title+'_BayesianOptimization_AVERAGEregret.png')
    plt.savefig(title+'_BayesianOptimization_AVERAGEregret.svg')
    if show:
        plt.show(block=True)


# 1. 1D test
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution and starting samples:
x0 = 0.33
def OF1(x):
    return (x[0]-x0)**2 - 10

bounds = np.array([[-4.0, 4.0]])

sol = np.array([x0])

X = np.array([[-5.0],[3.0],[6.0]])


##- Optimize OF:
# y_list = [0.0] * max_iter
# for i in range(0,N_runs_to_average):
#     model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,normalize_y=False,adaptive=True,noise=noise)
#     result = model.optimize(OF1,bounds,X,max_iter=max_iter,plot_acquisition=show_plots)[1]
#     print("Run {}: Result: {}".format(i+1,result))
#     CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,OF1(sol)),tols))
#     y_list = list(map(add,y_list,model.y_output_list))

# y_list = list(map(lambda x: x / N_runs_to_average,y_list))

# full_path = test_images_folder_path + '1D_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs'
# BO.plot_regret(f_list=y_list,f_opt=OF1(sol),title=full_path,show=True)

# CCalc.compute_stats()
# CCalc.pretty_print(tols)


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
sol = res.x
print(sol,ForresterOF(sol))

##- PI tends to exploit around the best point, more than others AFs.
##- Thus, it needs a good starting configuration, either denser or "luckier".
if Acquisition_function in [BO.PI,BO.SEI,BO.MES]:
    X = np.array([[0.1],[0.5],[0.8]])
else:
    X = np.array([[0.1],[0.5],[0.9]])


##- Optimize OF:
# y_list = [0.0] * max_iter
# for i in range(0,N_runs_to_average):
#     model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,normalize_y=False,adaptive=True,noise=noise)
#     result = model.optimize(ForresterOF,bounds,X,max_iter=max_iter,plot_acquisition=show_plots)[1]
#     print("Run {}: Result: {}".format(i+1,result))
#     CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,ForresterOF(sol)),tols))
#     y_list = list(map(add,y_list,model.y_output_list))

# y_list = list(map(lambda x: x / N_runs_to_average,y_list))

# full_path = test_images_folder_path + '1D_Forrester_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs'
# BO.plot_regret(f_list=y_list,f_opt=ForresterOF(sol),title=full_path,show=True)

# CCalc.compute_stats()
# CCalc.pretty_print(tols)


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
res = minimize(fun=BraninOF,x0=x0.reshape(1,-1),method='L-BFGS-B',bounds=np.array([[-5.0, 0.0], [10.0, 15.0]]))
print("Branin OF minimum: x = {}, f(x) = {}".format(res.x,res.fun))
sol = res.x


##- Optimize OF:
# y_list = [0.0] * max_iter
# all_average_lists = []
# all_y_lists = []
# for iAF in [BO.CB, BO.PI, BO.EI, BO.SEI]:
#     print("***** Run: {} *****".format(str(iAF)))
#     seed = 0
#     np.random.seed(seed)
#     CCalc = CComputeIterStats()
#     for i in range(0,N_runs_to_average):

#         ls = np.mean(bounds[:,1] - bounds[:,0]) / 2
#         kernel = Matern(length_scale=ls,nu=2.5,length_scale_bounds='fixed')
#         kernel = C(constant_value=1e2,constant_value_bounds='fixed') * kernel + W(noise_level=1e-4,noise_level_bounds='fixed')

#         model = BO.CBayesOpt(iAF,kernel,greater_is_better=False,adaptive=False,noise=True,random_state=seed,verbose=False)
#         result = model.optimize(BraninOF,bounds,n_samples=21,max_iter=max_iter,plot_acquisition=show_plots)[1]
#         print("Run {}: Result: {}".format(i+1,result))
#         CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,BraninOF(sol)),tols))
#         all_y_lists.append(model.y_output_list)
#         y_list = list(map(add,y_list,model.y_output_list))

#     y_list = list(map(lambda x: x / N_runs_to_average,y_list))
#     all_average_lists.append(y_list)

#     CCalc.compute_stats()
#     CCalc.pretty_print(tols)

# full_path = test_images_folder_path + 'NONADAPTIVE_2D_Branin_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs_ALL'
# # BO.plot_regret(f_list=y_list,f_opt=BraninOF(sol),title=full_path,show=True)
# plot_all_y_lists(f_list=all_y_lists,f_opt=BraninOF(sol),title=full_path,show=True)
# plot_regret(f_list=all_average_lists,f_opt=BraninOF(sol),title=full_path,show=True)


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
# y_list = [0.0] * 200
# for i in range(0,N_runs_to_average):
#     model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,normalize_y=False,adaptive=True,noise=noise)
#     result = model.optimize(RosenbrockOF,bounds,n_samples=21,max_iter=200,plot_acquisition=show_plots)[1]
#     print("Run {}: Result: {}".format(i+1,result))
#     CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,RosenbrockOF(sol)),tols))
#     y_list = list(map(add,y_list,model.y_output_list))

# y_list = list(map(lambda x: x / N_runs_to_average,y_list))

# full_path = test_images_folder_path + '2D_Rosenbrock_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs'
# BO.plot_regret(f_list=y_list,f_opt=RosenbrockOF(sol),title=full_path,show=True)

# CCalc.compute_stats()
# CCalc.pretty_print(tols)


# 5. 4D test
#####################################################################################################################

##- Define toy objective function, bounds (constraints), solution:
def OF4(x):
    return (x[0]-2)**2 + (x[1]-3)**2 + (x[2]-0)**2 + (x[3]-1)**2 - 10

bounds = np.full((4,2),[-1., 5.])

sol = np.array([2., 3., 0., 1.])


##- Optimize OF:
# y_list = [0.0] * max_iter
# for i in range(0,N_runs_to_average):
#     model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,normalize_y=True,adaptive=True,noise=noise)
#     result = model.optimize(OF4,bounds,n_samples=6,max_iter=max_iter)[1]
#     print("Run {}: Result: {}".format(i+1,result))
#     CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,OF4(sol)),tols))
#     y_list = list(map(add,y_list,model.y_output_list))

# y_list = list(map(lambda x: x / N_runs_to_average,y_list))

# full_path = test_images_folder_path + '4D_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs'
# BO.plot_regret(f_list=y_list,f_opt=OF4(sol),title=full_path,show=True)

# CCalc.compute_stats()
# CCalc.pretty_print(tols)


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
y_list = [0.0] * max_iter
all_average_lists = []
all_y_lists = []
for iAF in [BO.CB, BO.PI, BO.EI, BO.SEI]:
# for iAF in [BO.PI, BO.EI, BO.SEI]:
    print("***** Run: {} *****".format(str(iAF)))
    seed = 0
    np.random.seed(seed)
    CCalc = CComputeIterStats()
    for i in range(0,N_runs_to_average):

        # ls = np.mean(bounds[:,1] - bounds[:,0]) / 2
        # kernel = Matern(length_scale=ls,nu=2.5,length_scale_bounds='fixed')
        # kernel = C(constant_value=1e2,constant_value_bounds='fixed') * kernel + W(noise_level=1e-4,noise_level_bounds='fixed')

        kernel = Matern(length_scale=[1.0],nu=2.5)

        model = BO.CBayesOpt(iAF,kernel,greater_is_better=False,adaptive=True,noise=True,random_state=seed,verbose=False)
        result = model.optimize(Hartmann6OF,bounds,n_samples=72,max_iter=max_iter,plot_acquisition=show_plots)[1]
        print("Run {}: Result: {}".format(i+1,result))
        CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,Hartmann6OF(sol)),tols))
        all_y_lists.append(model.y_output_list)
        y_list = list(map(add,y_list,model.y_output_list))

    y_list = list(map(lambda x: x / N_runs_to_average,y_list))
    all_average_lists.append(y_list)

    CCalc.compute_stats()
    CCalc.pretty_print(tols)

full_path = test_images_folder_path + 'ADAPTIVE_6D_Hartmann_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs_ALL'
# BO.plot_regret(f_list=y_list,f_opt=Hartmann6OF(sol),title=full_path,show=True)
plot_all_y_lists(f_list=all_y_lists,f_opt=Hartmann6OF(sol),title=full_path,show=True)
plot_regret(f_list=all_average_lists,f_opt=Hartmann6OF(sol),title=full_path,show=True)


##- Optimize OF:
# y_list = [0.0] * max_iter
# for i in range(0,N_runs_to_average):
#     model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,normalize_y=False,adaptive=True,noise=noise)
#     result = model.optimize(Hartmann6OF,bounds,n_samples=60,max_iter=max_iter)[1]
#     print("Run {}: Result: {}".format(i+1,result))
#     CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,Hartmann6OF(sol)),tols))
#     y_list = list(map(add,y_list,model.y_output_list))

# y_list = list(map(lambda x: x / N_runs_to_average,y_list))

# full_path = test_images_folder_path + '6D_Hartmann_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs'
# BO.plot_regret(f_list=y_list,f_opt=Hartmann6OF(sol),title=full_path,show=True)

# CCalc.compute_stats()
# CCalc.pretty_print(tols)


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


##- Optimize OF:
# y_list = [0.0] * max_iter
# for i in range(0,N_runs_to_average):
#     model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,normalize_y=False,adaptive=True,noise=noise)
#     result = model.optimize(AckleyOF,bounds,n_samples=10*D,max_iter=max_iter)[1]
#     print("Run {}: Result: {}".format(i+1,result))
#     CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,AckleyOF(sol)),tols))
#     y_list = list(map(add,y_list,model.y_output_list))

# y_list = list(map(lambda x: x / N_runs_to_average,y_list))

# full_path = test_images_folder_path + '10D_Ackley_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs'
# BO.plot_regret(f_list=y_list,f_opt=AckleyOF(sol),title=full_path,show=True)

# CCalc.compute_stats()
# CCalc.pretty_print(tols)



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


##- Optimize OF:
# y_list = [0.0] * max_iter
# for i in range(0,N_runs_to_average):
#     model = BO.CBayesOpt(Acquisition_function,kernel,greater_is_better=False,normalize_y=False,adaptive=True,noise=noise)
#     result = model.optimize(PrayOF,bounds,n_samples=10*D,max_iter=max_iter)[1]
#     print("Run {}: Result: {}".format(i+1,result))
#     CCalc.store(find_first_index_below(compute_simpleregret(model.y_output_list,PrayOF(sol)),tols))
#     y_list = list(map(add,y_list,model.y_output_list))

# y_list = list(map(lambda x: x / N_runs_to_average,y_list))

# full_path = test_images_folder_path + '10D_DixonPrice_' + AF_name + '_' + K_key + '_' + str(N_runs_to_average) + 'runs'
# BO.plot_regret(f_list=y_list,f_opt=PrayOF(sol),title=full_path,show=True)

# CCalc.compute_stats()
# CCalc.pretty_print(tols)


# Tests finished
#####################################################################################################################

end = time.time()

execution_time = str(end-start)
print("Execution Time: " + execution_time)

exe_time_file = open(os.path.join("./","output.txt"), 'a')
exe_time_file.write(str(datetime.datetime.today())+": "+str(execution_time)+"\n")
exe_time_file.close()
