## Script written by Jose` Y. Villafan.
## Last edited on 18/02/2020.
## BOSS-V algorithm.
# Copyright (c) 2020, the BOSS-V author (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE)

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 1. Define custom exception
#####################################################################################################################

class WrongCallToMethod(Exception):
    """Raise for call to wrong method exception."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)


# 2. Define helper functions
#####################################################################################################################

def proper_round(num, dec=0):
    """
        Define proper round() function.

        Input: num - float
            number to round
        Input: dec - int
            decimal digit where rounding should take place
        Output: rounded num to nearest decimal dec - float
    """
    return round(num+10**(-len(str(num))-1), dec)


def create_pipeline(norm,model):
    """
        Create the pipeline for normalization and regression fitting.

        Input: norm - int
            norm = [0, 1, 2]
            if 1 standard scaling will be performed on the data
            if 2 minmax scaling will be performed on the data
            else no scaling will be performed
        Input: model - linear regression model
        Output: pipeline - sklearn.pipeline.Pipeline
    """
    if norm == 1:
        scale = StandardScaler()
        pipe = Pipeline([('norm', scale), ('reg', model)])
    elif norm == 2:
        scale = MinMaxScaler()
        pipe = Pipeline([('norm', scale), ('reg', model)])
    else:
        pipe = Pipeline([('reg', model)])
    return pipe


def min_dist(X,x):
    """
        Find the distance and the index of the observation in X matrix which is closest in l2-norm to observation x.

        Input: X - numpy 2D array
            matrix of observed data
        Input: x - numpy 1D array
        Output: min_distance, index - [float, int]
    """
    X = X.astype(float)
    x = x.astype(float)

    min_distance = 0
    index = None
    for iObs in range(0,X.shape[0]):
        distance = np.linalg.norm(x-X[iObs,:],ord=2)
        if iObs == 0 or distance < min_distance:
            min_distance = distance
            index = iObs
    return min_distance, index


def relative_error(y_true,y_predicted):
    """
        Compute relative error between y_true and y_predicted.

        Input: y_true - numpy 1D array, or float
        Input: y_predicted - numpy 1D array, or float
        Output: relative error between y_true and y_predicted - numpy 1D array, or float
    """
    y_true, y_predicted = np.array(y_true), np.array(y_predicted)
    return (y_true - y_predicted) / y_true


def compute_simpleregret(f_list,f_opt):
    """
        Compute simple regret, see [Wang and Jegelka (2017); Chapter 2.3] for details.

        Input: f_list - list
            list of evaluations of the objective obtained from optimization procedure, tipically CBayesOpt.y_output_list
        Input: f_opt - float
            exact value of the objective evaluated at the optimum
        Output: simple regret - list
    """
    r = []
    for i in range(0,len(f_list)):
        ##- Compute regret
        regrets = [f_opt - f_list[k] for k in range(0,i+1)]
        r.append(np.min(list(map(lambda x: abs(x), regrets))))
        ##- Compute minimum
        # regrets = [f_list[k] for k in range(0,i+1)]
        # r.append(np.min(regrets))
    return r


# 3. Define performance metrics which compute the MAPE and MPE of a fitted model given the relative error
#####################################################################################################################
def MAPE(ratio):
    """
        Mean Absolute Percentage Error (MAPE) score evaluator.

        Input: ratio - numpy 1D array, or float
            relative error between predicted and true y values
        Output: MAPE score - float
    """
    return np.mean(np.abs(ratio)) * 100


def MPE(ratio):
    """
        Mean Percentage Error (MPE) score evaluator.

        Input: ratio - numpy 1D array, or float
            relative error between predicted and true y values
        Output: MPE score - float
    """
    return np.mean(ratio) * 100


# 4. Define function that computes iterations necessary for BO to reach a fixed tolerance given its output
#####################################################################################################################
def find_indices(lst, condition):
    """
        Find indices of lst satisfying the specified condition.

        Input: lst - list
        Input: condition - lambda function
        Output: list of indices - list
    """
    return [index for index, elem in enumerate(lst) if condition(elem)]


def find_human_indices(lst, condition):
    """
        Find indices of lst satisfying the specified condition.
        The output list contains indices starting from 1 rather than 0.

        Input: lst - list
        Input: condition - lambda function
        Output: list of indices - list
    """
    return [index+1 for index, elem in enumerate(lst) if condition(elem)]


def find_first_index_below(lst, condition_values_list):
    """
        Find index of first element in lst that satisfies the specified condition, otherwise returns -1.
        Returns a list if given a list of conditions.

        Input: lst - list
        Input: condition_values_list - list, or float
        Output: list of indices, or index - list
    """
    output = []
    for tol in condition_values_list:
        lista = find_human_indices(lst, lambda elem: elem < tol)
        if lista:
            output.append(min(lista))
        else:
            output.append(-1)
    return output


class CComputeIterStats(object):
    """Class that stores iteratively lists of elements of which
    to compute mean and standard deviation across iterations."""

    def store(self,lista):
        """
            Store the given list in memory if called once, otherwise append to the already stored sequence of lists.

            Input: lista - list
        """
        if not hasattr(self,"memory"):
            self.memory = np.array(lista).reshape(1,-1)
        else:
            self.memory = np.vstack((self.memory,lista))
        return self
    
    def compute_stats(self):
        """
            Compute mean and standard deviation column-wise on the stored sequence of lists.

            Input: lista - list
            Output: list of mean and standard deviation pairs - list of tuples
        """
        self.output = []
        for iCol in range(0,self.memory.shape[1]):
            vect = list(filter(lambda x : x>0, self.memory[:,iCol]))
            self.output.append((np.mean(vect),np.std(vect)))
        return self.output

    def pretty_print(self,condition_values_list):
        """
            Print mean and standard deviation computed by compute_stats() method if given the list
            of conditions that was used to store() elements in the first place.

            Input: condition_values_list - list
        """
        if len(self.output) != len(condition_values_list):
            raise ValueError('Wrong number of elements: was expecting {} elements, got {}.'.format(len(self.output),len(condition_values_list)))

        for i in range(0,len(condition_values_list)):
            mean, std = self.output[i]
            tol = condition_values_list[i]
            print("Tol = {}: iterations = {} +- {}".format(tol,mean,std))


# 5. Define class that checks whether the new observation obtained from BO is duplicate
#####################################################################################################################
class check_duplicate(object):
    """Callable class that checks whether the new observation is duplicate and randomly samples a new point if that is the case."""

    def __init__(self):
        self.__random_jumps = 0                                     # Random jumps counter

    def __call__(self,new_Obs,X,bounds,max_random_jumps=None,tol=1e-10):
        """
            Check whether the new observation is duplicate and randomly sample a new point if that is the case.

            Input: new_Obs - numpy 1D array
                configuration of which to check whether it is duplicate
            Input: X - numpy 2D array
                matrix of configurations against which to test for duplicates
            Input: bounds - numpy 2D array
                matrix of [lower_bound,upper_bound] where indices represent the dimension (starting from 0)
            Input: max_random_jumps - int
                maximum number of allowed random jumps;
                successive duplicates are sampled in a neighborhood of new_Obs of radius tol
            Input: tol - float
                radius of the neighborhood of new_Obs in which to check for duplicates
            Output: new_Obs, newly sampled configuration if duplicate,  otherwise the given one - numpy 1D array
                    info, its value is True when duplicate - bool
        """
        ##- Define maximum number of random jumps
        if max_random_jumps is None:
            max_random_jumps = 5 * bounds.shape[0]

        ##- Find out whether new_Obs is duplicate
        info = False
        duplicate = True if min_dist(X,new_Obs)[0] <= tol else False
        if duplicate:
            self.__random_jumps += 1
            info = True

        ##- Sample new point; safe because it does not yield a possible duplicate in spite of randomness
        while duplicate:
            if self.__random_jumps < max_random_jumps + 1:
                new_Obs = np.random.uniform(bounds[:, 0],bounds[:, 1],bounds.shape[0])
            else:
                new_Obs = np.random.uniform(new_Obs-1e2*tol,new_Obs+1e2*tol,bounds.shape[0])
            duplicate = True if min_dist(X,new_Obs)[0] <= tol else False

        return new_Obs, info
