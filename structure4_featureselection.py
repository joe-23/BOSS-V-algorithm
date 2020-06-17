## Script written by Jose` Y. Villafan.
## Last edited on 15/01/2020.
## BOSS-V algorithm.
# Copyright (c) 2020, the BOSS-V author (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE)

import operator, warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from structure1_utils import create_pipeline
warnings.filterwarnings('ignore')


# 1. Define Feature Selector class, which uses Cross-validated SFFS to select the best features
#####################################################################################################################
class CFeatureSelector(object):
    """Class that implements a single iteration of Sequential Forward Feature Selection (SFS), both Floating and not."""

    def __init__(self,model=Ridge,norm=0,floating=False,target_features=None,random_state=None):
        self.model      = model                                     # Linear regression model: must implement fit() and predict() methods
        self.floating   = floating                                  # Floating selection is performed when True
        self.desired_p  = target_features                           # Desired number of features
        self.rng_seed   = random_state                              # RandomState instance

        self.pipe       = create_pipeline(norm,self.model(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True,
                                                          max_iter=None, tol=0.001, random_state=self.rng_seed))


    def __call__(self,X,Y,X_add=None,Y_add=None,folds=None,initial_features=[],mandatory_features=[]):
        """
            Perform SFS via folds-fold Cross-validation (CV).

            Input: X - numpy 2D array
                matrix of observed data
            Input: Y - numpy 1D array
                vector of observed evaluations of the objective, Y = f(X)
            Input: X_add - numpy 2D array
                matrix of additional observed data to be used in the feature selection procedure
            Input: Y_add - numpy 1D array
                vector of additional observed evaluations of the objective, Y_add = f(X_add)
            Input: folds - int
                number of folds to be used in CV
                if None then LOO-CV is performed
            Input: initial_features - list
                list of indices (representing features) that are not to be excluded from the search
                if initial_features == 'all' then no feature selection is performed
            Input: mandatory_features - list
                list of indices (representing features) that are not to be removed and are automatically included in the initial_features list
            Output: optimal features, relative CV score, fitted model - [list, float, sklearn.linear_model]
        """
        ##- Add additional data if provided
        if X_add is not None and Y_add is not None:
            X, Y = np.vstack((X,X_add)), np.vstack((Y,Y_add))

        self.X          = X                                         # Matrix of known observations
        self.Y          = Y                                         # Vector of evaluations of the objective f at X
        self.nObs       = self.X.shape[0]                           # Number of observations
        self.d          = self.X.shape[1]                           # Dimensionality of the observations
        self.features   = list(range(0,self.d))                     # Complete dataset features/regressors list

        ##- Warn user about the error is going to happen if X matrix has less than 2 observations
        if self.nObs < 2:
            warnings.warn("CV score cannot be computed if training set is empty due to X matrix having less than 2 observations.",Warning)

        ##- Check whether to perform feature selection
        if initial_features == 'all':
            initial_features = self.features
        else:
            initial_features = list(sorted(set(initial_features).union(set(mandatory_features))))

        ##- Verify whether the desired number of features has already been reached and return a naively fitted regressor
        if self.desired_p is not None and len(initial_features) >= self.desired_p:
            fitted_model = self.compute_CV(self.X[:,initial_features],self.Y,CVfolds=folds)[1]
            return initial_features, -np.inf, fitted_model

        ##- Select subset of features, if there are none then return initial_features: this is a forward selector!
        available_features = list(set(self.features) - set(initial_features))
        if not available_features:
            fitted_model = self.compute_CV(self.X[:,initial_features],self.Y,CVfolds=folds)[1]
            return initial_features, -np.inf, fitted_model

        ##- Create dictionaries where CV scores will be collected
        Dict_best_features = dict()
        Dict_best_score = dict()
        Dict_best_model = dict()

        #--- Forward feature selection step ---#
        for iFeature in sorted(available_features,reverse=True):

            current_features = list(sorted(set(initial_features).union(set([iFeature]))))
            score, model, _ = self.compute_CV(self.X[:,current_features],self.Y,CVfolds=folds)

            Dict_best_features[iFeature] = current_features
            Dict_best_score[iFeature] = score
            Dict_best_model[iFeature] = model

        best_added_feature = max(Dict_best_score.items(), key=operator.itemgetter(1))[0]
        best_features = Dict_best_features[best_added_feature]
        best_score = Dict_best_score[best_added_feature]
        fitted_model = Dict_best_model[best_added_feature]

        #--- Floating feature selection step ---#
        if self.floating:
            for jFeature in sorted(initial_features):
                ##- Do not remove mandatory features
                if jFeature in mandatory_features:
                    pass
                else:
                    current_features = list(sorted(set(best_features) - set([jFeature])))
                    score, model, _ = self.compute_CV(self.X[:,current_features],self.Y,CVfolds=folds)

                    if score > Dict_best_score[best_added_feature]:
                        Dict_best_features[best_added_feature] = current_features
                        Dict_best_score[best_added_feature] = score
                        Dict_best_model[best_added_feature] = model

            best_features = Dict_best_features[best_added_feature]
            best_score = Dict_best_score[best_added_feature]
            fitted_model = Dict_best_model[best_added_feature]

        return best_features, best_score, fitted_model


    def compute_CV(self,X_train,Y_train,CVfolds=None,scoring='neg_mean_squared_error'):
        """
            Choose best model by evaluating MSE and tuning hyperparameters via cross-validation.
            This is achieved by performing grid search (GridSearchCV) over the suitable parameter grid.

            Input: X_train - numpy 2D array
                matrix of observed data
            Input: Y_train - numpy 1D array
                vector of observed evaluations of the objective, Y_train = f(X_train)
            Input: CVfolds - int
                number of folds to use in CV
            Input: scoring - string
                one of the allowed scoring criteria: see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
            Output: score of the best model object - float,
                    best regression model object - sklearn.pipeline.Pipeline,
                    parameters of the best model object - dictionary
        """
        ##- Select the number of folds for leave-one-out CV
        if CVfolds is None:
            CVfolds = X_train.shape[0]

        ##- Perform CV for both regressor performance evaluation and hyperparameter tuning
        estimator  = self.pipe
        param_grid = get_params_dict(self.model)
        model_grid = GridSearchCV(estimator,param_grid,scoring=scoring,cv=CVfolds,return_train_score=False)
        model_grid.fit(X_train,Y_train)

        return model_grid.best_score_, model_grid.best_estimator_, model_grid.best_params_


def get_params_dict(model):
    '''
        Generate dictionary containing the linear model parameters.

        Input: model - sklearn.linear_model
        Output: parameters_dict - dictionary
    '''
    PARAM_DICT = {
    Ridge: {
        'reg__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10],#[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10]
        'reg__copy_X': [True],#[True, False]
        'reg__fit_intercept': [True, False],#[True, False]
        'reg__solver': ['auto'],#['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        'reg__tol': [0.01]#[0.001, 0.01, 0.1, 1]
    },
    Lasso: {
        'reg__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10],#[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10]
        'reg__copy_X': [True],#[True, False]
        'reg__fit_intercept': [True, False],#[True, False]
        'reg__tol': [0.01]#[0.001, 0.01, 0.1, 1]
    }
    }

    return PARAM_DICT[model]
