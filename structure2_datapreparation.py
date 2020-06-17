## Script written by Jose` Y. Villafan.
## Last edited on 23/02/2020.
## BOSS-V algorithm.
# Copyright (c) 2020, the BOSS-V author (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE)

from configparser import SafeConfigParser
import ast,os
import numpy as np
import pandas as pd
from structure1_utils import WrongCallToMethod


# 1. Define class that generates datasets from data provided in a configuration file
#####################################################################################################################
class CDataPreparation(object):
    """Utility class whose purpose is to read parameters from a configuration file and
        generate four dictionaries containing X and Y matrices for successive analysis."""

    def __init__(self,config_path,verbose=False):
        """
            Initialize data parameters given configuration file path.

            Input: config_path - string
            Input: verbose - bool
        """
        self.parser             = SafeConfigParser()
        self.dataset_kind       = ""                                    # Dataset source: ["Query26","Kmeans","SparkDL"]
        self.split              = ""                                    # Analysis: ["interpolation","extrapolation"]
        self.box_model          = ""                                    # Box model: ["gray","black"]
        self.target_column      = ""                                    # Column name of function evaluations

        ##- Read config file
        if os.path.isfile(config_path):
            self.parser.read(config_path)
            self.dataset_kind   = str(self.parser.get('DataPreparation','dataset_kind'))
            dataset_name        = str(ast.literal_eval(self.parser.get('DataPreparation','dataset_path_and_name')))
            self.target_column  = str(self.parser.get('DataPreparation','target_column'))
            self.split          = str(self.parser.get('DataPreparation','split'))
            self.box_model      = str(self.parser.get('DataPreparation','box_model'))
        else:
            print("Config file not found")

        ##- Check whether splits (Interpolation/Extrapolation analysis) and cases (Configurations analysis) are well defined
        if self.dataset_kind not in ["Query26","Kmeans","SparkDL"]:
            raise ValueError("Configuration parameter 'dataset_kind' is not well specified.")
        if self.split not in ["extrapolation","interpolation"]:
            raise ValueError("Configuration parameter 'split' is not well specified.")

        ##- Read dataset
        self.dataset = pd.read_csv(os.path.join(dataset_name))
        self.dataset = self.dataset.apply(pd.to_numeric, errors = 'coerce')
        self.dataset.fillna(0, inplace = True) # Impute NA to 0s
        self.dataset.sort_values(by=['nContainers'], axis=0, ascending=True, inplace=True, kind='mergesort')

        ##- Add/remove necessary features for Gray and Black Box models analysis
        self.dataset["inverse_nContainers"] = self.dataset.apply(lambda row: 1.0/float(row["nContainers"]), axis=1)
        self.dataset["dataSize_over_nContainers"] = self.dataset.apply(lambda row: float(row["dataSize"])/float(row["nContainers"]), axis=1)
        self.dataset["log_nContainers"] = self.dataset.apply(lambda row: np.log(float(row["nContainers"])), axis=1)
        self.dataset["sqrt_dataSize_over_nContainers"] = self.dataset.apply(lambda row: np.sqrt(float(row["dataSize"])/float(row["nContainers"])), axis=1)
        self.dataset.drop(["run", "users"], axis=1, inplace=True)

        ##- If target column is application completion time, transform milliseconds into hours
        if self.target_column == "applicationCompletionTime":
            conversion_unit = 3600000
            self.dataset[self.target_column] = self.dataset.apply(lambda row: float(row[self.target_column])/conversion_unit, axis=1)

        ##- Generate lists of regressors for Gray and Black Box models
        nonDAGfeatures = ["dataSize","nContainers","inverse_nContainers","dataSize_over_nContainers","log_nContainers","sqrt_dataSize_over_nContainers"]
        self.nonDAGfeatures_number = len(nonDAGfeatures)
        if self.box_model == "gray":
            self.model_regressors = self.dataset.drop([self.target_column],axis=1).columns
        elif self.box_model == "black":
            self.model_regressors = self.dataset[nonDAGfeatures].columns
        else:
            raise ValueError("Configuration parameter 'box_model' is not well specified.")

        ##- Print information regarding current dataset
        if verbose:
            self.__print_info()


    def generate_XY(self,apply_log_y=False):
        """
            Generate four dictionaries containing X, Y matrices divided divided by splits, cases and kind = ["train","test"].
            They have the same keys as the dictionaries 'splits' and 'cases' defined in splits_and_cases_dict().

            Input: apply_log_y - bool
                apply log() to Y column when True
            Output: X_train, X_test, y_train, y_test - 2 key dictionaries
        """
        ##- Define hidden var for printing
        self._generate_XY_called = True

        ##- Generate splits (Interpolation/Extrapolation analysis) and cases (Configurations analysis)
        splits_dict, cases_dict = splits_and_cases_dict(self.dataset_kind,self.split)

        ##- Generate X matrix, y vector dictionaries separated by train/test, with splits.keys() and cases.keys()
        train_dict = self.__generate_data_for_LR(splits_dict,cases_dict,"train")
        test_dict = self.__generate_data_for_LR(splits_dict,cases_dict,"test")
        X_train = self.__generate_X(train_dict)
        y_train = self.__generate_y(train_dict,apply_log_y)
        X_test = self.__generate_X(test_dict)
        y_test = self.__generate_y(test_dict,apply_log_y)

        ##- Replace test values with averages of train values in Gray Box models; see [Villafan (2020); Chapter 3.2.2] for details
        if self.box_model == "gray":
            X_test = self.__replace_test_values(X_train,X_test)

        return X_train, X_test, y_train, y_test


    def __generate_data_for_LR(self, splits, cases, kind):
        """
            Generate a dictionary containing all the data divided by splits, cases and kind for Linear Regression model validation

            Input: splits - 2 key dictionary
                dictionary obtained from splits_and_cases_dict()
            Input: cases - 2 key dictionary
                dictionary obtained from splits_and_cases_dict()
            Input: kind - string
                kind = ["train","test"]
            Output: gen_data - 2 key dictionary
        """
        gen_data = dict()
        for iSplit in splits.keys():
            temp_data = dict()
            for iCase in cases.keys():
                dataSize_selected = splits[iSplit][kind]
                nContainers_selected = cases[iCase][kind]
                ##- Select all the containers in testing dataset for extrapolation analysis
                if kind == "test" and self.split == "extrapolation":
                    temp_data[iCase] = self.dataset[self.dataset["dataSize"].isin(dataSize_selected)]
                else:
                    temp_data[iCase] = self.dataset[self.dataset["dataSize"].isin(dataSize_selected) & \
                                       self.dataset["nContainers"].isin(nContainers_selected)]
            gen_data[iSplit] = temp_data
        return gen_data


    def __generate_X(self,data):
        """
            Generate the disctionary of X matrices with the same structure as  __generate_data_for_LR() output.

            Input: data - 2 key dictionary
                dictionary obtained from generate_data()
            Output: X - 2 key dictionary
        """
        X = dict()
        for iSplit in data.keys():
            X[iSplit] = dict()
            for iCase in data[iSplit].keys():
                X[iSplit][iCase] = data[iSplit][iCase][self.model_regressors].values
        return X


    def __generate_y(self,data,apply_log_y):
        """
            Generate the disctionary of Y vectors with the same structure as  __generate_data_for_LR() output.

            Input: data - 2 key dictionary
                dictionary obtained from generate_data()
            Input: apply_log_y - bool
                apply log() to Y column when True
            Output: y - 2 key dictionary
        """
        y = dict()
        for iSplit in data.keys():
            y[iSplit] = dict()
            for iCase in data[iSplit].keys():
                if apply_log_y:
                    ##- We compute log(Y) as explained in [Villafan (2020); Chapter 3.1]
                    y[iSplit][iCase] = np.log(data[iSplit][iCase][self.target_column].values).reshape(-1,1)
                else:
                    y[iSplit][iCase] = data[iSplit][iCase][self.target_column].values.reshape(-1,1)
        return y


    def generate_XY_for_BO(self,price_function,apply_log_y=False):
        """
            Generate four dictionaries containing X, Y matrices divided divided by splits, cases and kind = ["train","test"].
            They have the same keys as the dictionaries 'splits' and 'cases' defined in splits_and_cases_dict().

            Input: price_function - function
                price per unit time P(x), function of the configuration
            Input: apply_log_y - bool
                apply log() to y values when True
            Output: X_train, X_test, y_train, y_test, T_train, T_test, P_train, P_test - 4 key dictionaries
        """
        ##- Define hidden var for printing
        self._generate_XY_for_BO_called = True

        ##- Generate splits (Interpolation/Extrapolation analysis) and cases (Configurations analysis)
        splits_dict, cases_dict = splits_and_cases_dict(self.dataset_kind,self.split)

        ##- Generate X matrix, y vector dictionaries separated by train/test, with splits.keys() and cases.keys()
        train_dict = self.__generate_data_for_BO(splits_dict,cases_dict,"train")
        test_dict = self.__generate_data_for_BO(splits_dict,cases_dict,"test")
        X_train, y_train, T_train, P_train = self.__generate_Xy_for_BO(train_dict,price_function,apply_log_y)
        X_test, y_test, T_test, P_test = self.__generate_Xy_for_BO(test_dict,price_function,apply_log_y)

        return X_train, X_test, y_train, y_test, T_train, T_test, P_train, P_test


    def __generate_data_for_BO(self, splits, cases, kind):
        """
            Generate a dictionary containing all the data divided by splits, cases and kind for Linear Regression model validation via Bayesian optimization

            Input: splits - 2 key dictionary
                dictionary obtained from splits_and_cases_dict()
            Input: cases - 2 key dictionary
                dictionary obtained from splits_and_cases_dict()
            Input: kind - string
                kind = ["train","test"]
            Output: gen_data - 3 key dictionary
        """
        gen_data = dict()
        for iSplit in splits.keys():
            temp_data = dict()
            for iCase in cases.keys():
                dataSize_selected = splits[iSplit][kind]
                nContainers_selected_train = cases[iCase]["train"]
                nContainers_selected_test = cases[iCase]["test"]
                DS_data = dict()
                for idataSize in dataSize_selected:
                    DS_data[idataSize] = dict()
                    DS_data[idataSize]["train"] = self.dataset[self.dataset["dataSize"].isin([idataSize]) & \
                                                  self.dataset["nContainers"].isin(nContainers_selected_train)]
                    DS_data[idataSize]["test"] = self.dataset[self.dataset["dataSize"].isin([idataSize]) & \
                                                 self.dataset["nContainers"].isin(nContainers_selected_test)]
                temp_data[iCase] = DS_data
            gen_data[iSplit] = temp_data
        return gen_data


    def __generate_Xy_for_BO(self,data,price_function,apply_log_y):
        """
            Generate the disctionary of X matrices with the same structure as __generate_data_for_BO() output.

            Input: data - 2 key dictionary
                dictionary obtained from generate_data()
            Input: price_function - function
                price per unit time P(x), function of the configuration
            Input: apply_log_y - bool
                apply log() to time, price, cost values when True
            Output: X, y, T, P - Four 4 key dictionaries
        """
        X = dict()                                                  # Configurations
        y = dict()                                                  # Cost = T * P
        T = dict()                                                  # T, Time
        P = dict()                                                  # P, Prize
        for iSplit in data.keys():
            X[iSplit] = dict()
            y[iSplit] = dict()
            T[iSplit] = dict()
            P[iSplit] = dict()
            for iCase in data[iSplit].keys():
                X[iSplit][iCase] = dict()
                y[iSplit][iCase] = dict()
                T[iSplit][iCase] = dict()
                P[iSplit][iCase] = dict()
                for idataSize in data[iSplit][iCase].keys():
                    X[iSplit][iCase][idataSize] = dict()
                    y[iSplit][iCase][idataSize] = dict()
                    T[iSplit][iCase][idataSize] = dict()
                    P[iSplit][iCase][idataSize] = dict()

                    ##- X matrix
                    X_train = data[iSplit][iCase][idataSize]["train"][self.model_regressors]
                    X_test = data[iSplit][iCase][idataSize]["test"][self.model_regressors]
                    X[iSplit][iCase][idataSize]["train"] = X_train.values
                    X[iSplit][iCase][idataSize]["test"] = X_test.values

                    ##- Y vector
                    ##- Application completion time T(x) is stored in the dataset
                    ##- Price P(x) is proportional to the number of cores
                    T_train = data[iSplit][iCase][idataSize]["train"][self.target_column]
                    T_test = data[iSplit][iCase][idataSize]["test"][self.target_column]
                    P_train = np.array(list(map(price_function,X_train["nContainers"].values)))
                    P_test = np.array(list(map(price_function,X_test["nContainers"].values)))
                    if apply_log_y:
                        ##- We compute log(Y) as explained in [Villafan (2020); Chapter 3.1]
                        T[iSplit][iCase][idataSize]["train"] = np.log(T_train.values).reshape(-1,1)
                        T[iSplit][iCase][idataSize]["test"] = np.log(T_test.values).reshape(-1,1)
                        P[iSplit][iCase][idataSize]["train"] = np.log(P_train).reshape(-1,1)
                        P[iSplit][iCase][idataSize]["test"] = np.log(P_test).reshape(-1,1)
                        y[iSplit][iCase][idataSize]["train"] = np.log(T_train.values).reshape(-1,1) + np.log(P_train).reshape(-1,1)
                        y[iSplit][iCase][idataSize]["test"] = np.log(T_test.values).reshape(-1,1) + np.log(P_test).reshape(-1,1)
                    else:
                        T[iSplit][iCase][idataSize]["train"] = T_train.values.reshape(-1,1)
                        T[iSplit][iCase][idataSize]["test"] = T_test.values.reshape(-1,1)
                        P[iSplit][iCase][idataSize]["train"] = P_train.reshape(-1,1)
                        P[iSplit][iCase][idataSize]["test"] = P_test.reshape(-1,1)
                        y[iSplit][iCase][idataSize]["train"] = T_train.values.reshape(-1,1) * P_train.reshape(-1,1)
                        y[iSplit][iCase][idataSize]["test"] = T_test.values.reshape(-1,1) * P_test.reshape(-1,1)
        return X, y, T, P


    def __replace_test_values(self,X_train,X_test):
        """
            Replace DAG related features in the test set by their respective averages from the training data.

            Input: data - 2 key dictionary
                X_train dictionary obtained from generate_X()
            Input: data - 2 key dictionary
                X_test dictionary obtained from generate_X()
            Output: modified input X_test dictionary - 2 key dictionary
        """
        for iSplit in X_train.keys():
            for iCase in X_train[iSplit].keys():
                avrg_values = np.mean(X_train[iSplit][iCase][:,:-self.nonDAGfeatures_number],axis=0)
                X_test[iSplit][iCase][:,:-self.nonDAGfeatures_number] = avrg_values

        return X_test


    def __print_info(self):
        """
            Print some information regarding current dataset, 'dataSize' and 'nContainer' values.
        """
        ##- Generate dictionary of {dataSize: number of observations} and {nContainers: number of observations}
        lista_dataSize = list(self.dataset.loc[:,"dataSize"])
        ndataSize = {x: lista_dataSize.count(x) for x in lista_dataSize}
        lista_containers = list(self.dataset.loc[:,"nContainers"])
        ncontainers = {x: lista_containers.count(x) for x in lista_containers}

        print("Number of observations: ", len(self.dataset))
        print("Number of variables: ", len(self.dataset.columns))
        print("Number of regressors in {}-box models: {}".format(self.box_model,len(self.model_regressors)))
        print("All data sizes:")
        print(pd.DataFrame(ndataSize,index=['# observations']))
        print("All cores:")
        print(pd.DataFrame(ncontainers,index=['# observations']))


    def get_splits_and_cases_dict(self):
        '''
            Generate three dictionaries containing splits and cases based on current dataset_kind and split.

            Output: splits, cases - 2 key dictionaries
        '''
        return splits_and_cases_dict(self.dataset_kind,self.split)


    def print_info(self,X_train,X_test,iSplit,iCase):
        """
            Print information regarding dataset kind, 'dataSize' and 'nContainer' values.
                WARNING: generate_XY() method must first be called.
        """
        if not hasattr(self,"_generate_XY_called"):
            raise WrongCallToMethod("The method generate_XY() must first be called.")

        print("\nTRAINING is performed on the following 'dataSize' and 'nContainer' values:")
        df_X_train = pd.DataFrame(data=X_train[iSplit][iCase], columns=self.model_regressors)
        lista_dataSize = list(df_X_train.loc[:,"dataSize"])
        print(pd.DataFrame({x: lista_dataSize.count(x) for x in lista_dataSize},index=["# observations"]))
        lista_containers = list(df_X_train.loc[:,"nContainers"])
        print(pd.DataFrame({x: lista_containers.count(x) for x in lista_containers},index=["# observations"]))

        print("\nTESTING is performed on the following 'dataSize' and 'nContainer' values:")
        df_X_test = pd.DataFrame(data=X_test[iSplit][iCase], columns=self.model_regressors)
        lista_dataSize = list(df_X_test.loc[:,"dataSize"])
        print(pd.DataFrame({x: lista_dataSize.count(x) for x in lista_dataSize},index=["# observations"]))
        lista_containers = list(df_X_test.loc[:,"nContainers"])
        print(pd.DataFrame({x: lista_containers.count(x) for x in lista_containers},index=["# observations"]))


    def print_info_for_BO(self,X_train,X_test,iSplit,iCase):
        """
            Print information regarding dataset kind, 'dataSize' and 'nContainer' values.
                WARNING: generate_XY_for_BO() method must first be called.
        """
        if not hasattr(self,"_generate_XY_for_BO_called"):
            raise WrongCallToMethod("The method generate_XY_for_BO() must first be called.")

        for idataSize in X_train[iSplit][iCase].keys():

            print("\nTRAINING is performed on the following 'dataSize' and 'nContainer' values:")
            df_X_train = pd.DataFrame(data=X_train[iSplit][iCase][idataSize]["train"], columns=self.model_regressors)
            lista_dataSize = list(df_X_train.loc[:,"dataSize"])
            print(pd.DataFrame({x: lista_dataSize.count(x) for x in lista_dataSize},index=["# observations"]))
            lista_containers = list(df_X_train.loc[:,"nContainers"])
            print(pd.DataFrame({x: lista_containers.count(x) for x in lista_containers},index=["# observations"]))

            print("\nBO observations are taken also from these values:")
            df_X_train = pd.DataFrame(data=X_train[iSplit][iCase][idataSize]["test"], columns=self.model_regressors)
            lista_dataSize = list(df_X_train.loc[:,"dataSize"])
            print(pd.DataFrame({x: lista_dataSize.count(x) for x in lista_dataSize},index=["# observations"]))
            lista_containers = list(df_X_train.loc[:,"nContainers"])
            print(pd.DataFrame({x: lista_containers.count(x) for x in lista_containers},index=["# observations"]))

        for idataSize in X_test[iSplit][iCase].keys():

            print("\nTESTING may be performed on the following 'dataSize' and 'nContainer' values:")
            df_X_test = pd.DataFrame(data=X_test[iSplit][iCase][idataSize]["train"], columns=self.model_regressors)
            lista_dataSize = list(df_X_test.loc[:,"dataSize"])
            print(pd.DataFrame({x: lista_dataSize.count(x) for x in lista_dataSize},index=["# observations"]))
            lista_containers = list(df_X_test.loc[:,"nContainers"])
            print(pd.DataFrame({x: lista_containers.count(x) for x in lista_containers},index=["# observations"]))

            print("\nTESTING is performed AT LEAST on the following 'dataSize' and 'nContainer' values:")
            df_X_test = pd.DataFrame(data=X_test[iSplit][iCase][idataSize]["test"], columns=self.model_regressors)
            lista_dataSize = list(df_X_test.loc[:,"dataSize"])
            print(pd.DataFrame({x: lista_dataSize.count(x) for x in lista_dataSize},index=["# observations"]))
            lista_containers = list(df_X_test.loc[:,"nContainers"])
            print(pd.DataFrame({x: lista_containers.count(x) for x in lista_containers},index=["# observations"]))


def splits_and_cases_dict(dataset_kind,split):
    '''
        Generate three dictionaries containing splits and cases based on dataset_kind.

        Input: dataset_kind - string
            dataset_kind = ["Query26","Kmeans","SparkDL"]
        Input: split - string
            split = ["extrapolation","interpolation"]
        Output: splits, cases - 2 key dictionaries
    '''
    splits, cases = dict(), dict()

    if dataset_kind == "Query26":
        if split == "extrapolation":
            splits = {
                1: {"train": [250, 1000],
                    "test": [750]},
                2: {"train": [250, 750],
                    "test": [1000]}
            }
        if split == "interpolation":
            splits = {
                1: {"train": [250],
                    "test": [250]},
                2: {"train": [750],
                    "test": [750]},
                3: {"train": [1000],
                    "test": [1000]}
            }
        ##- Observations for which nContainers = 20 are dropped if dataset_kind = "Query26"
        cases = {
            1: {"train": [6, 10, 14, 18, 24, 28, 32, 36, 40, 44],
                "test": [8, 12, 16, 22, 26, 30, 34, 38, 42]},
            
            2: {"train": [6, 12, 18, 26, 32, 38, 44],
                "test": [8, 10, 14, 16, 22, 24, 28, 30, 34, 36, 40, 42]},
            
            3: {"train": [6, 14, 24, 32, 40, 44],
                "test": [8, 10, 12, 16, 18, 22, 26, 28, 30, 34, 36, 38, 42]},
            
            4: {"train": [6, 16, 30, 42, 44],
                "test": [8, 10, 12, 14, 18, 22, 24, 26, 28, 32, 34, 36, 38, 40]},
            
            5: {"train": [6, 8, 18, 32, 44],
                "test": [10, 12, 14, 16, 22, 24, 26, 28, 30, 34, 36, 38, 40, 42]},
            
            6: {"train": [6, 10, 16, 26, 44],
                "test": [8, 12, 14, 18, 22, 24, 28, 30, 32, 34, 36, 38, 40, 42]}
        }

    if dataset_kind == "Kmeans":
        if split == "extrapolation":
            splits = {
                1: {"train": [5, 10],
                    "test": [15, 20]},
                2: {"train": [5, 10, 15],
                    "test": [20]}
            }
        if split == "interpolation":
            splits = {
                1: {"train": [5],
                    "test": [5]},
                2: {"train": [10],
                    "test": [10]},
                3: {"train": [15],
                    "test": [15]},
                4: {"train": [20],
                    "test": [20]}
            }
        cases = {
            1: {"train": [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46],
                "test": [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]},
            
            2: {"train": [6, 12, 18, 24, 30, 36, 42, 48],
                "test": [8, 10, 14, 16, 20, 22, 26, 28, 32, 34, 38, 40, 44, 46]},
            
            3: {"train": [6, 14, 22, 30, 38, 48],
                "test": [8, 10, 12, 16, 18, 20, 24, 26, 28, 32, 34, 36, 40, 42, 44, 46]},
            
            4: {"train": [6, 16, 30, 36, 42, 48],
                "test": [8, 10, 12, 14, 18, 20, 22, 24, 26, 28, 32, 34, 38, 40, 44, 46]},
            
            5: {"train": [6, 8, 18, 30, 40, 48],
                "test": [10, 12, 14, 16, 20, 22, 24, 26, 28, 32, 34, 36, 38, 42, 44, 46]},
            
            6: {"train": [6, 10, 16, 24, 34, 48],
                "test": [8, 12, 14, 18, 20, 22, 26, 28, 30, 32, 36, 38, 40, 42, 44, 46]},
            
            7: {"train": [6, 16, 26, 36, 46],
                "test": [8, 10, 12, 14, 18, 20, 22, 24, 28, 30, 32, 34, 38, 40, 42, 44, 48]}
        }

    if dataset_kind == "SparkDL":
        if split == "extrapolation":
            splits = {
                1: {"train": [1000, 2000],
                    "test": [1500]},
                2: {"train": [1000, 1500],
                    "test": [2000]}
            }
        if split == "interpolation":
            splits = {
                1: {"train": [1000],
                    "test": [1000]},
                2: {"train": [1500],
                    "test": [1500]}
            }
        cases = {
            1: {"train": [2, 8, 14, 20, 26, 32, 38, 44],
                "test": [4, 6, 10, 12, 16, 18, 22, 24, 28, 30, 34, 36, 40, 42, 46, 48]},

            2: {"train": [2, 10, 18, 26, 34, 42],
                "test": [4, 6, 8, 12, 14, 16, 20, 22, 24, 28, 30, 32, 36, 38, 40, 44, 46, 48]},

            3: {"train": [2, 12, 22, 32, 42],
                "test": [4, 6, 8, 10, 14, 16, 18, 20, 24, 26, 28, 30, 34, 36, 38, 40, 44, 46, 48]}
        }

    return splits, cases
