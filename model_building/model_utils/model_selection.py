import pandas as pd
import numpy as np
from model_building.model_utils import model_constants as mc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
        roc_auc_score


class ModelSelection:

    def __init__(self):
        self.classifiers_ = mc.CLASSIFIERS
        self.regressors_ = mc.REGRESSORS
        self.all_models = mc.ALL_MODELS


    def model_init(self, model_list):

        return None
    
    def train_test_compare(self, model_obj, xtrain, xtest, ytrain, ytest, objective=mc.BINARY):
        """
        This function return a dataframe with the classification metrics comparison
        between train & test datasets.

        Parameters
        ----------
        model_obj : Model fit object
            The final model object which is to be used to calculate the metrics.
        xtrain : pd.DataFrame
            Train dataset with the independent features.
        xtest : pd.DataFrame
            Test dataset with the independent features.
        ytrain : pd.Series
            Target variable data for the train dataset.
        ytest : pd.Series
            Target variable data for the test dataset.
        objective : str
            String value that indicates the type of problem being solved. Binary, multi-class or regression.

        Returns
        -------
        pd.DataFrame - A Dataframe containing.

        """
        
        metrics_df = pd.DataFrame()
        train = list()
        test = list()

        if objective == mc.BINARY:
            train_pred = model_obj.predict(xtrain)
            test_pred = model_obj.predict(xtest)

            data_dict = {'train': [ytrain, train_pred], 'test': [ytest, test_pred]}

            metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'au-roc']
            for key, data in data_dict.items():

                acc = accuracy_score(data[0], data[1])
                pr = precision_score(data[0], data[1])
                rcl = recall_score(data[0], data[1])
                f1 = f1_score(data[0], data[1])
                auc = roc_auc_score(data[0], data[1])

                if key == 'train':
                    train = [acc, pr, rcl, f1, auc]
                else:
                    test = [acc, pr, rcl, f1, auc]

            metrics_df = pd.DataFrame({'metrics': metrics,
                                       'train': train,
                                       'test': test})
        
        return metrics_df
