import pandas as pd
import numpy as np
from . import model_constants as mc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
        roc_auc_score


class ModelSelection:

    def __init__(self):
        self.classifiers_ = mc.CLASSIFIERS
        self.regressors_ = mc.REGRESSORS
        self.all_models = mc.ALL_MODELS
        self.all_binary_metrics = mc.ALL_METRIC_NAMES_BIN


    def model_init(self, model_list):

        return None
    
    def train_test_compare(self, model_objs, xtrain, xtest, ytrain, ytest
                           , metrics_list=mc.ALL_METRICS, objective=mc.BINARY):
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
            for i in range(len(model_objs['models'])):
                model_obj = model_objs['models'][i]
                model_name = model_objs['model_name'][i]
                
                if 'nn' in model_name:
                    train_pred = np.where(model_obj.predict(xtrain)>=0.5,1,0)
                    test_pred = np.where(model_obj.predict(xtest)>=0.5,1,0)
                    train_pred_prob = model_obj.predict(xtrain)
                    test_pred_prob = model_obj.predict(xtest)
                else:
                    train_pred = model_obj.predict(xtrain)
                    test_pred = model_obj.predict(xtest)
                    train_pred_prob = model_obj.predict_proba(xtrain)[:,1]
                    test_pred_prob = model_obj.predict_proba(xtest)[:,1]
                    
                data_dict = {'train': [ytrain, train_pred, train_pred_prob], 'test': [ytest, test_pred, test_pred_prob]}
                
                if metrics_list == mc.ALL_METRICS:
                    metrics = self.all_binary_metrics
                else:
                    metrics = metrics_list
                    
                for key, data in data_dict.items():
                    metrics_vals = list()
                    
                    if 'accuracy' in metrics:
                        acc = accuracy_score(data[0], data[1])
                        metrics_vals.append(acc)
                    if 'precision' in metrics:
                        pr = precision_score(data[0], data[1])
                        metrics_vals.append(pr)
                    if 'recall' in metrics:
                        rcl = recall_score(data[0], data[1])
                        metrics_vals.append(rcl)
                    if 'f1 score' in metrics:
                        f1 = f1_score(data[0], data[1])
                        metrics_vals.append(f1)
                    if 'au-roc' in metrics:
                        auc = roc_auc_score(data[0], data[2])
                        metrics_vals.append(auc)
    
                    if key == 'train':
                        train = metrics_vals
                    else:
                        test = metrics_vals

                metrics_df = pd.concat( [metrics_df, 
                                       pd.DataFrame({'model':[model_name for i in metrics], 
                                                     'metrics': metrics, 'train': train, 'test': test})]
                                       , axis=0, ignore_index=True
                                       )
        metrics_df.set_index(['model', 'metrics'], inplace=True)
        
        return metrics_df
    
    def choose_best_model(self, metrics_df, eval_metric):
        metrics = metrics_df.reset_index()
        best_idx = metrics.loc[metrics['metrics']==eval_metric,['test']].idxmax()
        best_model_name = metrics.iloc[best_idx,0].values[0]
        best_val = round(metrics.loc[best_idx,'test'].values[0],4)
        print(f'Based on the {eval_metric} values the {best_model_name} model has the best performance ({best_val}) on the validation dataset.')
        
    def model_coeff_linear(self, model_obj):
        
        coef_df = pd.DataFrame({'feature_name':model_obj.feature_names_in_,'coefficients':model_obj.coef_[0].tolist()})
        coef_df = pd.concat([coef_df,pd.DataFrame({'feature_name':['intercept'],'coefficients':model_obj.intercept_})],
                                      axis=0,ignore_index=True)
        
        return coef_df
        