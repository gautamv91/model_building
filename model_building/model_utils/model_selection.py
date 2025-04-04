import pandas as pd
import numpy as np
from . import model_constants as mc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
        roc_auc_score, mean_absolute_percentage_error, mean_squared_error, \
        root_mean_squared_error, r2_score
        
import seaborn as sns
import matplotlib.pyplot as plt


class ModelSelection:

    def __init__(self):
        self.classifiers_ = mc.CLASSIFIERS
        self.regressors_ = mc.REGRESSORS
        self.all_models = mc.ALL_MODELS
        self.all_binary_metrics = mc.ALL_METRIC_NAMES_BIN
        self.all_regression_metrics = mc.ALL_METRIC_NAMES_REG


    def model_init(self, model_list):

        return None
    
    #### Code for Hyper-parameter tuning using Optuna
    def objective_optuna(self, model_type, ):
        
        pass
    
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
        elif objective==mc.REGRESSION:
            for i in range(len(model_objs['models'])):
                model_obj = model_objs['models'][i]
                model_name = model_objs['model_name'][i]
                
                train_pred = model_obj.predict(xtrain)
                test_pred = model_obj.predict(xtest)
                
                data_dict = {'train': [ytrain, train_pred], 'test': [ytest, test_pred]}
                
                if metrics_list == mc.ALL_METRICS:
                    metrics = self.all_binary_metrics
                else:
                    metrics = metrics_list
                
                for key, data in data_dict.items():
                    metrics_vals = list()
                    
                    if 'mse' in metrics:
                        mse = mean_squared_error(data[0], data[1])
                        metrics_vals.append(mse)
                    if 'rmse' in metrics:
                        rmse = root_mean_squared_error(data[0], data[1])
                        metrics_vals.append(rmse)
                    if 'mape' in metrics:
                        mape = mean_absolute_percentage_error(data[0], data[1])
                        metrics_vals.append(mape)
                    if 'r2' in metrics:
                        r2 = r2_score(data[0], data[1])
                        metrics_vals.append(r2)
                        
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
        
        return metrics_df.round(5)
    
    def choose_best_model(self, metrics_df, eval_metric):
        metrics = metrics_df.reset_index()
        if eval_metric in self.all_binary_metrics:
            best_idx = metrics.loc[metrics['metrics']==eval_metric,['test']].idxmax()
        else:
            best_idx = metrics.loc[metrics['metrics']==eval_metric,['test']].idxmin()
            
        best_model_name = metrics.iloc[best_idx,0].values[0]
        best_val = round(metrics.loc[best_idx,'test'].values[0],4)
        print(f'Based on the {eval_metric} values the {best_model_name} model has the best performance ({best_val}) on the validation dataset.')
        
    def model_coeff_linear(self, model_obj):
        
        coef_df = pd.DataFrame({'feature_name':model_obj.feature_names_in_,'coefficients':model_obj.coef_.tolist()})
        coef_df = pd.concat([coef_df,pd.DataFrame({'feature_name':['intercept'],'coefficients':model_obj.intercept_})],
                                      axis=0,ignore_index=True)
        
        return coef_df.round(4)
    
    def get_feature_importances(self, model_obj_dict):
        
        feat_imp_df = pd.DataFrame()
        for i in range(len(model_obj_dict['models'])):
            model_obj = model_obj_dict['models'][i]
            model_name = model_obj_dict['model_name'][i]
            hp_tuned = model_obj_dict['hp_tuned'][i]
            
            if 'nn' not in model_name:
                if hp_tuned == 'N':
                    try:
                        fe_imps = model_obj.coef_.tolist()
                    except:
                        fe_imps = model_obj.coef_.feature_importances_.tolist()
                        
                else:
                    try:
                        fe_imps = model_obj.best_estimator_.coef_.tolist()
                    except:
                        fe_imps = model_obj.best_estimator_.feature_importances_.tolist()
                
                if i==0:
                    feat_imp_df = pd.DataFrame({'feature_name':model_obj.feature_names_in_, 
                                                model_name:fe_imps})  
                else:
                    feat_imp_df[model_name] = fe_imps
                
        return feat_imp_df.round(4)
    
    def plot_actual_v_pred(self, y_actual, y_pred):
        y_act_sort = y_actual.sort_values(ascending=True)
        y_pred_sort = pd.Series(y_pred, index=y_actual.index).reindex(y_act_sort.index)
        y_act_sort.reset_index(drop=True, inplace=True)
        y_pred_sort.reset_index(drop=True, inplace=True)
        
        plt.figure(figsize=(15, 13))
        plt.plot(y_act_sort.index, y_act_sort, linestyle="-", color='blue', label='actual')
        plt.plot(y_act_sort.index, y_pred_sort, linestyle="-", color='red', label='pred')
        plt.legend()
        plt.show()
        return None
        