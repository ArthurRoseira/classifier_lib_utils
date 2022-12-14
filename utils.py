
import re
# from src.libs import params
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import ElasticNetCV, LinearRegression,LassoCV,RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,recall_score,roc_auc_score,accuracy_score,precision_score,roc_curve,auc, confusion_matrix
from scikitplot.helpers import binary_ks_curve
import scikitplot as skplt
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_theme(style="darkgrid")

class Models:

    def __init__(self):
        self.regression_models = ['xgboost','lightgbm','randomforest','adaboost','linear','elasticnet','lasso','ridge']
        self.classification_models = ['xgboost','lightgbm','randomforest','adaboost','logistic']

    def get_regression_model(self,model_or_name, threads=-1, seed=0,**param_dist):

        regression_models = {
            'xgboost': {'model':XGBRegressor,'params':{'max_depth':6, 'n_jobs':threads, 'random_state':seed},'name':'XGBoostRegressor'},
            'lightgbm': {'model':LGBMRegressor,'params':{'n_jobs':threads, 'max_depth':2, 'random_state':seed, 'verbose':-1},'name':'LGBMRegressor'},
            'randomforest': {'model':RandomForestRegressor,'params':{'n_estimators':100, 'max_depth':3, 'n_jobs':threads}, 'name':'RandomForestRegressor'},
            'adaboost': {'model':AdaBoostRegressor,'params':{},'name':'AdaBoostRegressor'},
            'linear': {'model':LinearRegression,'params':{},'name':'LinearRegression'},
            'elasticnet': {'model':ElasticNetCV,'params':{'positive':True}, 'name':'ElasticNetCV'},
            'lasso': {'model':LassoCV,'params':{'positive':True},'name':'LassoCV'},
            'ridge': {'model':RidgeCV,'params':{},'name':'Ridge'},
        }
        if isinstance(model_or_name, str):
            
            model_and_name = regression_models.get(model_or_name.lower())
            if not model_and_name:
                raise Exception("unrecognized model: '{}'".format(model_or_name))
            else:
                model_dict = model_and_name
                model_dict['params'] = {**model_dict['params'],**param_dist}
            if param_dist == None:
                return model_dict['model'](), model_dict['name']
            else:
                return model_dict['model'](**model_dict['params']), model_dict['name']
        else:
            model = model_or_name
            name = re.search("\w+", str(model)).group(0)
            return model,name

    def get_classification_model(self,model_or_name, threads=-1, seed=0,**param_dist):

        classification_models = {
            'xgboost': {'model':XGBClassifier,'params':{'max_depth':6, 'n_jobs':threads, 'random_state':seed},'name':'XGBoostClassifier'},
            'lightgbm': {'model':LGBMClassifier,'params':{'n_jobs':threads, 'max_depth':2, 'random_state':seed, 'verbose':-1},'name':'LGBMClassifier'},
            'randomforest': {'model':RandomForestClassifier,'params':{'n_estimators':100, 'max_depth':3, 'n_jobs':threads}, 'name':'RandomForestClassifier'},
            'adaboost': {'model':AdaBoostClassifier,'params':{},'name':'AdaBoostRegressor'},
            'logistic':{'model':LogisticRegressionCV,'params':{},'name':'LogisticRegression'}
        }
        if isinstance(model_or_name, str):
            
            model_and_name = classification_models.get(model_or_name.lower())
            if not model_and_name:
                raise Exception("unrecognized model: '{}'".format(model_or_name))
            else:
                model_dict = model_and_name
                model_dict['params'] = {**model_dict['params'],**param_dist}
            if param_dist == None:
                return model_dict['model'](), model_dict['name']
            else:
                return model_dict['model'](**model_dict['params']), model_dict['name']
        else:
            model = model_or_name
            name = re.search("\w+", str(model)).group(0)
            return model,name



    def evaluate_regression(self,y_pred,y,p_feat,n):
        performances = {}
        performances['MSE']=mean_squared_error(y, y_pred)
        performances['RSME']=math.sqrt(mean_squared_error(y, y_pred))
        performances['MAE']=mean_absolute_error(y, y_pred)
        performances['R2']=r2_score(y, y_pred)
        performances['R2_adjusted']= 1 - (1-r2_score(y, y_pred)) * (n - 1) /(n - p_feat - 1)
        return performances

    def evaluate_classification(self,y_pred,y,y_proba):
        performances = {}
        performances['recall']=recall_score(y, y_pred)
        performances['AUC_score']=roc_auc_score(y, y_proba)
        performances['accuracy']=accuracy_score(y, y_pred)
        performances['precision']=precision_score(y, y_pred)
        return performances

    def print_scatter(self,x,y,label_x,label_y):   
        fig = plt.figure()
        ax = fig.add_subplot(111)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, y)
        ax = sns.regplot(x=x, y=y, line_kws={
                        'color': 'red', 'label': "y={0:.1f}x+{1:.1f}".format(slope, intercept)})
        #sns.kdeplot(x, y)
        ax.legend(loc='upper left')
        ax.set(xlabel=label_x, ylabel=label_y, title= label_x+" x "+label_y)
        plt.legend(loc='upper left');
        plt.show()


    def print_roc_curve(self,y,y_pred_proba):
        fpr, tpr, _  = roc_curve(y, y_pred_proba[:, 1])
        fpr_micro, tpr_micro, _ = roc_curve(y.ravel(), y_pred_proba[:, 1].ravel())
        roc_auc = auc(fpr_micro, tpr_micro)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.show()

    def confusion_matrix(self,y,y_pred):

        cf_matrix = confusion_matrix(y, y_pred)
        fig, axes = plt.subplots(1, 2,sharex=True,sharey=True,figsize=(12.8,9))
        fig.suptitle('Confusion Matrix')
        axes[0].set_title('Absolute')
        axes[1].set_title('Percent')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        sns.heatmap(cf_matrix,ax=axes[0] ,annot=True,fmt="d")
        sns.heatmap(cf_matrix/np.sum(cf_matrix), ax=axes[1] ,annot=True, fmt='.2%', cmap='Blues')
        plt.show()

    def ks_centil100(self,data:pd.DataFrame,target:str, prob:str)->pd.DataFrame:
        data['prob0'] = 1 - data[prob]
        data['target0'] = 1 - data[target]
        data['bucket100'] = pd.qcut(data['prob0'], 100)
    #     decil_i = sorted(data.bucket.unique())[0]
    #     data = data[data.bucket == decil_i]
    #     data['bucket100'] = pd.qcut(data['prob0'], 10)
        grouped = data.groupby('bucket100', as_index = False)
        kstable = pd.DataFrame()
        kstable['min_prob'] = grouped['prob0'].min()['prob0']
        kstable['max_prob'] = grouped['prob0'].max()['prob0']
        kstable['events']   = grouped[target].sum()[target]
        kstable['nonevents'] = grouped['target0'].sum()['target0']
        kstable = kstable.sort_values(by="min_prob", ascending=True).reset_index(drop = True)
        kstable['event_rate'] = (kstable.events / (kstable.events + kstable.nonevents)).apply('{0:.2%}'.format)
        kstable['nonevent_rate'] = (kstable.nonevents / (kstable.events + kstable.nonevents)).apply('{0:.2%}'.format)
        kstable['cum_eventrate']=(kstable.events).cumsum()
        kstable['perc_cum_eventrate']=(kstable.cum_eventrate/data[target].sum())
        kstable['cum_noneventrate']=(kstable.nonevents).cumsum()
        kstable['perc_cum_noneventrate']=(kstable.cum_noneventrate/data['target0'].sum())
        kstable['KS'] = np.round(kstable['perc_cum_eventrate']-kstable['perc_cum_noneventrate'], 3) * 100
        #Formating
    #     kstable['perc_cum_eventrate']= kstable['perc_cum_eventrate'].apply('{0:.2%}'.format)
    #     kstable['perc_cum_noneventrate']= kstable['perc_cum_noneventrate'].apply('{0:.2%}'.format)
        kstable.index = range(1,101)
        kstable.index.rename('Centile_100', inplace=True)
        pd.set_option('display.max_columns', 9)
        # #Display KS
    #     from colorama import Fore
    #     print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at centile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
        print("KS is " + str(max(kstable['KS']))+"%"+ " at centile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
        return(kstable)
    #     return(kstable[['min_prob', 'max_prob', 'events

    def plot_stacked(self,data_plot:pd.DataFrame,pivot_column:str,stack_column:str,ylim:int):
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        data_plot.pivot(columns=pivot_column)[stack_column].plot(kind = 'hist', bins= 50,stacked=True,ax=axes)
        axes.set_title('Distribuição')
        axes.set_ylabel('Contagem')
        axes.set_xlabel('Score')
        axes.set_ylim([0,ylim])
        plt.plot()

    def plot_ks_curve(self,y,y_proba):
        y_true = np.array(y)
        y_probas = np.array(y_proba)

        classes = np.unique(y_true)
        if len(classes) != 2:
            raise ValueError('Cannot calculate KS statistic for data with '
                            '{} category/ies'.format(len(classes)))

        # Compute KS Statistic curves
        thresholds, pct1, pct2, ks_statistic, \
            max_distance_at, classes = binary_ks_curve(y_true,
                                                    y_probas[:, 1].ravel())
        print('KS Statistic: {:.3f} at {:.3f}'.format(ks_statistic,
                                                                    max_distance_at))
        skplt.metrics.plot_ks_statistic(y_true, y_probas)
        plt.show()


if __name__=="__main__":
    
    from sklearn.datasets import make_regression
    # generate regression dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

    models = Models()
    models.ks_centil100()
    


