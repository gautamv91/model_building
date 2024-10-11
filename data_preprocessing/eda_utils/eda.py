"""
This class 'EDA' consists of functions that are usefull for performing exploratory data analysis
on datasets.
"""

import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    
    def __init__(self):
        return None
    
    def count_na(self, col):
        """
        This function is used to count the number of missing values in a column of a pandas DF.
        :param col: Name of the column
        :return: int - Count of missing values in the column
        """
        return col.isna().sum()

    def percentile_25(self, col):
        """
        This function is used to calculate the 25th percentile of a column in a pandas DF.
        :param col: Name of the column
        :return: float - 25th percentile of the column
        """
        return col.quantile(.25)

    def percentile_75(self, col):
        """
        This function is used to calculate the 75th percentile of a column in a pandas DF.
        :param col: Name of the column
        :return: float - 75th percentile of the column
        """
        return col.quantile(.75)

    def create_data_summary(self, df, dtypes_df):
        """
        This function creates a dataframe with the summary stats like count of unique values,
        count of rows, count of missing values, min, max, mean, median, etc.
        :param df: Input data
        :param dtypes_df: DF containing the column names and datatypes for each column of "df"
        :return: DF - Dataframe with the summary stats
        """
        num_agg = ['nunique', 'count', self.count_na, 'min', 'max', self.percentile_25, 'mean', 'median'
                   , self.percentile_75, 'std']
        cat_agg = ['nunique', 'count', self.count_na]
        
        agg_dict = dict()
        
        for i in dtypes_df.index:
            if dtypes_df.iloc[i,1]=='object':
                agg_dict[dtypes_df.iloc[i,0]] = cat_agg
            else:
                agg_dict[dtypes_df.iloc[i,0]] = num_agg
        
        data_summary = df.agg(agg_dict)
        data_summary = round(data_summary.T.reset_index(names='col_name'),2)
        
        # Calculating the mode for each feature. If there are multiple modes then the 1st values
        # will be considered as the mode value for that feature.
        mode_df = df.mode(axis=0,numeric_only=False,dropna=True).iloc[0,:].reset_index()
        mode_df.columns = ['col_name', 'mode']
        
        data_summary = pd.merge(data_summary, mode_df, how='inner', on='col_name')
        data_summary = data_summary[['col_name', 'nunique', 'count', 'count_na', 'mode', 'min', 'max'
                                    , 'percentile_25', 'mean', 'median', 'percentile_75', 'std']]
        
        return data_summary
    
    def eda_plots(self):
        """
        This function contains multiple plot functions that can be used for
        :return: None
        """
        
        def dist_plot(df, cols, bin_num=10):
            """
            This creates a histogram for each numeric column from the "cols" list.
            :param df: Input dataframe
            :param cols: List of columns
            :param bin_num: Number of bins to be created. Default is 10.
            :return: None
            """
            for i in cols:
                sns.set_style('whitegrid')
                sns.distplot(df[i], kde=False, color='red', bins=bin_num)
                plt.title(f'Distribution plot of {i}')
                plt.show()
             
            return None
        
        def count_plot(df, cols, split_by=None):
            """
            This creates a count plot for each categorical column from the "cols" list.
            :param df: Input dataframe
            :param cols: List of columns
            :param split_by: Optional parameter that'll be used to split the data by the categorical variable provided
            :return: None
            """
            for i in cols:
                sns.countplot(x=i, data=df, order=df[i].value_counts().index, hue=split_by)
                plt.title(f'Count plot of {i}')
                plt.show()
        
            return None
        
        def box_plots(df, cols):
            """
            This function will create box-plots for the list of numeric columns provided.
            :param df: Input data
            :param cols: List of numeric columns
            :return: None
            """
            for i in cols:
                sns.boxplot(df[i])
                plt.title(f'Box Plot of {i}')
                plt.show()
            
            return None
        
        def bivar_box_plt(df, cat_cols, num_cols):
            """
            This function will create a grid of box-plots based on all combinations of numeric & categorical columns,
            that can be used for bi-variate analysis.
            :param df: Input data
            :param cat_cols: List of categorical columns
            :param num_cols: List of numerical columns
            :return: None
            """
            num_len = len(num_cols)
            cat_len = len(cat_cols)
            grid_rows = grid_cols = 1
            
            if num_len > cat_len:
                grid_rows = num_len
                grid_cols = cat_len
                row_var = num_cols
                col_var = cat_cols
            else:
                grid_rows = cat_len
                grid_cols = num_len
                row_var = cat_cols
                col_var = num_cols
            
            figure, axis = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*3,grid_rows*2))
            figure.tight_layout()
            
            for i in range(grid_rows):
                for j in range(grid_cols):
                    sns.boxplot(ax=axis[i, j], x=row_var[i], y=col_var[j], data=df)
                    
            plt.show()
                    
            return None
        
        def correlation(data, fig_size):
            """
            This function will plot the correlation matrix of the given data.
            :param data: Input data
            :param fig_size: Tuple - Containing height and width of the plot. Eg: (4,4)
            :return: None
            """
            corr = data.corr()

            sns.set(rc={"figure.figsize": fig_size})
            sns.heatmap(corr, annot=True)

            sns.set(rc={"figure.figsize": (3, 4)})
            
            return None
    
    