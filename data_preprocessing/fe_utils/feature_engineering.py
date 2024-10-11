import numpy as np
# import pandas as pd

class FeatureEngineering:


    def __init__(self):
        
        return None
    
    def drop_cols(self, df, col_list):
        """
        This function is used to drop columns from a dataframe
        :param df: Input data
        :param col_list: List of columns to drop
        :return: None
        """
        df.drop(columns=col_list, inplace=True)

    def extract_substr(self, var_data, start, end=None, dtype=None):
        """
        This function is used to extract parts of a string in a pandas Series.
        :param var_data: Pandas Series data
        :param start: Starting index of sub-string
        :param end: Optional - Will be used as end of sub-string if provided
        :param dtype: Optional - The data type that the sub-string needs to be converted to. Eg: 'int', 'float'
        :return: Pandas Series data
        """
        if end is None and dtype is None:
            return var_data.str.strip().str[start:]
        elif end is None and dtype is not None:    
            return var_data.str.strip().str[start:].astype(dtype)
        elif end is not None and dtype is None:
            return var_data.str.strip().str[start:end]
        
    def create_flags(self, df, flag_map):
        """
        This function is used to convert numeric columns into flag columns based on the threshold provided.
        :param df: Input data
        :param flag_map: Dict - Containing columns as keys and threshold for flag as values.
        :return: Dataframe
        """
        for i in flag_map.keys():
            flag_val = flag_map[i]
            df[i] = np.where(df[i] > flag_val, 1, 0)
            
        return df
    
    def apply_grouping(self, var_data, grp_config):
        """
        This function will apply the given grouping for a categorical variable.
        :param var_data: Pandas Series data
        :param grp_config: Dict - Containing original string as key and replacement as value
        :return: Pandas Series data
        """
        for key, val in grp_config.items():
            var_data[var_data.isin(val)] = key
            
        return var_data
