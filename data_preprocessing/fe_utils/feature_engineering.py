import numpy as np
import pandas as pd

class FeatureEngineering():
    
    def __init__(self):
        
        return None
    
    def drop_cols(self, df, col_list):
        
        df.drop(columns=col_list, inplace=True)

    def extract_substr(self, var_data,start,end=None,dtype=None):
        
        if end is None and dtype is None:
            return var_data.str.strip().str[start:]
        elif end is None and dtype is not None:    
            return var_data.str.strip().str[start:].astype(dtype)
        elif end is not None and dtype is None:
            return var_data.str.strip().str[start:end]
        
    def create_flags(self, df, flag_map):
        
        for i in flag_map.keys():
            flag_val = flag_map[i]
            df[i] = np.where(df[i]>flag_val,1,0)
            
        return df
    
    def apply_grouping(self, var_data, grp_config):
        
        for key,val in grp_config.items():
            var_data[var_data.isin(val)] = key
            
        return var_data
