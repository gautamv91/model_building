import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder

class DataPreprocessing:
    
    def __init__(self):
        
        return None
    
    def outlier_fit(df, col_list):
        outlier_config = dict()
        
        for col in col_list:
            q1 = np.percentile(df[col].dropna(), 25, method='midpoint')
            q3 = np.percentile(df[col].dropna(), 75, method='midpoint')
            iqr = q3 - q1
            upper_limit = round(q3 + (1.5*iqr),2)
            lower_limit = round(q1 - (1.5*iqr),2)
            
            outlier_config[col] = [lower_limit, upper_limit]
        
        return outlier_config

    # Outlier treatment will happen by capping the outliers to the upper & lower limit values

    def outlier_transform(config,df, col_list):
        
        for col in col_list:
            otl_vals = config[col]
            df[col] = np.where(df[col]<=otl_vals[0],otl_vals[0],np.where(df[col]>=otl_vals[1],otl_vals[1],df[col]))
        
        return df
    
    def missing_val_fit(df,num_vars,cat_vars,num_treat='median',cat_treat='mode'):
        replace_num_df = pd.DataFrame()
        if num_vars:
            replace_num_df = df[num_vars].median().reset_index()
            replace_num_df.columns = ['feature','replace_val']
        
        replace_cat_df = pd.DataFrame()
        if cat_vars:
            replace_cat_df = df[cat_vars].mode(axis=0,numeric_only=False,dropna=True).iloc[0,:].reset_index()
            replace_cat_df.columns = ['feature','replace_val']
            
        miss_config = replace_num_df.append(replace_cat_df, ignore_index=True)
        miss_config = miss_config.set_index('feature').T.to_dict('list')
        miss_config = {x:y[0] for x,y in miss_config.items()}
        
        return miss_config

    def missing_val_transform(df, missing_treat_config):
        
        final_df = df.fillna(value=missing_treat_config)
        
        return final_df
    
    def std_scaler_fit(train_data,scale_cols):
        
        ss_scaler = StandardScaler()
        
        scale_mod = ss_scaler.fit(train_data[scale_cols])
        scaled_feats = pd.DataFrame(scale_mod.transform(train_data[scale_cols]),columns = scale_cols)
        train_scale_df = train_data.loc[:,~train_data.columns.isin(scale_cols)]
        scaled_feats.index = train_scale_df.index
        # train_data.drop(num_feats,axis=1, inplace=True)
        train_scale_df = train_scale_df.merge(scaled_feats,how='inner',left_index=True, right_index=True)
        
        return train_scale_df, scale_mod

    def std_scaler_transform(scale_config, data, scale_cols):
        
        scaled_feats = pd.DataFrame(scale_config.transform(data[scale_cols]),columns = scale_cols)
        scale_df = data.loc[:,~data.columns.isin(scale_cols)]
        scaled_feats.index = scale_df.index
        scale_df = scale_df.merge(scaled_feats,how='inner',left_index=True, right_index=True)
        
        return scale_df
    
    def label_enc_fit(train_data, mapping=None):
        cols = list(train_data.columns)
        
        if mapping is not None:
            mapping_cols = list(mapping.keys())
            sk_le_cols = list(set(cols)-set(mapping_cols))
        
        le_config = dict()
        le_results = dict()
        
        if sk_le_cols:
            for i in sk_le_cols:
                le = LabelEncoder()
                le.fit(train_data[i])
                le_config[i] = {"type":"sklearn","encoder":le}
                le_results[i] = [i for i in zip(range(len(le.classes_)),le.classes_)]
        if mapping_cols:
            for j in mapping_cols:
                le_config[j] = {"type":"custom","encoder":mapping[j]}
                le_results[j] = [(val,key) for key,val in mapping[j].items()]
            
            
        return le_config, le_results

    def label_enc_transform(data, le_config):
        cols = list(le_config.keys())
        le_data = data.copy(deep=True)
        
        for i in cols:
            le_type = le_config[i]["type"]
            col_le = le_config[i]["encoder"]
            if le_type=='sklearn':
                le_data[i] = col_le.transform(le_data[i])
            else:
                le_data[i]=le_data[i].map(col_le)
            
        return le_data
    
    def freq_enc_fit(df, cat_cols):
        freq_enc_map = dict()
        
        for i in cat_cols:
            freq_data = df[i].value_counts(normalize=True)
            freq_enc_map[i] = dict()
            freq_enc_map[i]['freq_map'] = freq_data.to_dict()
            freq_enc_map[i]['freq_mode'] = freq_data[0]
            
        return freq_enc_map

    def freq_enc_transform(df, freq_config):
        
        for key,value in freq_config.items():
            cat_freq_map = value['freq_map']
            df[key] = df[key].map(cat_freq_map)
            df.loc[df[key].isna(),key] = value['freq_mode']
        
        return df
    
    def ohe_fit(df, cols):
        ohe = OneHotEncoder(handle_unknown='ignore')
        df_subset = df[cols].copy(deep=True)
    
        ohe_mod = ohe.fit(df_subset)
    
        new_col_names = dict()
    
        for i in cols:
            unq_vals = list(df[i].unique())
            new_col_names[i] = [i+"_"+str(j) for j in unq_vals]
            new_col_names[i].sort()
    
        return ohe_mod,new_col_names

    def ohe_transform(df, cols, ohe_mod, new_col_names, drop_vals=None):
        ohe_new_cols = list()
    
        for col in cols:
            ohe_new_cols.extend(new_col_names[col])
    
        new_df = pd.DataFrame(ohe_mod.transform(df[cols]).toarray(), columns=ohe_new_cols)
    
        # creating the list of transformed features to drop if dummy encoding is required
        if drop_vals is not None:
            cols_to_drop = list()
            for key,val in drop_vals.items():
                cols_to_drop.append(key+"_"+str(val))
    
            new_df.drop(columns=cols_to_drop, inplace=True)
    
        return new_df
    
    