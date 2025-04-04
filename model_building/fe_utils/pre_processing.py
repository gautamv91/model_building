import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


class DataPreprocessing:
    
    def __init__(self):
        
        return None
    
    def outlier_fit(self, df, col_list):
        """
        This is the fit function for outlier treatment, which returns a config file with the limits.

        :param df: Input data.
        :param col_list: List of numerical columns to be treated for outliers.
        :return: Dict - Config file with the lower and upper limits for clipping outliers.
        """
        outlier_config = dict()
        
        for col in col_list:
            q1 = np.percentile(df[col].dropna(), 25, method='midpoint')
            q3 = np.percentile(df[col].dropna(), 75, method='midpoint')
            iqr = q3 - q1
            upper_limit = round(q3 + (1.5*iqr), 2)
            lower_limit = round(q1 - (1.5*iqr), 2)
            
            outlier_config[col] = [lower_limit, upper_limit]
        
        return outlier_config

    # Outlier treatment will happen by capping the outliers to the upper & lower limit values

    def outlier_transform(self, config, df):
        """
        This is the transform function for outlier treatment, which returns the transformed data.

        :param config: Dict - Config file from the fit function.
        :param df: Input data.
        :return: DataFrame - Containing the treated data.
        """
        for col in config.keys():
            otl_vals = config[col]
            df[col] = np.where(df[col] <= otl_vals[0], otl_vals[0], np.where(df[col] >= otl_vals[1], otl_vals[1], df[col]))
        
        return df
    
    def missing_val_fit(self, df, num_vars, cat_vars, num_treat='median', cat_treat='mode'):
        """
        This is the fit function for missing value treatment, which returns a config file with the replacement values.

        :param df: Input data.
        :param num_vars: List of numerical features.
        :param cat_vars: List of categorical features.
        :param num_treat: Treatment method for numerical features. Default is median.
        :param cat_treat: Treatment method for categorical features.Default is mode.
        :return: Dict - Containing the replacement values for each column.
        """
        replace_num_df = pd.DataFrame()
        if num_vars:
            if num_treat == 'median':
                replace_num_df = df[num_vars].median().reset_index()
            elif num_treat == 'mean':
                replace_num_df = df[num_vars].mean().reset_index()
            replace_num_df.columns = ['feature', 'replace_val']
                
        
        replace_cat_df = pd.DataFrame()
        if cat_vars:
            if cat_treat == 'mode':
                replace_cat_df = df[cat_vars].mode(axis=0, numeric_only=False, dropna=True).iloc[0,:].reset_index()
                replace_cat_df.columns = ['feature', 'replace_val']
            
        # miss_config = replace_num_df.append(replace_cat_df, ignore_index=True)
        miss_config = pd.concat([replace_num_df, replace_cat_df], ignore_index=True, axis=0)
        miss_config = miss_config.set_index('feature').T.to_dict('list')
        miss_config = {x: y[0] for x, y in miss_config.items()}
        
        return miss_config

    def missing_val_transform(self, df, missing_treat_config):
        """
        This is the transform function for missing value treatment.

        :param df: Input data.
        :param missing_treat_config: Config file from the fit function.
        :return: DataFrame - Treated data.
        """

        final_df = df.fillna(value=missing_treat_config)
        
        return final_df
    
    def std_scaler_fit(self, df, scale_cols):
        """
        This is the fit function for standard scaler.

        :param df: Input data.
        :param scale_cols: List of columns to be scaled.
        :return: Scaling model.
        """
        ss_scaler = StandardScaler()
        
        scale_mod = ss_scaler.fit(df[scale_cols])

        # scaled_feats = pd.DataFrame(scale_mod.transform(df[scale_cols]), columns=scale_cols)
        # train_scale_df = df.loc[:, ~df.columns.isin(scale_cols)]
        # scaled_feats.index = train_scale_df.index
        # train_scale_df = train_scale_df.merge(scaled_feats, how='inner', left_index=True, right_index=True)
        
        # return train_scale_df, scale_mod
        return scale_mod

    def std_scaler_transform(self, scale_config, data, scale_cols, precision=4):
        """
        This is the transform function for scaling the data.

        :param scale_config: Scale_mod from the fit function.
        :param data: Input data.
        :param scale_cols: list of columns to be scaled.
        :return: DataFrame - Treated data.
        """
        scaled_feats = pd.DataFrame(scale_config.transform(data[scale_cols]), columns=scale_cols).round(precision)
        scale_df = data.loc[:, ~data.columns.isin(scale_cols)]
        scaled_feats.index = scale_df.index
        # scale_df = scale_df.merge(scaled_feats, how='inner', left_index=True, right_index=True)
        scale_df = pd.concat([scale_df, scaled_feats], axis=1)
        
        return scale_df
    
    def label_enc_fit(self, df, mapping=None):
        """
        This is the fit functino for label encoding of categorical features.

        :param df: Input data.
        :param mapping: Optional - Dict - Used to provide custom labels to categories.
        :return: Tuple - (config, results) - The config file is to be passed to the transform function &
        the results file can be used for plotting.
        """
        cols = list(df.columns)
        mapping_cols = []
        
        if mapping is not None:
            mapping_cols = list(mapping.keys())
            sk_le_cols = list(set(cols)-set(mapping_cols))
        else:
            sk_le_cols = cols
        
        le_config = dict()
        le_results = dict()
        
        if sk_le_cols:
            for i in sk_le_cols:
                le = LabelEncoder()
                le.fit(df[i])
                le_config[i] = {"type": "sklearn", "encoder": le}
                le_results[i] = [i for i in zip(range(len(le.classes_)), le.classes_)]
        if mapping_cols:
            for j in mapping_cols:
                le_config[j] = {"type": "custom", "encoder": mapping[j]}
                le_results[j] = [(val, key) for key, val in mapping[j].items()]

        return le_config, le_results

    def label_enc_transform(self, data, le_config):
        """
        This is the transform function for label encoding categorical features.

        :param data: Input data.
        :param le_config: Config file from the fit function.
        :return: DataFrame - Transformed data.
        """
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
    
    def freq_enc_fit(self, df, cat_cols):
        """
        This is the fit functino for frequency encoding of categorical features.

        :param df: Input data.
        :param cat_cols: list of columns to be encoded.
        :return: Dict - Mapping of categories with frequencies.
        """
        freq_enc_map = dict()
        
        for i in cat_cols:
            freq_data = df[i].value_counts(normalize=True)
            freq_enc_map[i] = dict()
            freq_enc_map[i]['freq_map'] = freq_data.to_dict()
            freq_enc_map[i]['freq_mode'] = freq_data[freq_data.index[0]]
            
        return freq_enc_map

    def freq_enc_transform(self, df, freq_config):
        """
        This is the transform function for frequency encoding categorical features.

        :param df: Input data.
        :param freq_config: Mapping file from fit function.
        :return: DataFrame - Transformed data.
        """
        fdf = df.copy(deep=True)
        for key, value in freq_config.items():
            cat_freq_map = value['freq_map']
            fdf[key] = fdf[key].map(cat_freq_map)
            fdf.loc[fdf[key].isna(), key] = value['freq_mode']
        
        return fdf
    
    def ohe_fit(self, df, cols):
        """
        This is the fit functino for one-hot encoding of categorical features.

        :param df: Input data.
        :param cols: list of columns to be encoded.
        :return: Tuple -- (ohe_mod, new_column_names) -- The ohe_mod file contains the configurations to be
        passed to the transform function & the new_column_names dict contains the list of new OHE features names.
        """
        ohe = OneHotEncoder(handle_unknown='ignore')
        df_subset = df[cols].copy(deep=True)
    
        ohe_mod = ohe.fit(df_subset)
    
        new_col_names = dict()
    
        for i in cols:
            unq_vals = list(df[i].unique())
            new_col_names[i] = [i+"_"+str(j) for j in unq_vals]
            new_col_names[i].sort()
    
        return ohe_mod, new_col_names

    def ohe_transform(self, df, ohe_mod, new_col_names, drop_vals=None):
        """
        This is the transform function for one-hot encoding categorical features.

        :param df: Input data.
        :param ohe_mod: OHE model from fit function.
        :param new_col_names: New feature names mapping from fit function.
        :param drop_vals: Optional -- Dict containing the category to be dropped from the transformed features. This mimics dummy encoding behaviour, by dropping a specific value rather than the first value.
        :return: DataFrame - Transformed data.
        """
        ohe_new_cols = list()
        orig_cols = list(new_col_names.keys())

        for col in orig_cols:
            ohe_new_cols.extend(new_col_names[col])
    
        new_df = pd.DataFrame(ohe_mod.transform(df[orig_cols]).toarray(), columns=ohe_new_cols)
    
        # creating the list of transformed features to drop if dummy encoding is required
        if drop_vals is not None:
            cols_to_drop = list()
            for key, val in drop_vals.items():
                cols_to_drop.append(key+"_"+str(val))
    
            new_df.drop(columns=cols_to_drop, inplace=True)
            
        new_df.index = df.index
        # new_df = df.merge(new_df, how='inner', left_index=True, right_index=True)
        new_df = pd.concat([df, new_df], axis=1)
        new_df.drop(columns=orig_cols, inplace=True)
        
        return new_df
    
    ### Other methods to handle missing values
    # Update missing values for categorical features with mode for each data segment
    def get_mode_by_group(self, data, columns, group_by):
        """
        This function will update the values in a categorical field
        with the mode calculated at the group level provided.
        It also returns the overall mode for each column so that it can be used for any unseen
        categories in the test dataset.

        Parameters
        ----------
        data : DataFrame.
        columns : List of categorical columns for which the mode is to be calculated.
        group_by : List of columns at the level at which aggregation needs to be carried out.

        Returns
        -------
        DataFrame: With mode for each group.
        Dictionary: With the overall mode for the column.

        """
        map_df = pd.DataFrame()
        col_mode_dict = dict()
        for col in columns:
            
            col_mode = data[col].value_counts().reset_index().iloc[0,0]
            data[col].fillna(col_mode, inplace=True)
            group_cols = group_by + [col]
            col_mode_dict[col] = col_mode
            
            count_df = data.groupby(by=group_cols).agg({col:'count'})
            count_df.columns = ['count']
            count_df.reset_index(inplace=True)
            max_by_grp = count_df.groupby(by=group_by).agg({'count':'max'})
            max_by_grp.columns = ['max']
            max_by_grp.reset_index(inplace=True)
            count_df = count_df.merge(max_by_grp, how='inner',on=group_by, validate='many_to_one')
            agg_result = count_df.loc[count_df['count']==count_df['max'], group_cols]
            
            # if there are multiple rows with same max value, the choose 1
            agg_result['row_num']= agg_result.groupby(group_by).cumcount()
            inter_df = agg_result.loc[agg_result['row_num']==0, group_cols]
            
            if col==columns[0]:
                map_df = inter_df
            else:
                #final_df = pd.concat([final_df, agg_result[col]], axis=1, ignore_index=False)
                map_df = final_df.merge(inter_df, how='inner', on=group_by, validate='one_to_one')
            
        return map_df, col_mode_dict
    
    # Update missing values for numerical features with median for each data segment
    def get_median_by_group(self, data, columns, group_by):
        """
        This function will update the values in a numerical field with the 
        median calculated at the group level provided.
        It also returns the overall median for each column so that it can be 
        used for any unseen categories in the test dataset.

        Parameters
        ----------
        data : DataFrame.
        columns : List of numerical columns for which the median is to be calculated.
        group_by : List of columns at the level at which aggregation needs to be carried out.

        Returns
        -------
        DataFrame: With median for each group.
        Dictionary: With the overall median for the column.

        """
        col_median_df = data[columns].median().reset_index()
        #col_median_df.columns=['feature','median_val']
        col_median_dict = {col_median_df.iloc[i,0]: col_median_df.iloc[i,1] for i in range(col_median_df.shape[0])}
        
        map_df = data.groupby(by=group_by).agg({i:'median' for i in columns}).reset_index()
        map_df.columns = group_by + columns
        
        return map_df, col_median_dict
    
    def update_median_by_group(self, data, map_df, overall_median_dict, group_by):
        """
        This function uses the output of the get_median_by_group function to update the 
        missing values of numeric features in a dataset with the median value by the 
        group defined.

        Parameters
        ----------
        data : DataFrame
        map_df : Mapping dataframe containing the median values for each broup for the relevant columns.
        overall_median_dict : Dictionary containing the column level median values.
        group_by : List of columns for which median is calculated for each group.

        Returns
        -------
        final_df : DataFrame: The final DataFrame with missing values updated 
                    with the median by group.

        """
        cols_with_missing = list(overall_median_dict.keys())
        final_df = data.merge(map_df, how='inner', on=group_by, validate='many_to_one')
        for i in cols_with_missing:
            final_df[i+'_x'] = np.where(final_df[i+'_x'].isna(),final_df[i+'_y'],final_df[i+'_x'])

        final_df.drop([i+'_y' for i in cols_with_missing], axis=1,inplace=True)
        fnl_cols = {i+'_x':i for i in cols_with_missing}
        final_df.rename(columns=fnl_cols, inplace=True)
        
        return final_df
    
    