# %%
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pyarrow
import fastparquet

# %%
def fill_sic2_with_mode(df):
    """
    This function takes a DataFrame and fills the NaN values in the 'sic2' column with the mode value 
    calculated for each group defined by the 'permno' column.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'permno' and 'sic2' columns.

    Returns:
    - pd.DataFrame: The modified DataFrame with NaN values in 'sic2' filled with the mode value.
    """
    # Step 1: Calculate the mode of 'sic2' for each 'permno' group.
    # If there is no mode (i.e., all values are unique or NaN), return None.
    mode_sic2 = df.groupby('permno')['sic2'].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None).reset_index(name='mode_sic2')
    
    # Step 2: Merge the original DataFrame with the DataFrame containing modes.
    # This allows us to use the mode values to fill NaNs in the original DataFrame.
    df_with_mode = df.merge(mode_sic2[['permno', 'mode_sic2']], on='permno', how='left')
    
    # Step 3: Fill NaN values in the 'sic2' column with the corresponding mode values from the merged DataFrame.
    df_with_mode['sic2'] = df_with_mode['sic2'].fillna(df_with_mode['mode_sic2'])
    
    # Step 4: Drop the temporary 'mode_sic2' column since it's no longer needed.
    df_with_mode.drop(columns=['mode_sic2'], inplace=True)

    df_sic2_with_mode = df_with_mode.dropna(subset=['sic2'])
    
    return df_sic2_with_mode

def fill_na_with_median_by_date(df):
    """
    This function takes a DataFrame and fills the NaN values in all columns except 'sic2', 'DATE', and 'permno'
    with the median values calculated for each group defined by the 'DATE' column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'DATE', 'permno', 'sic2', and other columns.

    Returns:
    - pd.DataFrame: The modified DataFrame with NaN values filled using the median values.
    """
    
    # Step 1: Identify the columns that need to be filled with medians (excluding 'sic2', 'DATE', and 'permno')
    columns_to_fill = [col for col in df.columns if col not in ['sic2', 'DATE', 'permno','RET']]
    
    # Step 2: Group the DataFrame by 'DATE' and calculate the median for the specified columns
    medians_by_date = df.groupby('DATE')[columns_to_fill].transform('median')
    
    # Step 3: Fill NaN values in the specified columns with the corresponding median values
    df_filled = df.copy()
    for col in columns_to_fill:
         df_filled[col] = df_filled[col].fillna(medians_by_date[col])
    
    return df_filled

def fill_na_with_zero(df):
    """
    This function takes a DataFrame and fills the NaN values in all columns except 'sic2', 'DATE', and 'permno'
    with zeros.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'DATE', 'permno', 'sic2', and other columns.

    Returns:
    - pd.DataFrame: The modified DataFrame with NaN values filled with zeros.
    """

    # Identify the columns that need to be filled with zeros (excluding 'sic2', 'DATE', 'permno')
    columns_to_fill = [col for col in df.columns if col not in ['sic2', 'DATE', 'permno','RET']]

    # Fill NaN values in the specified columns with zeros
    df_filled = df.copy()
    df_filled[columns_to_fill] = df_filled[columns_to_fill].fillna(0)

    return df_filled

def one_hot_encode_sic2(df):
    """
    This function takes a DataFrame and performs one-hot encoding on the 'sic2' column.
    It returns a new DataFrame with one-hot encoded columns for each unique value in 'sic2'.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the 'sic2' column.

    Returns:
    - pd.DataFrame: The modified DataFrame with one-hot encoded columns for 'sic2'.
    """
    
    # Perform one-hot encoding on the 'sic2' column
    sic2_encoded = pd.get_dummies(df['sic2'], prefix='sic2').astype(int)
    
    # Concatenate the original DataFrame with the one-hot encoded columns
    df_encoded = pd.concat([df.drop('sic2', axis=1), sic2_encoded], axis=1)
    
    return df_encoded




def rank_to_minus_one_to_one(df):
    
    for col in df.columns:
        if col != "DATE" and col !="permno" and col!= "sic2" and col!="RET":  # remain DATE and permno
            val = df[col].values

            # Rank the values, with 1 being the lowest rank
            ranks = rankdata(val, method='average')  # method='average' handles ties by assigning average rank

            # Map ranks to [0, 1] interval first
            normalized_ranks = (ranks - 1) / (len(val) - 1)

            # Map normalized ranks to [-1, 1]
            df[col] = 2 * normalized_ranks - 1
        
    return df

# cross-sectionally rank all stock characteristics period-by-period 
# and map these ranks into the [-1,1] interval
def data_mapping(df):

    grouped = df.groupby(["DATE"]).apply(rank_to_minus_one_to_one)

    return grouped


def cal_macro(data, macro_columns,start_date= '1957-03-01', end_date = '2023-12-31'):
    data = data.loc[start_date:end_date]
    data['Index'] = data['Index'].str.replace(',','').astype('float64')
    data['macro_dp'] = np.log(data['D12']) - np.log(data['Index'])
    data['macro_ep'] = np.log(data['E12']) - np.log(data['Index'])
    data['macro_tms'] = data['lty'] - data['tbl']
    data['macro_dfy'] = data['BAA'] - data['AAA']
    data.rename({'b/m':'macro_bm'},axis=1,inplace=True)
    data.rename({'ntis':'macro_ntis'},axis=1,inplace=True)
    data.rename({'tbl':'macro_tbl'},axis=1,inplace=True)
    data.rename({'svar':'macro_svar'},axis=1,inplace=True)
    data = data[macro_columns]
    return data


def generate_covariates(data,macro, cols_char, cols_macro):
    
    m = data.merge(macro, left_on='DATE', right_index=True, how='left')
    m['RET'] = m['RET'] - m['macro_tbl'] 
    for i in cols_char:  
        for j in cols_macro:
            new_column_name  = i + '_' + j
            m[new_column_name] = m[i] * m[j]

    m.drop(columns=cols_macro, inplace=True)
    #m.rename({'RET':'excess_ret'},axis=1,inplace=True)
    return m 


def data_preprocessing(data, macro_data, macro_columns, start_date= '1957-01-01', end_date = '2016-12-31'):
    data['DATE'] = pd.to_datetime(data['DATE'],format='%Y%m%d')
    data['DATE'] = data['DATE'] + pd.offsets.MonthEnd(0)
    data.set_index('DATE', inplace=True)
    data = data.loc[start_date: end_date]
    data = data.dropna(subset=['RET'])
    data.reset_index(inplace=True)
    macro_data['yyyymm'] = pd.to_datetime(macro_data['yyyymm'],format='%Y%m')+pd.offsets.MonthEnd(0)
    macro_data.rename(columns={'yyyymm': 'DATE'}, inplace=True)
    macro_data.set_index('DATE', inplace=True)
    macro_data = macro_data.loc[start_date: end_date]
    data = fill_sic2_with_mode(data)
    data = fill_na_with_median_by_date(data)
    data = fill_na_with_zero(data)
    data = data_mapping(data)
    data =  data.reset_index(drop=True)
    data.set_index('DATE', inplace=True)
    macro_data = cal_macro(macro_data,macro_columns,start_date=start_date, end_date =end_date)
    data = data.merge(macro_data, left_on='DATE', right_index=True, how='left')
    data['RET'] = data['RET'] - data['macro_tbl'] # 处理label
    data = generate_covariates(data,macro_data, data.columns.drop(['permno','sic2','RET']), macro_data.columns)
    data = one_hot_encode_sic2(data)
    return data

# %%
data = pd.read_parquet('/home/cheam/fintech_pro2/datashare/GKX_20201231.parquet').drop(columns=['SHROUT','mve0','prc'])

macro_columns = ['macro_dp','macro_ep','macro_bm','macro_ntis','macro_tbl','macro_tms','macro_dfy','macro_svar']
macro_data = pd.read_csv('/home/cheam/fintech_pro2/datashare/PredictorData2023.csv')


# %% [markdown]
# # top1000

# %%
# top1000
data_top1000 = data.groupby('DATE').apply(lambda x: x.nlargest(20, 'mvel1')).reset_index(drop=True)
data_top1000

# %%
data_top1000 = data_preprocessing(data_top1000, macro_data, macro_columns, start_date= '1957-01-01', end_date = '2016-12-31')
data_top1000

# %%
data_top1000.reset_index(inplace=True)
data_top1000

# %%
data_top1000.to_parquet('/home/cheam/fintech_pro2/datashare/GKX_top20.parquet', index=False)

# %% [markdown]
# # bottom1000

# %%
data_bottom1000 = data.sort_values('mvel1', ascending = False).groupby('DATE').tail(20)

# %%
data_bottom1000 = data_bottom1000.reset_index(drop = True)
data_bottom1000

# %%
data_bottom1000.sort_values(['DATE','permno'], inplace=True)
data_bottom1000

# %%
data_bottom1000=data_preprocessing(data_bottom1000, macro_data, macro_columns, start_date= '1957-01-01', end_date = '2016-12-31')
data_bottom1000

# %%
data_bottom1000.reset_index(inplace=True)
data_bottom1000

# %%
data_bottom1000

# %%
data_bottom1000.to_parquet('/home/cheam/fintech_pro2/datashare/GKX_bottom20.parquet', index=False)

# %% [markdown]
# # all data

# %%
data = data_preprocessing(data, macro_data, macro_columns)
data

# %%
data.reset_index(inplace=True)
data

# %%
data.reset_index(inplace=True)
data

# %%
data.to_parquet('/home/cheam/fintech_pro2/datashare/GKX_fullset.parquet', index=False)


