# %% [markdown] {"execution":{"iopub.status.busy":"2024-09-26T19:29:03.316565Z","iopub.execute_input":"2024-09-26T19:29:03.317047Z","iopub.status.idle":"2024-09-26T19:29:19.759696Z","shell.execute_reply.started":"2024-09-26T19:29:03.316992Z","shell.execute_reply":"2024-09-26T19:29:19.757928Z"}}
# # pip install dask distributed

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:32.913007Z","iopub.execute_input":"2024-10-04T09:38:32.913523Z","iopub.status.idle":"2024-10-04T09:38:49.041805Z","shell.execute_reply.started":"2024-10-04T09:38:32.913475Z","shell.execute_reply":"2024-10-04T09:38:49.040010Z"}}
pip install dask distributed

# %% [markdown]
# ## Load libraries

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.044564Z","iopub.execute_input":"2024-10-04T09:38:49.045035Z","iopub.status.idle":"2024-10-04T09:38:49.063681Z","shell.execute_reply.started":"2024-10-04T09:38:49.044987Z","shell.execute_reply":"2024-10-04T09:38:49.062206Z"}}
import numpy as np 
import pandas as pd 
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
print(os.listdir("../input"))
from dask.distributed import Client
import plotly.graph_objs as go
import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.graph_objs as go
import plotly.offline as py
import gc

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.065391Z","iopub.execute_input":"2024-10-04T09:38:49.065928Z","iopub.status.idle":"2024-10-04T09:38:49.077966Z","shell.execute_reply.started":"2024-10-04T09:38:49.065871Z","shell.execute_reply":"2024-10-04T09:38:49.076420Z"}}
gc.enable()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.081895Z","iopub.execute_input":"2024-10-04T09:38:49.082601Z","iopub.status.idle":"2024-10-04T09:38:49.102849Z","shell.execute_reply.started":"2024-10-04T09:38:49.082541Z","shell.execute_reply":"2024-10-04T09:38:49.101558Z"}}


dtype = {
    'IsBeta':'int8', 
    'IsSxsPassiveMode':'int8',
    'HasTpm':'int8',
    'AutoSampleOptIn':'int8',
    'Census_HasOpticalDiskDrive':'int8',
    'Census_IsPortableOperatingSystem':'int8',
    'Census_IsSecureBootEnabled':'int8',
    'Census_IsTouchEnabled':'int8',
    'Census_IsPenCapable':'int8',
    'HasDetections':'int8',
    'CountryIdentifier':'int16',
    'LocaleEnglishNameIdentifier':'int16',
    'OsBuild':'int16',
    'OsSuite':'int16',
    'Census_OSBuildNumber':'int16',
    'Census_OSUILocaleIdentifier':'int16',
    'Census_OSBuildRevision':'int32',
    'RtpStateBitfield':'float64',
    'DefaultBrowsersIdentifier':'float64',
    'AVProductStatesIdentifier':'float64',
    'AVProductsInstalled':'float64',
    'AVProductsEnabled':'float64',
    'CityIdentifier':'float64',
    'OrganizationIdentifier':'float64',
    'GeoNameIdentifier':'float64',
    'IsProtected':'float64',
    'SMode':'float64',
    'IeVerIdentifier':'float64',
    'Firewall':'float64',
    'UacLuaenable':'float64',
    'Census_OEMNameIdentifier':'float64',
    'Census_OEMModelIdentifier':'float64',
    'Census_ProcessorCoreCount':'float64',
    'Census_ProcessorManufacturerIdentifier':'float64',
    'Census_ProcessorModelIdentifier':'float64',
    'Census_PrimaryDiskTotalCapacity':'float64',
    'Census_SystemVolumeTotalCapacity':'float64',
    'Census_TotalPhysicalRAM':'float64',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches':'float64',
    'Census_InternalPrimaryDisplayResolutionHorizontal':'float64',
    'Census_InternalPrimaryDisplayResolutionVertical':'float64',
    'Census_InternalBatteryNumberOfCharges':'float64',
    'Census_OSInstallLanguageIdentifier':'float64',
    'Census_IsFlightingInternal':'float64',
    'Census_IsFlightsDisabled':'float64',
    'Census_ThresholdOptIn':'float64',
    'Census_FirmwareManufacturerIdentifier':'float64',
    'Census_FirmwareVersionIdentifier':'float64',
    'Census_IsWIMBootEnabled':'float64',
    'Census_IsVirtualDevice':'float64',
    'Census_IsAlwaysOnAlwaysConnectedCapable':'float64',
    'Wdft_IsGamer':'float64',
    'Wdft_RegionIdentifier':'float64',
    'MachineIdentifier':'object',
    'ProductName':'object',
    'EngineVersion':'object',
    'AppVersion':'object',
    'AvSigVersion':'object',
    'Platform':'object',
    'Processor':'object',
    'OsVer':'object',
    'OsPlatformSubRelease':'object',
    'OsBuildLab':'object',
    'SkuEdition':'object',
    'PuaMode':'object',
    'SmartScreen':'object',
    'Census_MDC2FormFactor':'object',
    'Census_DeviceFamily':'object',
    'Census_ProcessorClass':'object',
    'Census_PrimaryDiskTypeName':'object',
    'Census_ChassisTypeName':'object',
    'Census_PowerPlatformRoleName':'object',
    'Census_InternalBatteryType':'object',
    'Census_OSVersion':'object',
    'Census_OSArchitecture':'object',
    'Census_OSBranch':'object',
    'Census_OSEdition':'object',
    'Census_OSSkuName':'object',
    'Census_OSInstallTypeName':'object',
    'Census_OSWUAutoUpdateOptionsName':'object',
    'Census_GenuineStateName':'object',
    'Census_ActivationChannel':'object',
    'Census_FlightRing':'object'
}

data = dd.read_csv('/kaggle/input/microsoft-malware-prediction/train.csv', dtype=dtype)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.104347Z","iopub.execute_input":"2024-10-04T09:38:49.104887Z","iopub.status.idle":"2024-10-04T09:38:49.122032Z","shell.execute_reply.started":"2024-10-04T09:38:49.104841Z","shell.execute_reply":"2024-10-04T09:38:49.120705Z"}}
#  drop cols has more than 60% nulls or has dublicated more than 90%
def drop_cols(train_df):  
    cols_to_delete = [
        'ProductName', 'IsBeta', 'IsSxsPassiveMode', 'HasTpm', 
        'AutoSampleOptIn', 'PuaMode', 'UacLuaenable', 'MachineIdentifier',
        'Census_DeviceFamily', 'Census_ProcessorClass','Firewall', 
        'Census_InternalBatteryType', 'Census_IsFlightingInternal', 
        'Census_IsFlightsDisabled', 'Census_FlightRing', 'IsProtected',
        'Census_ThresholdOptIn', 'Census_IsWIMBootEnabled', 'OsVer',
        'Census_IsVirtualDevice', 'Census_IsPenCapable', 'Platform',
        'Census_IsAlwaysOnAlwaysConnectedCapable','RtpStateBitfield','DefaultBrowsersIdentifier'
         ]

    train_df = train_df.drop(columns=cols_to_delete)
    return train_df
data=drop_cols(data)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.124036Z","iopub.execute_input":"2024-10-04T09:38:49.124880Z","iopub.status.idle":"2024-10-04T09:38:49.547093Z","shell.execute_reply.started":"2024-10-04T09:38:49.124832Z","shell.execute_reply":"2024-10-04T09:38:49.545812Z"}}
# get randome sample of data
def sample(data,count): 
    total_rows = data.shape[0]
    frac = count / total_rows
    train_df = data.sample(frac=frac, replace=False)
    return train_df


train_df =sample(data,2000000)

del data
gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.548959Z","iopub.execute_input":"2024-10-04T09:38:49.549490Z","iopub.status.idle":"2024-10-04T09:38:49.618862Z","shell.execute_reply.started":"2024-10-04T09:38:49.549408Z","shell.execute_reply":"2024-10-04T09:38:49.613457Z"}}

import dask.dataframe as dd

int_cols = train_df.select_dtypes(include='int64').columns
train_df[int_cols] = train_df[int_cols].astype('int32')

float_cols = train_df.select_dtypes(include='float64').columns
train_df[float_cols] = train_df[float_cols].astype('float32')

print(train_df.dtypes)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:40:18.863987Z","iopub.execute_input":"2024-10-04T09:40:18.864569Z","iopub.status.idle":"2024-10-04T09:40:18.873922Z","shell.execute_reply.started":"2024-10-04T09:40:18.864520Z","shell.execute_reply":"2024-10-04T09:40:18.872710Z"}}
def reduce_mem_usage(df):

    for col in df.select_dtypes(include=[np.number]).columns:
        col_type = df[col].dtype
        if np.issubdtype(col_type, np.integer):
            # Determine the smallest integer data type that can accommodate the data
            min_value = df[col].min()
            max_value = df[col].max()
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if np.iinfo(dtype).min <= min_value and np.iinfo(dtype).max >= max_value:
                    df[col] = df[col].astype(dtype)
                    break

    return df

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:40:21.816948Z","iopub.execute_input":"2024-10-04T09:40:21.817470Z","iopub.status.idle":"2024-10-04T09:40:21.889620Z","shell.execute_reply.started":"2024-10-04T09:40:21.817398Z","shell.execute_reply":"2024-10-04T09:40:21.887781Z"}}
start_mem = train_df.memory_usage().sum() / 1024**3
print(f"Memory usage of dataframe is {start_mem:.2f} GB")
train = reduce_mem_usage(train_df)
end_mem = train.memory_usage().sum() / 1024**3
print(f"Memory usage after optimization is: {end_mem:.2f} GB")
print(f"Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:40:26.459402Z","iopub.execute_input":"2024-10-04T09:40:26.460077Z","iopub.status.idle":"2024-10-04T09:40:26.597715Z","shell.execute_reply.started":"2024-10-04T09:40:26.460018Z","shell.execute_reply":"2024-10-04T09:40:26.596044Z"}}

def num_cat_split(data):
    num_cols = list(set(data.describe().columns.to_list()))
    cat_cols= list(set(data.columns.to_list()) - set(num_cols))
    return num_cols,cat_cols

num,cat=num_cat_split(train_df)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:40:30.602359Z","iopub.execute_input":"2024-10-04T09:40:30.603370Z","iopub.status.idle":"2024-10-04T09:42:24.868021Z","shell.execute_reply.started":"2024-10-04T09:40:30.603314Z","shell.execute_reply":"2024-10-04T09:42:24.847500Z"}}
# null_count_persent_datatype
def dataframe_summary(df):
    summary = []
    for col in df.columns:
        col_name = col
        null_count = df[col].isnull().sum()
        null_percent = (null_count / len(df)) * 100
        data_type = df[col].dtype
        
        summary.append([col_name, null_count, null_percent, data_type])
    
    summary_df = pd.DataFrame(summary, columns=['Column Name', 'Null Count', 'Null Percentage', 'Data Type'])
    
    return summary_df

nulls=dataframe_summary(train_df)
del nulls
gc.collect

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.632104Z","iopub.status.idle":"2024-10-04T09:38:49.632654Z","shell.execute_reply.started":"2024-10-04T09:38:49.632363Z","shell.execute_reply":"2024-10-04T09:38:49.632388Z"}}
train_df.head(10)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.634497Z","iopub.status.idle":"2024-10-04T09:38:49.635015Z","shell.execute_reply.started":"2024-10-04T09:38:49.634770Z","shell.execute_reply":"2024-10-04T09:38:49.634795Z"}}
train_df.info()

# %% [markdown]
# # funcion ip_errors selet the most founded catigores and replace others with word "other"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.636251Z","iopub.status.idle":"2024-10-04T09:38:49.636773Z","shell.execute_reply.started":"2024-10-04T09:38:49.636526Z","shell.execute_reply":"2024-10-04T09:38:49.636551Z"}}
def ip_errors(df):
     
    SkuEdition_accepted=['Home','Pro','Invalid']
    df['SkuEdition'] = df['SkuEdition'].where(df['SkuEdition'].isin(SkuEdition_accepted), 'other')
    
    Census_MDC2FormFactor_accepted=['Notebook','Desktop','Convertible','Detachable','AllInOne','PCOther']
    df['Census_MDC2FormFactor'] = df['Census_MDC2FormFactor'].where(df['Census_MDC2FormFactor'].isin(Census_MDC2FormFactor_accepted), 'other')
    
    Census_OSEdition_accepted = ["Core", "Professional", "CoreSingleLanguage", "CoreCountrySpecific"]
    df['Census_OSEdition'] = df['Census_OSEdition'].where(df['Census_OSEdition'].isin(Census_OSEdition_accepted), 'other')
     
    Census_OSBranch_accepted=['rs4_release','rs3_release','rs3_release_svc_escrow','rs2_release','th2_release','rs1_release','th2_release_sec','th1_st1','th1']
    df['Census_OSBranch'] = df['Census_OSBranch'].where(df['Census_OSBranch'].isin(Census_OSBranch_accepted), 'other')
    
    Census_ChassisTypeName_accepted = ["Notebook", "Desktop", "Laptop", "Portable", "Allinone"]
    df['Census_ChassisTypeName'] = df['Census_ChassisTypeName'].where(df['Census_ChassisTypeName'].isin(Census_ChassisTypeName_accepted), 'other')
    
    Census_OSSkuName_accepted = ["CORE", "PROFESSIONAL", "CORE_SINGLELANGUAGE", "CORE_COUNTRYSPECIFIC"]
    train_df['Census_OSSkuName'] = train_df['Census_OSSkuName'].where(train_df['Census_OSSkuName'].isin(Census_OSSkuName_accepted), 'other')
    
    return df

train_df=ip_errors(train_df)


# %% [markdown]
# # visualization

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.639250Z","iopub.status.idle":"2024-10-04T09:38:49.639935Z","shell.execute_reply.started":"2024-10-04T09:38:49.639602Z","shell.execute_reply":"2024-10-04T09:38:49.639636Z"}}
# is scketched bar chare in normale barplot added on it target data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_impact(data, target, n=5):
    """
    Visualize the percentage of target (0 and 1) for the top `n` most frequent values in each column,
    with `target == 1` in red and `target == 0` in green, using a stacked bar chart.
    
    Parameters:
    data (pd.DataFrame): The input dataframe.
    target (str): The target column name.
    n (int): The number of most frequent values to consider for each column.
    """
    columns = data.columns.drop(target)
    
    for col in columns:
        # Get the top n most frequent values in the column
        top_n_values = data[col].value_counts().nlargest(n).index
        
        # Filter the data for the top n values
        filtered_data = data[data[col].isin(top_n_values)]
        
        # Create a pivot table to calculate the percentage of target values (0 and 1)
        pivot_table = pd.crosstab(filtered_data[col], filtered_data[target], normalize='index') * 100
        
        # Plot the stacked bar chart
        pivot_table.plot(kind='bar', stacked=True, color=['blue', 'orange'], figsize=(10, 6))
        
        # Add annotations for the percentage
        for i in range(len(pivot_table)):
            for j, val in enumerate(pivot_table.iloc[i]):
                plt.text(i, pivot_table.iloc[i, :j+1].sum() - (val / 2), f'{val:.1f}%', 
                         ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        
        # Update the title and labels
        plt.title(f'Target Impact on {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel(f'Percentage of {target}', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend([f'{target} = 0', f'{target} = 1'], title=target)
        
        plt.tight_layout()
        plt.show()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.641532Z","iopub.status.idle":"2024-10-04T09:38:49.642198Z","shell.execute_reply.started":"2024-10-04T09:38:49.641871Z","shell.execute_reply":"2024-10-04T09:38:49.641903Z"}}
cat.append('HasDetections')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.643917Z","iopub.status.idle":"2024-10-04T09:38:49.644565Z","shell.execute_reply.started":"2024-10-04T09:38:49.644223Z","shell.execute_reply":"2024-10-04T09:38:49.644254Z"}}
plot_feature_impact(train_df[cat],'HasDetections',8)

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.646665Z","iopub.status.idle":"2024-10-04T09:38:49.647313Z","shell.execute_reply.started":"2024-10-04T09:38:49.646990Z","shell.execute_reply":"2024-10-04T09:38:49.647022Z"}}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_num(data, target, n=5):
    """
    Visualize the percentage of target (0 and 1) for the top `n` most frequent values in each column,
    with `target == 1` in red and `target == 0` in green, using a stacked bar chart.
    
    Parameters:
    data (pd.DataFrame): The input dataframe.
    target (str): The target column name.
    n (int): The number of most frequent values to consider for each column.
    """
    columns = data.columns.drop(target)
    
    for col in columns:
        if data[col].dtype in ['int64', 'float64']:  # Check if the column is numerical
            # Create bins for the numerical column
            data[f'{col}_binned'] = pd.cut(data[col], bins=n, labels=[f'Bin {i+1}' for i in range(n)])
            col_to_plot = f'{col}_binned'
        else:
            col_to_plot = col

        # Get the top n most frequent values in the column or the binned column
        top_n_values = data[col_to_plot].value_counts().nlargest(n).index
        
        # Filter the data for the top n values
        filtered_data = data[data[col_to_plot].isin(top_n_values)]
        
        # Create a pivot table to calculate the percentage of target values (0 and 1)
        pivot_table = pd.crosstab(filtered_data[col_to_plot], filtered_data[target], normalize='index') * 100
        
        # Plot the stacked bar chart
        pivot_table.plot(kind='bar', stacked=True, color=['blue', 'orange'], figsize=(10, 6))
        
        # Add annotations for the percentage
        for i in range(len(pivot_table)):
            for j, val in enumerate(pivot_table.iloc[i]):
                plt.text(i, pivot_table.iloc[i, :j+1].sum() - (val / 2), f'{val:.1f}%', 
                         ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        
        # Update the title and labels
        plt.title(f'Target Impact on {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel(f'Percentage of {target}', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend([f'{target} = 0', f'{target} = 1'], title=target)
        
        plt.tight_layout()
        plt.show()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.650078Z","iopub.status.idle":"2024-10-04T09:38:49.650752Z","shell.execute_reply.started":"2024-10-04T09:38:49.650393Z","shell.execute_reply":"2024-10-04T09:38:49.650426Z"}}
plot_feature_num(train_df[num],'HasDetections',10)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.652180Z","iopub.status.idle":"2024-10-04T09:38:49.652869Z","shell.execute_reply.started":"2024-10-04T09:38:49.652498Z","shell.execute_reply":"2024-10-04T09:38:49.652533Z"}}
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_target_relationships(data, target, n=10):
    """
    Function to visualize relationships between a target variable and the top 'n' highest values 
    of each column in a DataFrame, with a line indicating the highest value.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    target (str): The name of the target column.
    n (int): The number of highest values to plot for numerical columns. Default is 10.

    Returns:
    None: Displays visualizations for each column.
    """
    # Define the target variable
    target_data = data[target]

    # Loop through each column in the DataFrame (except the target column)
    for col in data.columns:
        if col == target:
            continue  # Skip the target column
        
        plt.figure(figsize=(10, 6))
        
        # Check if the column is numerical
        if data[col].dtype in ['int64', 'float64']:
            # Get the top 'n' highest values for the column
            top_n_values = data[[col, target]].nlargest(n, col)

            # Plot the distribution of the top 'n' values vs the target
            sns.histplot(data=top_n_values, x=col, hue=target, kde=True, element='step')
            
            # Find the highest value in the top 'n' values
            max_value = top_n_values[col].max()
            
            # Add a vertical line to indicate the highest value
            plt.axvline(max_value, color='red', linestyle='--', label=f'Highest {col}: {max_value}')
            
            plt.title(f'Top {n} {col} values with respect to {target}')
            plt.legend()
        
        # If the column is categorical
        else:
            # Plot the count for the top 'n' categories if applicable (limit to unique categories)
            sns.countplot(data=data, x=col, hue=target, order=data[col].value_counts().index[:n])
            plt.title(f'Count of top {n} {col} categories with respect to {target}')
        
        plt.tight_layout()
        plt.show()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.654678Z","iopub.status.idle":"2024-10-04T09:38:49.655286Z","shell.execute_reply.started":"2024-10-04T09:38:49.654970Z","shell.execute_reply":"2024-10-04T09:38:49.655000Z"}}
visualize_target_relationships(train_df[cat],'HasDetections')

# %% [markdown]
# # one_hot_Encoding

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.656533Z","iopub.status.idle":"2024-10-04T09:38:49.657139Z","shell.execute_reply.started":"2024-10-04T09:38:49.656835Z","shell.execute_reply":"2024-10-04T09:38:49.656866Z"}}
import dask.dataframe as dd
import dask

# One-hot encoding function
def one_hot_encoding(train_df):
    categorical_columns = [
        'OsPlatformSubRelease', 'SkuEdition', 'Census_MDC2FormFactor', 
        'Census_PrimaryDiskTypeName', 'Census_OSArchitecture', 'Census_OSBranch', 
        'Census_OSEdition', 'Census_OSSkuName', 'Census_OSInstallTypeName', 
        'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName', 
        'Census_ActivationChannel','Processor','Census_ChassisTypeName','Census_PowerPlatformRoleName'
    ]
    
    # Use Dask's get_dummies function
    train_df = dd.get_dummies(train_df, columns=categorical_columns)
    return train_df

# Apply one-hot encoding
train_df = one_hot_encoding(train_df)



# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.658422Z","iopub.status.idle":"2024-10-04T09:38:49.659052Z","shell.execute_reply.started":"2024-10-04T09:38:49.658744Z","shell.execute_reply":"2024-10-04T09:38:49.658776Z"}}
#  from bool to int
train_df[train_df.select_dtypes(include=['bool']).columns] = train_df.select_dtypes(include=['bool']).astype(int)

# %% [markdown]
# # Lable_Encoding

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.661156Z","iopub.status.idle":"2024-10-04T09:38:49.661789Z","shell.execute_reply.started":"2024-10-04T09:38:49.661463Z","shell.execute_reply":"2024-10-04T09:38:49.661498Z"}}
def lable_encoding(train_df):   
    columns_to_encode = [
       'OsBuildLab', 'SmartScreen', 'AppVersion', 'AvSigVersion', 
       'Census_OSVersion',"Census_InternalBatteryNumberOfCharges","EngineVersion"]

# Initialize a LabelEncoder
    label_encoders = {col: LabelEncoder() for col in columns_to_encode}

# Apply label encoding to each specified column
    for col in columns_to_encode:
    # Fit the LabelEncoder and transform the column
        train_df[col] = label_encoders[col].fit_transform(train_df[col])
    return train_df

train_df=lable_encoding(train_df)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.664093Z","iopub.status.idle":"2024-10-04T09:38:49.664746Z","shell.execute_reply.started":"2024-10-04T09:38:49.664393Z","shell.execute_reply":"2024-10-04T09:38:49.664427Z"}}
# pd.set_option('display.max_row',100)
# pd.set_option('display.max_column',None)
# train_df.head(10)

# %% [markdown]
# 

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.666304Z","iopub.status.idle":"2024-10-04T09:38:49.666988Z","shell.execute_reply.started":"2024-10-04T09:38:49.666673Z","shell.execute_reply":"2024-10-04T09:38:49.666707Z"}}
train_df.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.668561Z","iopub.status.idle":"2024-10-04T09:38:49.669198Z","shell.execute_reply.started":"2024-10-04T09:38:49.668880Z","shell.execute_reply":"2024-10-04T09:38:49.668922Z"}}
# funciont take df and retrun cols has data 0 or 1
def get_binary_columns(df):
    binary_cols = []
    for col in df.columns:
        unique_values = df[col].dropna().unique()
        if set(unique_values).issubset({0, 1}):
            binary_cols.append(col)
    return binary_cols

binary_columns = get_binary_columns(train_df)

# print("Columns with only 0 or 1:", binary_columns)
len(binary_columns)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.671248Z","iopub.status.idle":"2024-10-04T09:38:49.671888Z","shell.execute_reply.started":"2024-10-04T09:38:49.671559Z","shell.execute_reply":"2024-10-04T09:38:49.671590Z"}}
train_df.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.673307Z","iopub.status.idle":"2024-10-04T09:38:49.673932Z","shell.execute_reply.started":"2024-10-04T09:38:49.673611Z","shell.execute_reply":"2024-10-04T09:38:49.673651Z"}}
# conver from pandas to dask 
train_df = dd.from_pandas(train_df,2)

# %% [markdown]
# # full Nulls in nomircal cols has data (0,1) beased on persentage of 0 and 1

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.675317Z","iopub.status.idle":"2024-10-04T09:38:49.675940Z","shell.execute_reply.started":"2024-10-04T09:38:49.675630Z","shell.execute_reply":"2024-10-04T09:38:49.675663Z"}}
import dask.dataframe as dd
import numpy as np
import pandas as pd

def fill_nulls_based_on_existing_values_dask(df, *cols):
    
    
    for col in cols:
        if col in df.columns:
            # Count 0s and 1s in each partition
            value_counts = df[col].value_counts().compute()
            count_0 = value_counts.get(0, 0)
            count_1 = value_counts.get(1, 0)
            total_filled = count_0 + count_1

            if total_filled == 0:
                # If there are no filled rows, fill with either 0 or 1
                df[col] = df[col].fillna(0)  # or fillna(1)
                continue

            # Calculate the percentages
            percentage_0 = count_0 / total_filled
            percentage_1 = count_1 / total_filled

            # Define a function to fill nulls based on calculated probabilities
            def fill_values(partition):
                num_nulls = partition[col].isnull().sum()
                if num_nulls > 0:
                    fill_values = np.random.choice(
                        [0, 1],
                        size=num_nulls,
                        p=[percentage_0, percentage_1]
                    )
                    partition.loc[partition[col].isnull(), col] = fill_values
                return partition

            # Use map_partitions to apply the filling function to each partition
            df = df.map_partitions(fill_values)

    return df

# Assuming you have train_df and binary_columns defined
ddf_filled = fill_nulls_based_on_existing_values_dask(train_df, *binary_columns)

# Trigger computation
final_result = ddf_filled

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.678783Z","iopub.status.idle":"2024-10-04T09:38:49.679405Z","shell.execute_reply.started":"2024-10-04T09:38:49.679092Z","shell.execute_reply":"2024-10-04T09:38:49.679124Z"}}
train_df=final_result
train_df
del final_result,ddf_filled
gc.collect

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.680827Z","iopub.status.idle":"2024-10-04T09:38:49.681454Z","shell.execute_reply.started":"2024-10-04T09:38:49.681124Z","shell.execute_reply":"2024-10-04T09:38:49.681157Z"}}
# show Nulls 
# pd.set_option('display.max_row',None)
# nulls=dataframe_summary(train_df)
# print(nulls)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.683095Z","iopub.status.idle":"2024-10-04T09:38:49.683876Z","shell.execute_reply.started":"2024-10-04T09:38:49.683389Z","shell.execute_reply":"2024-10-04T09:38:49.683421Z"}}
# cals that has num data not binary(0,1)
cals= list(set(train_df.columns.to_list()) - set(binary_columns))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.686500Z","iopub.status.idle":"2024-10-04T09:38:49.687204Z","shell.execute_reply.started":"2024-10-04T09:38:49.686848Z","shell.execute_reply":"2024-10-04T09:38:49.686883Z"}}
cals.append("SMode")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.689046Z","iopub.status.idle":"2024-10-04T09:38:49.689721Z","shell.execute_reply.started":"2024-10-04T09:38:49.689362Z","shell.execute_reply":"2024-10-04T09:38:49.689394Z"}}
cals

# %% [markdown]
# # full normal nomerical nulls

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.692291Z","iopub.status.idle":"2024-10-04T09:38:49.692962Z","shell.execute_reply.started":"2024-10-04T09:38:49.692636Z","shell.execute_reply":"2024-10-04T09:38:49.692670Z"}}
import dask.dataframe as dd
import numpy as np

def fill_null_with_median(data: dd.DataFrame, num_cols: list) -> dd.DataFrame:
    """
    Fills null values in the specified columns of the Dask DataFrame with the median of each column.

    Parameters:
        data (dd.DataFrame): The input Dask DataFrame.
        num_cols (list): The list of columns to process.

    Returns:
        dd.DataFrame: The Dask DataFrame with null values filled with median.
    """
    # Ensure only valid columns are processed
    num_cols = [col for col in num_cols if col in data.columns]

    # Fill null values with the median for the specified columns
    for col in num_cols:
        if data[col].dtype in [np.float64, np.int64]:  # Check if the column is numeric
            median_value = data[col].median()  # Compute the median
            data[col] = data[col].fillna(median_value)

    return data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.694301Z","iopub.status.idle":"2024-10-04T09:38:49.694975Z","shell.execute_reply.started":"2024-10-04T09:38:49.694647Z","shell.execute_reply":"2024-10-04T09:38:49.694683Z"}}
train_df=fill_null_with_median(train_df,cals)

# %% [markdown]
# # convert data from int64 to int32 and from float64 to float32

# %% [markdown]
# 

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.696811Z","iopub.status.idle":"2024-10-04T09:38:49.697466Z","shell.execute_reply.started":"2024-10-04T09:38:49.697122Z","shell.execute_reply":"2024-10-04T09:38:49.697154Z"}}
pd.set_option('display.max_row',None)
nulls=dataframe_summary(train_df)
nulls

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.699079Z","iopub.status.idle":"2024-10-04T09:38:49.699755Z","shell.execute_reply.started":"2024-10-04T09:38:49.699390Z","shell.execute_reply":"2024-10-04T09:38:49.699423Z"}}
train_df.head(20)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.701994Z","iopub.status.idle":"2024-10-04T09:38:49.702509Z","shell.execute_reply.started":"2024-10-04T09:38:49.702243Z","shell.execute_reply":"2024-10-04T09:38:49.702269Z"}}
train_df.info()

# %% [markdown]
# # handleing outliers use IQR

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.703736Z","iopub.status.idle":"2024-10-04T09:38:49.704172Z","shell.execute_reply.started":"2024-10-04T09:38:49.703960Z","shell.execute_reply":"2024-10-04T09:38:49.703981Z"}}
columns=train_df.columns
def handle_outliers(df, columns):
    for column in columns:
        if df[column].dtype in [int, float]:  
        
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Determine outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df

train_df = handle_outliers(train_df,cals)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.705566Z","iopub.status.idle":"2024-10-04T09:38:49.706071Z","shell.execute_reply.started":"2024-10-04T09:38:49.705844Z","shell.execute_reply":"2024-10-04T09:38:49.705868Z"}}
train_df.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.707938Z","iopub.status.idle":"2024-10-04T09:38:49.708397Z","shell.execute_reply.started":"2024-10-04T09:38:49.708162Z","shell.execute_reply":"2024-10-04T09:38:49.708185Z"}}
train_df.info()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-04T09:38:49.714942Z","iopub.status.idle":"2024-10-04T09:38:49.715371Z","shell.execute_reply.started":"2024-10-04T09:38:49.715159Z","shell.execute_reply":"2024-10-04T09:38:49.715180Z"}}
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib 

def xgboost_with_params(df, target_column, learning_rate, balance_weight=True):
    # Split the data into features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance by calculating scale_pos_weight if needed
    if balance_weight:
        classes = y_train.value_counts()
        majority_class = classes.max()
        minority_class = classes.min()
        scale_pos_weight = majority_class / minority_class
    else:
        scale_pos_weight = 1  # Default, no imbalance handling

    # Define the XGBoost model with provided learning rate and scale_pos_weight
    xgb = XGBClassifier(learning_rate=learning_rate, scale_pos_weight=scale_pos_weight)

    # Train the model
    xgb.fit(X_train, y_train)

    # Predict on the test data
    y_pred = xgb.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Print confusion matrix and classification report
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    
    predictions = xgb.predict(X)

    # Evaluate the model on the test data
    y_pred = xgb.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return xgb


# %% [code] {"execution":{"iopub.status.busy":"2024-10-04T09:38:49.716882Z","iopub.status.idle":"2024-10-04T09:38:49.717330Z","shell.execute_reply.started":"2024-10-04T09:38:49.717113Z","shell.execute_reply":"2024-10-04T09:38:49.717136Z"}}
import pickle

# Save the model
with open('model_1.pkl', 'wb') as file:
    pickle.dump(xgboost_with_params(train_df, 'HasDetections', learning_rate=0.5, balance_weight=True), file)



# %% [code] {"execution":{"iopub.status.busy":"2024-10-04T09:38:49.718930Z","iopub.status.idle":"2024-10-04T09:38:49.719392Z","shell.execute_reply.started":"2024-10-04T09:38:49.719167Z","shell.execute_reply":"2024-10-04T09:38:49.719191Z"}}
train_df.info()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-04T09:38:49.720685Z","iopub.status.idle":"2024-10-04T09:38:49.721094Z","shell.execute_reply.started":"2024-10-04T09:38:49.720889Z","shell.execute_reply":"2024-10-04T09:38:49.720909Z"}}
train_df.to_csv('cleand_data.csv', index=False)