################################################### Assignment 2 ###########################################################
# Unit: PRT564 - Data Analytics and Visualization                                                                          #
# Group Name: Group 10                                                                                                     #
#   Group member:                                                                                                          #
#       Anne (Dao Phuong Anh) Ta    - S359453                                                                              # 
#       Khai Quang Thang            - S367530                                                                              #   
#       Buu Dang Phan               - S373294                                                                              #
#       Van Phuc Vinh Ho            - S366270                                                                              #
############################################################################################################################
# Project objectives                                                                                                       #
# 1. To explore relevant, interesting, and actionable trends of past retractions (essential objective)                     #
# 2. To predict important aspects of future retractions (desirable objective)                                              #
############################################################################################################################
import pandas as pd
import re
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
def rank_and_group(df, column_name):
    # Define the group boundaries
    group_boundaries = np.arange(1000, 9001, 1000)

    # Create a new column for the groups
    df[column_name + '_group'] = np.digitize(df[column_name], bins=group_boundaries) + 1

    # Any value greater than 9000 or None is assigned to group 0
    df.loc[df[column_name] > 9000, column_name + '_group'] = 0
    df.loc[df[column_name].isna(), column_name + '_group'] = 0

    return df

df = pd.read_csv('download_data_model_v1.0.csv')
#print(df)
#--------------------------------------------------------------------------------
# Feature Engineering: No. of Institution
#--------------------------------------------------------------------------------
# Extract data for Institution
df['Institution_Count'] = df['Institution'].str.count(';') + 1
#--------------------------------------------------------------------------------
# Feature Engineering: ENCODER THE NEW COLUMN FROM JOURNAL
#--------------------------------------------------------------------------------
label_encoder = LabelEncoder()
df['Journal_encodeder'] = label_encoder.fit_transform(df['Journal'])
journal_counts = df['Journal'].value_counts()
top_10_journal = journal_counts.nlargest(10)
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Feature Engineering: ENCODER THE NEW COLUMN FROM PUBLISHER
#--------------------------------------------------------------------------------
label_encoder = LabelEncoder()
df['Publisher_encodeder'] = label_encoder.fit_transform(df['Publisher'])
pub_counts = df['Publisher'].value_counts()
top_10_publisher = pub_counts.nlargest(10)
# #--------------------------------------------------------------------------------
df = rank_and_group(df,'Rank')

# print(df)
df['Class'] = df['Document Type'].replace({'Retracted': 1, 'Article': 0})
# print(df.columns)

df1 = df[df['Class'] == 1]
# print('Data 1', df1)
df2 = df[df['Class'] == 0]
# print('Data 2', df2)


columns_model = ['Class','Cited by', 'Rank_group', 'Title Length', 'Country_Count',
       'Author_Count', 'Institution_Count', 'Journal_encodeder',
       'Publisher_encodeder',]
df1_model =df1[columns_model]
# print(df1_model)
# df1_model.to_csv('df_model1.csv', index= False)
df2_model = df2[columns_model]

def stratified_sampling(df, column_name, sample_size):
    # Create an empty DataFrame to store the sampled data
    sampled_df = pd.DataFrame(columns=df.columns)
    
    # Calculate the number of samples to be taken from each category
    category_counts = df[column_name].value_counts()
    category_sample_sizes = (category_counts / category_counts.sum() * sample_size).astype(int)
    
    # Iterate over the unique values of the stratifying column
    for category, size in category_sample_sizes.items():
        # Sample data for the current category
        category_samples = df[df[column_name] == category].sample(n=size, random_state=42)
        
        # Append the sampled data to the result DataFrame
        sampled_df = pd.concat([sampled_df, category_samples])
    
    return sampled_df

df2_model = stratified_sampling(df2_model,'Journal_encodeder', 1195)
# print(df2_model)
# df2_model.to_csv('df_model2.csv', index= False)

df_model= pd.concat([df1_model, df2_model])
# print(df_model)
df_model.to_csv('df_model.csv', index= False)