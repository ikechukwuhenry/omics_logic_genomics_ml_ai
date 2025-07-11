import pandas as pd
import numpy as np
import os

tcga_data_dir = os.getcwd()
filepath = os.path.join(tcga_data_dir, 'TCGA Gastric TPM.csv')
df = pd.read_csv(filepath)

print(df.head())
print(df.describe())
print(df.info())
# Check for missing values
print(df.isnull().sum())    
# Check for duplicate rows
print(df.duplicated().sum())
# Check for unique values in each column
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")
# Check for data types of each column
print(df.dtypes)    



# Get the name of the first column
status = df.columns[0]

# Rename the first column
new_column_name = 'label'  # New name for the first column
df.rename(columns={status: new_column_name}, inplace=True)

column_names = df.columns.to_list()
print("Column names in the dataset: ", column_names)
# df['label'] 
df.iloc[:420, 0] = 1 # Assuming the first column is the label
print("First 10 rows of the label column:")

df.iloc[420:, 0] = 0 
print(df.iloc[1:421, 0])
print(df.tail(10))
transformed_df = "transformed_tcga.csv"
new_path = os.path.join(tcga_data_dir, transformed_df)
# Save the transformed DataFrame to a new CSV file
df.to_csv(new_path, index=False)
# Check for outliers using boxplot
import matplotlib.pyplot as plt
import seaborn as sns       


plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Boxplot of TCGA Gastric TPM Data')
plt.show()