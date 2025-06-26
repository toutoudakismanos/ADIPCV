import pandas as pd

# Load the extracted features
df = pd.read_csv('train_radiomics_features.csv')

# 1. Inspect the first few rows
print(df.head())

# 2. Check dimensions: should be [#slices × (#features + 2)] 
#    (+2 for “subject” and “label” columns)
print("Shape:", df.shape)

# 3. Look for missing values
print("Missing per column:")
print(df.isnull().sum().sort_values())

# 4. Confirm label balance
print("Label counts:")
print(df['label'].value_counts())
