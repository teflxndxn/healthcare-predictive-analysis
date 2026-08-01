import pandas as pd

# Load the dataset
df = pd.read_csv("healthcare_dataset.csv")

# Show the first five rows
print("First 5 Rows:")
print(df.head())

# Show the number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Show information about the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())