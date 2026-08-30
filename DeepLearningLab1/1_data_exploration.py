import pandas as pd

df = pd.read_csv("data/banknote_authentication.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass counts:")
print(df["class"].value_counts())

print("\nDescribe:")
print(df.describe())
