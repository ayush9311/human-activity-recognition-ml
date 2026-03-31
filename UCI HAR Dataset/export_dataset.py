import pandas as pd

base_path = "/Users/ayushsingh/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/"

X_train = pd.read_csv(base_path + "train/X_train.txt", sep=r'\s+', header=None)
X_test  = pd.read_csv(base_path + "test/X_test.txt", sep=r'\s+', header=None)
y_train = pd.read_csv(base_path + "train/y_train.txt", header=None)
y_test  = pd.read_csv(base_path + "test/y_test.txt", header=None)

X_full = pd.concat([X_train, X_test], axis=0)
y_full = pd.concat([y_train, y_test], axis=0)
y_full.columns = ["Activity"]

dataset_full = pd.concat([X_full, y_full], axis=1)

dataset_full.to_excel(
    "UCI_HAR_Full_Dataset.xlsx",
    index=False,
    engine="openpyxl"
)

print("Excel file created successfully!")