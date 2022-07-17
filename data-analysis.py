import pandas as pd

# loading the data from csv file to a Pandas DataFrame 
parkinsons_data = pd.read_csv("/data/parkinsons.csv")
save_data = '/data/processed_data.pkl'

# printing the first five rows of the dataframe
parkinsons_data.head()

# number of rows and columns in the dataframe
parkinsons_data.shape 

# getting more informations about the dataset
parkinsons_data.info()

#checking for missing values in each column
parkinsons_data.isnull().sum()

# getting some statistical measures about the data
parkinsons_data.describe()

# distribution of target variable
parkinsons_data['status']. value_counts()

# growing the data based on the target variable
parkinsons_data.groupby('status').mean()

parkinsons_data.to_pickle(save_data)