import pandas as pd

def check_unique_years(csv_file):
 # Read the CSV file into a DataFrame
 df = pd.read_csv(csv_file)
 
 # Get the unique values in the 'Year' column
 unique_years = df['Year'].unique()
 
 return unique_years

if __name__ == "__main__":
 csv_file = "IMDB-Movie-Data.csv"
 unique_years = check_unique_years(csv_file)
 print(f"Unique years in the dataset: {unique_years}")