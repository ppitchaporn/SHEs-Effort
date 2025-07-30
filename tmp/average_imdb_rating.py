import pandas as pd
def calculate_average_rating(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Calculate the average rating
    average_rating = df['Rating'].mean()
    
    return average_rating

if __name__ == "__main__":
    file_path = "IMDB-Movie-Data.csv"
    average_rating = calculate_average_rating(file_path)
    print(average_rating)