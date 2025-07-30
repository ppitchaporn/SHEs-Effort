import pandas as pd

def count_movies_released_in_year(csv_file, year):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Count the number of movies released in the given year
        count = df[df['Year'] == year].shape[0]
        
        return count
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    csv_file = "IMDB-Movie-Data.csv"
    year = 2000
    count = count_movies_released_in_year(csv_file, year)
    print(f"Number of movies released in {year}: {count}")