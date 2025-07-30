import pandas as pd

def get_high_rating_movies(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Sort the DataFrame by 'Rating' in descending order
    df_sorted = df.sort_values(by='Rating', ascending=False)

    # Return the top 10 high-rated movies
    return df_sorted.head(10)

if __name__ == "__main__":
    csv_file = "IMDB-Movie-Data.csv"
    high_rating_movies = get_high_rating_movies(csv_file)
    print(high_rating_movies)