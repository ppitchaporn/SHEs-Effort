import pandas as pd

def get_genres_of_movie(csv_file, movie_title):
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)
        
        # Filter the DataFrame to find the row(s) where the 'Title' column matches the movie title
        movie_df = df[df['Title'] == movie_title]
        
        # Check if the movie is found
        if movie_df.empty:
            return f"Movie '{movie_title}' not found in the dataset."
        
        # Extract the 'Genre' column from the filtered DataFrame
        genres = movie_df['Genre'].values[0]
        
        return genres
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    csv_file = "IMDB-Movie-Data.csv"
    movie_title = "Inception"
    genres = get_genres_of_movie(csv_file, movie_title)
    print(genres)