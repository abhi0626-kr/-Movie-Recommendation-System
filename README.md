# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Streamlit and powered by the TMDB 5000 Movies Dataset. This application uses TF-IDF vectorization and cosine similarity to recommend movies based on content features like genres, keywords, cast, and director.

## Features

- **Content-Based Filtering**: Recommends movies based on similarity in genres, keywords, cast, director, and overview
- **Weighted Rating System**: Uses IMDb's weighted rating formula to rank recommendations
- **Interactive UI**: Built with Streamlit for easy movie search and recommendations
- **Case-Insensitive Search**: Finds movies regardless of capitalization
- **Smart Suggestions**: Shows matching movies as you type

## Dataset

This project uses the **TMDB 5000 Movies Dataset** which includes:
- `tmdb_5000_movies.csv` - Movie metadata (title, overview, genres, keywords, ratings)
- `tmdb_5000_credits.csv` - Cast and crew information

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install pandas numpy scikit-learn streamlit
   ```

3. **Ensure data files are present**
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`

## Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

3. **Get recommendations**
   - Enter a movie name in the search box
   - Select from the suggestions dropdown
   - Click "Recommend" to see similar movies

## How It Works

### 1. Data Processing
- Loads and merges movie and credits datasets
- Extracts features: genres, keywords, top 3 cast members, director
- Creates a combined "tags" field from all features

### 2. TF-IDF Vectorization
- Converts text tags into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency)
- Limits to 6000 features for efficiency
- Removes English stop words

### 3. Similarity Calculation
- Computes cosine similarity between all movie vectors
- Higher similarity scores indicate more similar content

### 4. Weighted Rating
Uses IMDb's weighted rating formula:
```
WR = (v/(v+m) × R) + (m/(m+v) × C)
```
Where:
- `v` = number of votes for the movie
- `m` = minimum votes required (70th percentile)
- `R` = average rating of the movie
- `C` = mean rating across all movies

### 5. Recommendation Algorithm
- Finds the selected movie in the dataset
- Retrieves top 50 similar movies by cosine similarity
- Sorts by weighted rating score
- Returns top 5 recommendations

## Project Structure

```
movie-recommendation/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── tmdb_5000_movies.csv       # Movie metadata
├── tmdb_5000_credits.csv      # Cast and crew data
└── README.md                   # This file
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (TF-IDF, cosine similarity)
- **streamlit**: Web application framework

## Example

**Input:** "The Dark Knight"

**Output:**
- Batman Begins
- The Dark Knight Rises
- Inception
- The Prestige
- Interstellar

## Limitations

- Content-based approach doesn't consider user preferences or collaborative filtering
- Recommendations are based solely on content similarity, not popularity trends
- Limited to movies in the TMDB 5000 dataset

## Future Enhancements

- [ ] Add movie posters using TMDB API
- [ ] Implement hybrid recommendation (content + collaborative filtering)
- [ ] Add user rating system
- [ ] Include release year filtering
- [ ] Add genre-based filtering
- [ ] Display movie details (runtime, budget, revenue)

## License

This project uses the TMDB 5000 Movies Dataset. Please refer to the dataset's license for usage rights.

## Author

Built with ❤️ using Streamlit and scikit-learn
