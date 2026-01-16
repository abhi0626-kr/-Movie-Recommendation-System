import streamlit as st
import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Content-Based Recommender using TMDB 5000 Movies Dataset")

# --------------------------------------------------
# Load Data (Cached)
# --------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    return movies

movies = load_data()

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def extract_names(text):
    try:
        return [i["name"] for i in ast.literal_eval(text)]
    except:
        return []

def get_top_cast(text):
    try:
        return [i["name"] for i in ast.literal_eval(text)[:3]]
    except:
        return []

def get_director(text):
    try:
        for i in ast.literal_eval(text):
            if i["job"] == "Director":
                return i["name"]
        return ""
    except:
        return ""

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
movies["overview"] = movies["overview"].fillna("")

movies["genres_list"] = movies["genres"].apply(extract_names)
movies["keywords"] = movies["keywords"].apply(extract_names)
movies["cast"] = movies["cast"].apply(get_top_cast)
movies["director"] = movies["crew"].apply(get_director)

# Normalize text
movies["overview"] = movies["overview"].str.lower()

for col in ["genres_list", "keywords", "cast"]:
    movies[col] = movies[col].apply(
        lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else []
    )

movies["director"] = movies["director"].fillna("").replace(" ", "")

# Combine into tags
movies["tags"] = (
    movies["overview"] + " " +
    movies["genres_list"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["director"]
)

# --------------------------------------------------
# ✅ SAFE final_df CREATION (MANDATORY FIX)
# --------------------------------------------------
final_df = movies[["title", "tags", "vote_average", "vote_count"]].copy()

final_df["tags"] = final_df["tags"].fillna("")
final_df = final_df[final_df["tags"].str.strip() != ""]
final_df.reset_index(drop=True, inplace=True)

# --------------------------------------------------
# TF-IDF Vectorization
# --------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=6000,
    stop_words="english"
)

vectors = tfidf.fit_transform(final_df["tags"])
similarity = cosine_similarity(vectors)

# --------------------------------------------------
# Weighted Rating (IMDb Formula)
# --------------------------------------------------
C = final_df["vote_average"].mean()
m = final_df["vote_count"].quantile(0.7)

def weighted_rating(x):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (m + v) * C)

final_df["score"] = final_df.apply(weighted_rating, axis=1)

# --------------------------------------------------
# Recommendation Function (Case-Insensitive)
# --------------------------------------------------
def recommend(movie_name, top_n=5):
    movie_name = movie_name.lower()
    titles = final_df["title"].str.lower()

    if movie_name not in titles.values:
        return None

    idx = final_df[titles == movie_name].index[0]

    similarity_scores = list(enumerate(similarity[idx]))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:50]

    candidates = final_df.iloc[[i[0] for i in similarity_scores]]
    candidates = candidates.sort_values(by="score", ascending=False)

    return candidates[["title", "vote_average", "score"]].head(top_n)

# --------------------------------------------------
# UI
# --------------------------------------------------
movie_input = st.text_input("Enter a movie name:")

if movie_input:
    suggestions = final_df[
        final_df["title"].str.lower().str.contains(movie_input.lower())
    ]["title"].head(10)

    selected_movie = st.selectbox(
        "Select a movie:",
        suggestions if not suggestions.empty else []
    )

    if st.button("Recommend"):
        result = recommend(selected_movie)

        if result is None:
            st.warning("Movie not found in dataset.")
        else:
            st.subheader("Recommended Movies")
            for _, row in result.iterrows():
                st.markdown(
                    f"**{row['title']}**  \n"
                    f"Rating: {row['vote_average']} | "
                    f"Score: {row['score']:.2f}"
                )
