from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Framefinder | Movie discovery",
    page_icon=":material/movie_filter:",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_data():
    movies = pd.read_csv(PROJECT_ROOT / "data/processed/movies_processed.csv")
    train_ratings = pd.read_csv(PROJECT_ROOT / "data/processed/train_ratings.csv")
    user_rated_movies = (
        train_ratings.groupby("userId")["movieId"].agg(set).to_dict()
    )
    user_stats = train_ratings.groupby("userId")["rating"].agg(
        ratings_count="count",
        average_rating="mean",
    )
    return movies, train_ratings, user_rated_movies, user_stats


@st.cache_resource
def load_models():
    with open(PROJECT_ROOT / "models/svd_model.pkl", "rb") as model_file:
        svd_model = pickle.load(model_file)

    with open(PROJECT_ROOT / "models/movie_similarity.pkl", "rb") as model_file:
        movie_similarity = pickle.load(model_file)

    return svd_model, movie_similarity


@st.cache_data(max_entries=128)
def get_user_genre_profile(user_id):
    rated_movies = train_ratings.loc[
        train_ratings["userId"] == user_id,
        ["movieId", "rating"],
    ].merge(movies[["movieId", "genres"]], on="movieId", how="left")
    genre_ratings = rated_movies.assign(
        genre=rated_movies["genres"].fillna("").str.split("|")
    ).explode("genre")
    genre_ratings = genre_ratings.loc[genre_ratings["genre"].ne("")]

    return (
        genre_ratings.groupby("genre", as_index=False)
        .agg(
            ratings_count=("rating", "count"),
            average_rating=("rating", "mean"),
        )
        .sort_values(["ratings_count", "average_rating"], ascending=[False, False])
        .head(10)
    )


def recommend_hybrid(user_id, n=10, selected_genres=()):
    if user_id not in user_rated_movies:
        return None

    rated_movies = user_rated_movies[user_id]
    candidate_movies = set(movies["movieId"]) - rated_movies

    if selected_genres:
        selected_genres_set = set(selected_genres)
        movie_genres = movies.set_index("movieId")["genres"].fillna("")
        candidate_movies = {
            movie_id
            for movie_id in candidate_movies
            if selected_genres_set.intersection(
                movie_genres.get(movie_id, "").split("|")
            )
        }

    rated_indices = [
        movie_id_to_index[movie_id]
        for movie_id in rated_movies
        if movie_id in movie_id_to_index
    ]
    recommendations = []

    for movie_id in candidate_movies:
        svd_prediction = svd_model.predict(user_id, movie_id).est
        movie_index = movie_id_to_index[movie_id]
        content_score = (
            np.max(movie_similarity[movie_index, rated_indices])
            if rated_indices
            else 0
        )
        svd_score = (svd_prediction - 1) / 4
        hybrid_score = 0.5 * svd_score + 0.5 * content_score

        recommendations.append(
            {
                "movieId": movie_id,
                "svd_score": svd_score,
                "content_score": content_score,
                "hybrid_score": hybrid_score,
            }
        )

    result_columns = [
        "movieId",
        "title",
        "genres",
        "svd_score",
        "content_score",
        "hybrid_score",
    ]
    if not recommendations:
        return pd.DataFrame(columns=result_columns)

    recommendations = (
        pd.DataFrame(recommendations)
        .sort_values("hybrid_score", ascending=False)
        .head(n)
        .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    )
    return recommendations[result_columns]


st.title("Movie recommendations", icon=":material/movie_filter:")
st.caption(
    "Discover films matched to a MovieLens profile, with collaborative and content-based signals."
)

try:
    with st.spinner("Loading the movie catalog and recommendation models..."):
        movies, train_ratings, user_rated_movies, user_stats = load_data()
        svd_model, movie_similarity = load_models()
except FileNotFoundError as error:
    st.error(f"A required project file is missing: {error.filename}")
    st.stop()

movie_id_to_index = pd.Series(movies.index, index=movies["movieId"]).to_dict()
genre_options = sorted(
    {
        genre
        for genres in movies["genres"].fillna("")
        for genre in genres.split("|")
        if genre
    }
)

with st.sidebar:
    st.markdown("## Framefinder")
    st.caption("Personalized movie discovery")
    st.metric("Movies in catalog", f"{len(movies):,}")
    st.metric("Profiles with ratings", f"{len(user_stats):,}")
    st.caption(f"Built from {len(train_ratings):,} training ratings")
    st.markdown("---")
    st.caption("Hybrid ranking blends predicted preference with similarity to films already rated.")

st.subheader("Build your recommendations", icon=":material/tune:")
with st.form("recommendation_preferences", border=True):
    user_column, count_column, genre_column = st.columns([1, 1, 2])
    with user_column:
        user_id = st.number_input(
            "MovieLens user ID",
            min_value=int(user_stats.index.min()),
            max_value=int(user_stats.index.max()),
            value=int(user_stats.index.min()),
            step=1,
            help="Choose a profile ID from the MovieLens 1M dataset.",
        )
    with count_column:
        n_recommendations = st.select_slider(
            "Movies to show",
            options=[5, 10, 15, 20],
            value=10,
        )
    with genre_column:
        selected_genres = st.multiselect(
            "Limit to genres",
            options=genre_options,
            placeholder="Any genre",
            help="Leave empty to include every genre. Selected genres are combined.",
        )
    submitted = st.form_submit_button(
        "Find movies",
        type="primary",
        icon=":material/auto_awesome:",
        width="stretch",
    )

if submitted:
    if int(user_id) not in user_rated_movies:
        st.session_state.pop("recommendation_results", None)
        st.session_state.pop("recommendation_user", None)
        st.error("That profile has no ratings in the training data. Try another ID.")
    else:
        with st.spinner("Finding films that fit this profile..."):
            st.session_state["recommendation_results"] = recommend_hybrid(
                int(user_id),
                int(n_recommendations),
                tuple(selected_genres),
            )
            st.session_state["recommendation_user"] = int(user_id)

results = st.session_state.get("recommendation_results")
result_user = st.session_state.get("recommendation_user")

if results is None:
    st.info("Choose a profile and select **Find movies** to see personalized picks.", icon=":material/lightbulb:")
elif results.empty:
    st.warning("No unrated films matched those genres. Try removing a genre filter.")
else:
    profile = user_stats.loc[result_user]
    with st.container(horizontal=True):
        st.metric("Recommendations", f"{len(results)}", border=True)
        st.metric("Films rated", f"{int(profile['ratings_count']):,}", border=True)
        st.metric("Average rating", f"{profile['average_rating']:.1f} / 5", border=True)

    st.subheader(f"Picked for profile {result_user}")
    st.caption("Already-rated films are excluded. Expand a pick to inspect its ranking signals.")

    st.subheader("Taste and score signals", icon=":material/monitoring:")
    genre_column, score_column = st.columns(2)
    with genre_column:
        st.markdown("**Genres in your ratings**")
        genre_profile = get_user_genre_profile(result_user)
        st.bar_chart(
            genre_profile.set_index("genre")[["ratings_count"]],
            horizontal=True,
            height=340,
            x_label="Movies rated",
        )
    with score_column:
        st.markdown("**Recommendation score comparison**")
        score_chart = results.head(10)[
            ["title", "svd_score", "content_score"]
        ].rename(
            columns={
                "title": "Movie",
                "svd_score": "Predicted preference",
                "content_score": "Content similarity",
            }
        ).set_index("Movie")
        st.bar_chart(
            score_chart,
            horizontal=True,
            height=340,
            x_label="Component score",
        )
        st.caption("Top 10 recommendations")

    for start in range(0, len(results), 2):
        result_columns = st.columns(2)
        displayed_movies = zip(
            result_columns,
            results.iloc[start : start + 2].iterrows(),
        )
        for rank, (column, (_, movie)) in enumerate(displayed_movies, start=start + 1):
            with column:
                with st.container(border=True):
                    st.caption(f"PICK {rank:02d}")
                    st.markdown(f"### {movie['title']}")
                    genre_text = " | ".join(
                        genre for genre in str(movie["genres"]).split("|") if genre
                    )
                    st.caption(genre_text or "Genre not listed")
                    st.metric("Hybrid score", f"{movie['hybrid_score']:.3f}")
                    with st.expander("Why this movie"):
                        score_column, similarity_column = st.columns(2)
                        score_column.metric("Predicted preference", f"{movie['svd_score']:.3f}")
                        similarity_column.metric("Content similarity", f"{movie['content_score']:.3f}")

    export_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export recommendations",
        data=export_data,
        file_name=f"framefinder-user-{result_user}.csv",
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )