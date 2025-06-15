"""
RecoFlix Configuration File
Central configuration for the movie recommendation system
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
ASSETS_DIR = PROJECT_ROOT / "streamlit_app" / "assets"
IMAGES_DIR = ASSETS_DIR / "images"

# Data file paths
DATA_FILES = {
    'ratings': RAW_DATA_DIR / 'ratings.csv',
    'movies': RAW_DATA_DIR / 'movies.csv',
    'tags': RAW_DATA_DIR / 'tags.csv',
    'links': RAW_DATA_DIR / 'links.csv'
}

# Processed data paths
PROCESSED_FILES = {
    'user_item_matrix': PROCESSED_DATA_DIR / 'user_item_matrix.pkl',
    'movie_features': PROCESSED_DATA_DIR / 'movie_features.pkl',
    'user_profiles': PROCESSED_DATA_DIR / 'user_profiles.pkl',
    'similarity_matrix': PROCESSED_DATA_DIR / 'similarity_matrix.pkl'
}

# Model paths
MODEL_FILES = {
    'collaborative_model': MODELS_DIR / 'collaborative_model.pkl',
    'content_model': MODELS_DIR / 'content_model.pkl',
    'hybrid_model': MODELS_DIR / 'hybrid_model.pkl'
}

# Recommendation system parameters
RECOMMENDATION_CONFIG = {
    'n_recommendations': 10,
    'min_ratings_per_user': 5,
    'min_ratings_per_movie': 5,
    'similarity_threshold': 0.1,
    'test_size': 0.2,
    'random_state': 42
}

# Streamlit app configuration
APP_CONFIG = {
    'page_title': 'RecoFlix - Movie Recommendation System',
    'page_icon': '🎬',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# UI Colors and Styling
COLORS = {
    'primary': '#E50914',      # Netflix red
    'secondary': '#221F1F',    # Dark gray
    'background': '#141414',   # Almost black
    'text': '#FFFFFF',         # White text
    'accent': '#FFA500'        # Orange accent
}

# Movie poster configuration
POSTER_CONFIG = {
    'default_poster': IMAGES_DIR / 'default_poster.jpg',
    'poster_width': 200,
    'poster_height': 300,
    'tmdb_api_key': os.getenv('TMDB_API_KEY', ''),  # Add your TMDB API key to .env
    'poster_base_url': 'https://image.tmdb.org/t/p/w500'
}

# Evaluation metrics
EVALUATION_METRICS = [
    'rmse',
    'mae', 
    'precision_at_k',
    'recall_at_k',
    'coverage',
    'diversity'
]

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("✅ RecoFlix configuration loaded successfully!")