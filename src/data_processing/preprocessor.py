"""
RecoFlix Data Preprocessor
Handles data cleaning, feature engineering, and preparation for recommendation algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DATA_DIR, RECOMMENDATION_CONFIG

class RecoFlixPreprocessor:
    """
    Data preprocessing class for RecoFlix recommendation system
    """
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.genre_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.scaler = StandardScaler()
        
        # Processed data storage
        self.user_item_matrix = None
        self.movie_features = None
        self.user_profiles = None
        self.train_data = None
        self.test_data = None
        
        self.is_fitted = False
        
    def fit_transform(self, ratings_df, movies_df, tags_df=None):
        """
        Complete preprocessing pipeline
        
        Args:
            ratings_df: User ratings DataFrame
            movies_df: Movies metadata DataFrame
            tags_df: User tags DataFrame (optional)
            
        Returns:
            dict: Dictionary containing processed datasets
        """
        print("🔄 Starting RecoFlix data preprocessing...")
        
        # Step 1: Clean and filter data
        clean_ratings, clean_movies = self._clean_data(ratings_df, movies_df)
        
        # Step 2: Create user-item interaction matrix
        self.user_item_matrix = self._create_user_item_matrix(clean_ratings)
        
        # Step 3: Process movie features
        self.movie_features = self._process_movie_features(clean_movies, tags_df)
        
        # Step 4: Create user profiles
        self.user_profiles = self._create_user_profiles(clean_ratings, self.movie_features)
        
        # Step 5: Split data for training/testing
        self.train_data, self.test_data = self._split_data(clean_ratings)
        
        # Step 6: Save processed data
        self._save_processed_data()
        
        self.is_fitted = True
        
        print("✅ Data preprocessing completed successfully!")
        
        return {
            'user_item_matrix': self.user_item_matrix,
            'movie_features': self.movie_features,
            'user_profiles': self.user_profiles,
            'train_data': self.train_data,
            'test_data': self.test_data
        }
    
    def _clean_data(self, ratings_df, movies_df):
        """Clean and filter the raw data"""
        print("🧹 Cleaning data...")
        
        # Remove users and movies with insufficient interactions
        min_ratings_user = RECOMMENDATION_CONFIG['min_ratings_per_user']
        min_ratings_movie = RECOMMENDATION_CONFIG['min_ratings_per_movie']
        
        # Filter users with minimum ratings
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings_user].index
        clean_ratings = ratings_df[ratings_df['userId'].isin(valid_users)].copy()
        
        # Filter movies with minimum ratings
        movie_counts = clean_ratings['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_ratings_movie].index
        clean_ratings = clean_ratings[clean_ratings['movieId'].isin(valid_movies)].copy()
        
        # Filter movies dataframe to match valid movies
        clean_movies = movies_df[movies_df['movieId'].isin(valid_movies)].copy()
        
        # Handle missing values
        clean_movies['genres'] = clean_movies['genres'].fillna('(no genres listed)')
        
        print(f"   • Filtered to {clean_ratings['userId'].nunique()} users")
        print(f"   • Filtered to {clean_ratings['movieId'].nunique()} movies")
        print(f"   • Remaining ratings: {len(clean_ratings):,}")
        
        return clean_ratings, clean_movies
    
    def _create_user_item_matrix(self, ratings_df):
        """Create user-item interaction matrix"""
        print("📊 Creating user-item matrix...")
        
        # Create pivot table
        user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating', 
            fill_value=0
        )
        
        print(f"   • Matrix shape: {user_item_matrix.shape}")
        print(f"   • Sparsity: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.2f}%")
        
        return user_item_matrix
    
    def _process_movie_features(self, movies_df, tags_df=None):
        """Process movie features for content-based filtering"""
        print("🎬 Processing movie features...")
        
        movie_features = movies_df.copy()
        
        # Extract year from title
        movie_features['year'] = movie_features['title'].str.extract(r'\((\d{4})\)')
        movie_features['year'] = pd.to_numeric(movie_features['year'], errors='coerce')
        movie_features['year'] = movie_features['year'].fillna(movie_features['year'].median())
        
        # Process genres
        movie_features['genre_list'] = movie_features['genres'].str.split('|')
        movie_features['num_genres'] = movie_features['genre_list'].apply(len)
        
        # Create genre binary features
        all_genres = []
        for genres in movie_features['genres']:
            if genres != '(no genres listed)':
                all_genres.extend(genres.split('|'))
        
        unique_genres = list(set(all_genres))
        for genre in unique_genres:
            movie_features[f'genre_{genre}'] = movie_features['genres'].str.contains(genre, na=False).astype(int)
        
        # Create TF-IDF features for genres
        genre_text = movie_features['genres'].str.replace('|', ' ')
        genre_tfidf = self.genre_vectorizer.fit_transform(genre_text)
        
        # Add TF-IDF features to movie features
        tfidf_df = pd.DataFrame(
            genre_tfidf.toarray(), 
            columns=[f'tfidf_{feature}' for feature in self.genre_vectorizer.get_feature_names_out()],
            index=movie_features.index
        )
        movie_features = pd.concat([movie_features, tfidf_df], axis=1)
        
        # Process tags if available
        if tags_df is not None and len(tags_df) > 0:
            # Aggregate tags per movie
            movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
            movie_features = movie_features.merge(movie_tags, on='movieId', how='left')
            movie_features['tag'] = movie_features['tag'].fillna('')
            
            # Create tag TF-IDF features
            if movie_features['tag'].str.len().sum() > 0:
                tag_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                tag_tfidf = tag_vectorizer.fit_transform(movie_features['tag'])
                tag_tfidf_df = pd.DataFrame(
                    tag_tfidf.toarray(),
                    columns=[f'tag_tfidf_{feature}' for feature in tag_vectorizer.get_feature_names_out()],
                    index=movie_features.index
                )
                movie_features = pd.concat([movie_features, tag_tfidf_df], axis=1)
        
        print(f"   • Total features per movie: {movie_features.shape[1]}")
        print(f"   • Unique genres: {len(unique_genres)}")
        
        return movie_features
    
    def _create_user_profiles(self, ratings_df, movie_features):
        """Create user preference profiles"""
        print("👤 Creating user profiles...")
        
        user_profiles = {}
        
        for user_id in ratings_df['userId'].unique():
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            # Calculate basic statistics
            profile = {
                'avg_rating': user_ratings['rating'].mean(),
                'rating_std': user_ratings['rating'].std(),
                'num_ratings': len(user_ratings),
                'rating_range': user_ratings['rating'].max() - user_ratings['rating'].min()
            }
            
            # Calculate genre preferences
            user_movies = user_ratings.merge(movie_features, on='movieId', how='left')
            
            # Weight by rating (higher ratings = stronger preference)
            weights = user_ratings.set_index('movieId')['rating']
            
            genre_preferences = {}
            genre_cols = [col for col in movie_features.columns if col.startswith('genre_')]
            
            for genre_col in genre_cols:
                genre_name = genre_col.replace('genre_', '')
                genre_movies = user_movies[user_movies[genre_col] == 1]
                if len(genre_movies) > 0:
                    weighted_rating = (genre_movies['rating'] * genre_movies[genre_col]).sum() / genre_movies[genre_col].sum()
                    genre_preferences[genre_name] = weighted_rating
                else:
                    genre_preferences[genre_name] = 0
            
            profile['genre_preferences'] = genre_preferences
            
            # Calculate temporal patterns if timestamp available
            if 'timestamp' in user_ratings.columns:
                user_ratings['datetime'] = pd.to_datetime(user_ratings['timestamp'], unit='s')
                user_ratings['hour'] = user_ratings['datetime'].dt.hour
                user_ratings['day_of_week'] = user_ratings['datetime'].dt.dayofweek
                
                profile['preferred_hours'] = user_ratings['hour'].mode().tolist()
                profile['preferred_days'] = user_ratings['day_of_week'].mode().tolist()
            
            user_profiles[user_id] = profile
        
        print(f"   • Created profiles for {len(user_profiles)} users")
        
        return user_profiles
    
    def _split_data(self, ratings_df):
        """Split data into training and testing sets"""
        print("🔀 Splitting data for training/testing...")
        
        # Use stratified split to maintain user distribution
        test_size = RECOMMENDATION_CONFIG['test_size']
        random_state = RECOMMENDATION_CONFIG['random_state']
        
        # For each user, split their ratings
        train_data = []
        test_data = []
        
        for user_id in ratings_df['userId'].unique():
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            if len(user_ratings) >= 4:  # Ensure enough data to split
                train_user, test_user = train_test_split(
                    user_ratings, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=None  # Can't stratify with small samples
                )
                train_data.append(train_user)
                test_data.append(test_user)
            else:
                # Put all data in training if too few ratings
                train_data.append(user_ratings)
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        print(f"   • Training set: {len(train_df):,} ratings")
        print(f"   • Test set: {len(test_df):,} ratings")
        
        return train_df, test_df
    
    def _save_processed_data(self):
        """Save processed data for later use"""
        print("💾 Saving processed data...")
        
        # Create directory if it doesn't exist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save data
        datasets = {
            'user_item_matrix': self.user_item_matrix,
            'movie_features': self.movie_features,
            'user_profiles': self.user_profiles,
            'train_data': self.train_data,
            'test_data': self.test_data
        }
        
        for name, data in datasets.items():
            file_path = PROCESSED_DATA_DIR / f'{name}.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"   • Saved {name}")
        
        # Save encoders and transformers
        transformers = {
            'user_encoder': self.user_encoder,
            'movie_encoder': self.movie_encoder,
            'genre_vectorizer': self.genre_vectorizer,
            'scaler': self.scaler
        }
        
        for name, transformer in transformers.items():
            file_path = PROCESSED_DATA_DIR / f'{name}.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(transformer, f)
        
        print("   • All processed data saved successfully!")
    
    def load_processed_data(self):
        """Load previously processed data"""
        try:
            datasets = {}
            for name in ['user_item_matrix', 'movie_features', 'user_profiles', 'train_data', 'test_data']:
                file_path = PROCESSED_DATA_DIR / f'{name}.pkl'
                with open(file_path, 'rb') as f:
                    datasets[name] = pickle.load(f)
            
            self.user_item_matrix = datasets['user_item_matrix']
            self.movie_features = datasets['movie_features']
            self.user_profiles = datasets['user_profiles']
            self.train_data = datasets['train_data']
            self.test_data = datasets['test_data']
            
            self.is_fitted = True
            print("✅ Processed data loaded successfully!")
            return datasets
            
        except FileNotFoundError:
            print("❌ No processed data found. Run fit_transform() first.")
            return None
    
    def get_recommendation_data(self):
        """Get data formatted for recommendation algorithms"""
        if not self.is_fitted:
            print("❌ Preprocessor not fitted. Run fit_transform() first.")
            return None
        
        return {
            'user_item_matrix': self.user_item_matrix,
            'movie_features': self.movie_features,
            'user_profiles': self.user_profiles,
            'train_data': self.train_data,
            'test_data': self.test_data,
            'movie_ids': self.user_item_matrix.columns.tolist(),
            'user_ids': self.user_item_matrix.index.tolist()
        }
    
    def get_data_summary(self):
        """Get summary of processed data"""
        if not self.is_fitted:
            return "Preprocessor not fitted."
        
        summary = {
            'users': len(self.user_item_matrix.index),
            'movies': len(self.user_item_matrix.columns),
            'total_ratings': self.user_item_matrix.values.sum(),
            'sparsity': (self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100,
            'avg_ratings_per_user': self.user_item_matrix.sum(axis=1).mean(),
            'avg_ratings_per_movie': self.user_item_matrix.sum(axis=0).mean(),
            'movie_features': self.movie_features.shape[1] if self.movie_features is not None else 0
        }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    from data_loader import RecoFlixDataLoader
    
    # Load data
    loader = RecoFlixDataLoader()
    data = loader.load_all_data()
    
    if data:
        # Preprocess data
        preprocessor = RecoFlixPreprocessor()
        processed_data = preprocessor.fit_transform(
            data['ratings'], 
            data['movies'], 
            data['tags']
        )
        
        # Print summary
        summary = preprocessor.get_data_summary()
        print("\n📊 RecoFlix Processed Data Summary:")
        print(f"• Users: {summary['users']}")
        print(f"• Movies: {summary['movies']}")
        print(f"• Data sparsity: {summary['sparsity']:.2f}%")
        print(f"• Movie features: {summary['movie_features']}")
        
        print("\n🎉 Data preprocessing completed! Ready for recommendation algorithms.")
    else:
        print("❌ Could not load data. Check your data files.")