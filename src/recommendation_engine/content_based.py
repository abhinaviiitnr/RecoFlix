"""
RecoFlix Content-Based Filtering Engine
Recommends movies based on movie features and user preferences
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, RECOMMENDATION_CONFIG

class RecoFlixContentFilter:
    """
    Content-Based Filtering Recommendation Engine for RecoFlix
    Uses movie features and user preferences to make recommendations
    """
    
    def __init__(self, feature_weights=None, n_recommendations=10):
        """
        Initialize the content-based filter
        
        Args:
            feature_weights: Dictionary of feature weights for recommendation
            n_recommendations: Number of recommendations to generate
        """
        self.feature_weights = feature_weights or {
            'genres': 0.4,
            'year': 0.1,
            'popularity': 0.2,
            'tags': 0.2,
            'user_profile': 0.1
        }
        self.n_recommendations = n_recommendations
        
        # Model components
        self.movie_features = None
        self.user_profiles = None
        self.feature_matrix = None
        self.movie_similarity_matrix = None
        
        # Transformers
        self.genre_vectorizer = TfidfVectorizer(max_features=50)
        self.tag_vectorizer = TfidfVectorizer(max_features=30, stop_words='english')
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        
    def fit(self, movie_features, user_profiles, user_item_matrix):
        """
        Train the content-based filtering model
        
        Args:
            movie_features: DataFrame with movie features
            user_profiles: Dictionary of user preference profiles
            user_item_matrix: User-item interaction matrix for popularity calculation
        """
        print("🎬 Training RecoFlix content-based filter...")
        
        self.movie_features = movie_features.copy()
        self.user_profiles = user_profiles
        
        # Prepare feature matrix
        self.feature_matrix = self._create_feature_matrix(user_item_matrix)
        
        # Calculate movie similarity matrix
        self.movie_similarity_matrix = self._calculate_movie_similarity()
        
        self.is_fitted = True
        print("✅ Content-based model trained successfully!")
        
    def _create_feature_matrix(self, user_item_matrix):
        """Create comprehensive feature matrix for movies"""
        print("   • Creating movie feature matrix...")
        
        features = []
        movie_ids = []
        
        for _, movie in self.movie_features.iterrows():
            movie_id = movie['movieId']
            
            if movie_id not in user_item_matrix.columns:
                continue
                
            movie_ids.append(movie_id)
            movie_vector = []
            
            # 1. Genre features (TF-IDF)
            genres_text = movie['genres'].replace('|', ' ') if pd.notna(movie['genres']) else ''
            
            # 2. Year feature (normalized)
            year = movie.get('year', 2000)
            normalized_year = (year - 1900) / 100  # Normalize to 0-1 range
            
            # 3. Popularity features
            movie_ratings = user_item_matrix[movie_id]
            popularity_score = (movie_ratings > 0).sum()  # Number of ratings
            avg_rating = movie_ratings[movie_ratings > 0].mean() if (movie_ratings > 0).sum() > 0 else 0
            
            # 4. Tag features (if available)
            tag_text = movie.get('tag', '') if 'tag' in movie else ''
            
            # Store text features for later processing
            movie_vector.extend([
                genres_text,
                normalized_year,
                popularity_score,
                avg_rating,
                tag_text
            ])
            
            features.append(movie_vector)
        
        # Convert to DataFrame for easier processing
        feature_df = pd.DataFrame(features, index=movie_ids, columns=[
            'genres_text', 'year_norm', 'popularity', 'avg_rating', 'tag_text'
        ])
        
        # Process text features with TF-IDF
        # Genre TF-IDF
        genre_tfidf = self.genre_vectorizer.fit_transform(feature_df['genres_text'])
        genre_features = pd.DataFrame(
            genre_tfidf.toarray(),
            index=feature_df.index,
            columns=[f'genre_tfidf_{i}' for i in range(genre_tfidf.shape[1])]
        )
        
        # Tag TF-IDF (if tags available)
        if feature_df['tag_text'].str.len().sum() > 0:
            tag_tfidf = self.tag_vectorizer.fit_transform(feature_df['tag_text'])
            tag_features = pd.DataFrame(
                tag_tfidf.toarray(),
                index=feature_df.index,
                columns=[f'tag_tfidf_{i}' for i in range(tag_tfidf.shape[1])]
            )
        else:
            tag_features = pd.DataFrame(index=feature_df.index)
        
        # Combine all features
        numerical_features = feature_df[['year_norm', 'popularity', 'avg_rating']]
        combined_features = pd.concat([numerical_features, genre_features, tag_features], axis=1)
        
        # Scale numerical features
        if len(combined_features) > 0:
            scaled_features = self.scaler.fit_transform(combined_features.fillna(0))
            feature_matrix = pd.DataFrame(scaled_features, index=combined_features.index, columns=combined_features.columns)
        else:
            feature_matrix = combined_features
        
        print(f"   • Feature matrix shape: {feature_matrix.shape}")
        return feature_matrix
        
    def _calculate_movie_similarity(self):
        """Calculate similarity matrix between movies based on features"""
        print("   • Computing movie-movie similarity matrix...")
        
        if self.feature_matrix.empty:
            return pd.DataFrame()
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(self.feature_matrix)
        
        similarity_matrix = pd.DataFrame(
            similarity_scores,
            index=self.feature_matrix.index,
            columns=self.feature_matrix.index
        )
        
        print(f"   • Movie similarity matrix shape: {similarity_matrix.shape}")
        return similarity_matrix
        
    def get_user_content_profile(self, user_id, user_item_matrix):
        """
        Create content-based profile for a user based on their rating history
        
        Args:
            user_id: Target user ID
            user_item_matrix: User-item interaction matrix
            
        Returns:
            User's content preference vector
        """
        if user_id not in user_item_matrix.index:
            return None
        
        user_ratings = user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            return None
        
        # Get features for rated movies
        rated_movie_features = self.feature_matrix.loc[rated_movies.index]
        
        # Weight features by ratings (higher ratings = stronger preference)
        weights = rated_movies.values
        weighted_features = rated_movie_features.multiply(weights, axis=0)
        
        # Calculate weighted average profile
        user_profile = weighted_features.sum() / weights.sum()
        
        return user_profile
        
    def recommend_movies(self, user_id, user_item_matrix, n_recommendations=None, exclude_rated=True):
        """
        Generate content-based movie recommendations for a user
        
        Args:
            user_id: Target user ID
            user_item_matrix: User-item interaction matrix
            n_recommendations: Number of recommendations
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if n_recommendations is None:
            n_recommendations = self.n_recommendations
        
        # Handle cold start - new users
        if user_id not in user_item_matrix.index:
            return self._get_popular_content_movies(n_recommendations)
        
        # Get user's content profile
        user_profile = self.get_user_content_profile(user_id, user_item_matrix)
        
        if user_profile is None:
            return self._get_popular_content_movies(n_recommendations)
        
        # Get candidate movies
        if exclude_rated:
            user_ratings = user_item_matrix.loc[user_id]
            unrated_movies = user_ratings[user_ratings == 0].index
            candidate_movies = [m for m in unrated_movies if m in self.feature_matrix.index]
        else:
            candidate_movies = self.feature_matrix.index.tolist()
        
        if len(candidate_movies) == 0:
            return []
        
        # Calculate similarity between user profile and candidate movies
        candidate_features = self.feature_matrix.loc[candidate_movies]
        
        # Calculate cosine similarity with user profile
        similarities = []
        for movie_id in candidate_movies:
            movie_features = candidate_features.loc[movie_id]
            similarity = self._cosine_similarity_vectors(user_profile, movie_features)
            similarities.append((movie_id, similarity))
        
        # Sort by similarity and return top-N
        recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
        
    def _cosine_similarity_vectors(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0
        
    def get_similar_movies_content(self, movie_id, n_similar=10):
        """
        Get movies similar to a given movie based on content features
        
        Args:
            movie_id: Target movie ID
            n_similar: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted or movie_id not in self.movie_similarity_matrix.index:
            return []
        
        similar_movies = self.movie_similarity_matrix.loc[movie_id].nlargest(n_similar + 1)[1:]  # Exclude self
        
        return [(movie, similarity) for movie, similarity in similar_movies.items()]
        
    def _get_popular_content_movies(self, n_recommendations):
        """Get popular movies for cold start problem"""
        if self.feature_matrix.empty:
            return []
        
        # Get movies with high average ratings and popularity
        popularity_scores = []
        
        for movie_id in self.feature_matrix.index:
            if 'avg_rating' in self.feature_matrix.columns and 'popularity' in self.feature_matrix.columns:
                avg_rating = self.feature_matrix.loc[movie_id, 'avg_rating']
                popularity = self.feature_matrix.loc[movie_id, 'popularity']
                score = avg_rating * popularity  # Simple popularity score
                popularity_scores.append((movie_id, score))
        
        if not popularity_scores:
            # Fallback to random selection
            random_movies = np.random.choice(self.feature_matrix.index, 
                                           min(n_recommendations, len(self.feature_matrix)), 
                                           replace=False)
            return [(movie_id, 0.5) for movie_id in random_movies]
        
        # Sort by popularity score
        popular_movies = sorted(popularity_scores, key=lambda x: x[1], reverse=True)
        
        return popular_movies[:n_recommendations]
        
    def explain_recommendation(self, user_id, movie_id, user_item_matrix):
        """
        Provide explanation for why a movie was recommended
        
        Args:
            user_id: Target user ID
            movie_id: Recommended movie ID
            user_item_matrix: User-item interaction matrix
            
        Returns:
            Dictionary with explanation details
        """
        if not self.is_fitted:
            return {}
        
        explanation = {
            'movie_id': movie_id,
            'user_id': user_id,
            'reasons': []
        }
        
        # Get user profile and movie features
        user_profile = self.get_user_content_profile(user_id, user_item_matrix)
        
        if user_profile is None or movie_id not in self.feature_matrix.index:
            explanation['reasons'].append("Based on overall popularity")
            return explanation
        
        movie_features = self.feature_matrix.loc[movie_id]
        
        # Analyze genre preferences
        user_genre_prefs = self.user_profiles.get(user_id, {}).get('genre_preferences', {})
        if user_genre_prefs:
            top_genres = sorted(user_genre_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
            for genre, rating in top_genres:
                if rating > 3.5:  # Good rating threshold
                    explanation['reasons'].append(f"You rated {genre} movies highly (avg: {rating:.1f})")
        
        # Overall similarity
        similarity = self._cosine_similarity_vectors(user_profile, movie_features)
        explanation['content_similarity'] = similarity
        explanation['reasons'].append(f"Content similarity score: {similarity:.2f}")
        
        return explanation
        
    def evaluate_model(self, test_data, user_item_matrix):
        """
        Evaluate content-based model performance
        
        Args:
            test_data: DataFrame with userId, movieId, rating columns
            user_item_matrix: User-item interaction matrix
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        print("📊 Evaluating content-based model...")
        
        # For content-based, we evaluate recommendation quality
        recommendations_made = 0
        total_users = 0
        coverage_movies = set()
        
        for user_id in test_data['userId'].unique():
            if user_id in user_item_matrix.index:
                recommendations = self.recommend_movies(user_id, user_item_matrix, n_recommendations=10)
                if recommendations:
                    recommendations_made += len(recommendations)
                    coverage_movies.update([rec[0] for rec in recommendations])
                total_users += 1
        
        # Calculate metrics
        avg_recommendations_per_user = recommendations_made / total_users if total_users > 0 else 0
        coverage = len(coverage_movies) / len(self.feature_matrix) if len(self.feature_matrix) > 0 else 0
        
        metrics = {
            'avg_recommendations_per_user': avg_recommendations_per_user,
            'coverage': coverage,
            'unique_movies_recommended': len(coverage_movies),
            'total_users_served': total_users,
            'method': 'content_based'
        }
        
        print(f"   • Average recommendations per user: {avg_recommendations_per_user:.1f}")
        print(f"   • Coverage: {coverage:.3f}")
        print(f"   • Users served: {total_users}")
        
        return metrics
        
    def save_model(self, model_name='content_based_model.pkl'):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / model_name
        
        model_data = {
            'feature_weights': self.feature_weights,
            'n_recommendations': self.n_recommendations,
            'movie_features': self.movie_features,
            'user_profiles': self.user_profiles,
            'feature_matrix': self.feature_matrix,
            'movie_similarity_matrix': self.movie_similarity_matrix,
            'genre_vectorizer': self.genre_vectorizer,
            'tag_vectorizer': self.tag_vectorizer,
            'scaler': self.scaler
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Content-based model saved as {model_path}")
        
    def load_model(self, model_name='content_based_model.pkl'):
        """Load a previously saved model"""
        model_path = MODELS_DIR / model_name
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.feature_weights = model_data['feature_weights']
            self.n_recommendations = model_data['n_recommendations']
            self.movie_features = model_data['movie_features']
            self.user_profiles = model_data['user_profiles']
            self.feature_matrix = model_data['feature_matrix']
            self.movie_similarity_matrix = model_data['movie_similarity_matrix']
            self.genre_vectorizer = model_data['genre_vectorizer']
            self.tag_vectorizer = model_data['tag_vectorizer']
            self.scaler = model_data['scaler']
            
            self.is_fitted = True
            print(f"✅ Content-based model loaded from {model_path}")
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Ensure parent directory is in sys.path for module imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from data_processing.data_loader import RecoFlixDataLoader
    from data_processing.preprocessor import RecoFlixPreprocessor
    
    print("🎬 Testing RecoFlix Content-Based Filtering Engine...")
    
    # Load and preprocess data
    loader = RecoFlixDataLoader()
    data = loader.load_all_data()
    
    if data:
        preprocessor = RecoFlixPreprocessor()
        
        # Load processed data
        processed_data = preprocessor.load_processed_data()
        if processed_data is None:
            processed_data = preprocessor.fit_transform(
                data['ratings'], data['movies'], data['tags']
            )
        
        movie_features = processed_data['movie_features']
        user_profiles = processed_data['user_profiles']
        user_item_matrix = processed_data['user_item_matrix']
        test_data = processed_data['test_data']
        
        # Test content-based filtering
        print("\n🔄 Testing Content-Based Filtering...")
        content_filter = RecoFlixContentFilter()
        content_filter.fit(movie_features, user_profiles, user_item_matrix)
        
        # Get sample user and generate recommendations
        sample_user = user_item_matrix.index[0]
        recommendations = content_filter.recommend_movies(sample_user, user_item_matrix, n_recommendations=5)
        
        print(f"\nTop 5 content-based recommendations for User {sample_user}:")
        for i, (movie_id, similarity) in enumerate(recommendations, 1):
            print(f"{i}. Movie {movie_id}: {similarity:.3f}")
        
        # Test movie similarity
        if recommendations:
            sample_movie = recommendations[0][0]
            similar_movies = content_filter.get_similar_movies_content(sample_movie, n_similar=3)
            
            print(f"\nMovies similar to Movie {sample_movie}:")
            for movie_id, similarity in similar_movies:
                print(f"• Movie {movie_id}: {similarity:.3f}")
        
        # Get explanation for first recommendation
        if recommendations:
            explanation = content_filter.explain_recommendation(
                sample_user, recommendations[0][0], user_item_matrix
            )
            print(f"\nWhy Movie {recommendations[0][0]} was recommended:")
            for reason in explanation['reasons']:
                print(f"• {reason}")
        
        # Evaluate model if test data available
        if len(test_data) > 0:
            print("\n📊 Evaluating Content-Based Model...")
            metrics = content_filter.evaluate_model(test_data, user_item_matrix)
        
        print("\n🎉 Content-Based Filtering Engine test completed!")
        
    else:
        print("❌ Could not load data. Check your data files.")