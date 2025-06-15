"""
RecoFlix Collaborative Filtering Engine
Implements user-based and item-based collaborative filtering algorithms
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, RECOMMENDATION_CONFIG

class RecoFlixCollaborativeFilter:
    """
    Collaborative Filtering Recommendation Engine for RecoFlix
    Supports both user-based and item-based collaborative filtering
    """
    
    def __init__(self, method='user_based', n_neighbors=50, n_recommendations=10):
        """
        Initialize the collaborative filter
        
        Args:
            method: 'user_based' or 'item_based'
            n_neighbors: Number of similar users/items to consider
            n_recommendations: Number of recommendations to generate
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.n_recommendations = n_recommendations
        
        # Model components
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        
        # For matrix factorization
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        
        self.is_fitted = False
        
    def fit(self, user_item_matrix):
        """
        Train the collaborative filtering model
        
        Args:
            user_item_matrix: User-item interaction matrix (users as rows, items as columns)
        """
        print(f"🤖 Training RecoFlix {self.method} collaborative filter...")
        
        self.user_item_matrix = user_item_matrix.copy()
        
        # Calculate means for mean-centered recommendations
        self.user_means = self.user_item_matrix.mean(axis=1)
        self.item_means = self.user_item_matrix.mean(axis=0)
        self.global_mean = self.user_item_matrix.values[self.user_item_matrix.values > 0].mean()
        
        # Calculate similarity matrix
        if self.method == 'user_based':
            self._fit_user_based()
        elif self.method == 'item_based':
            self._fit_item_based()
        elif self.method == 'matrix_factorization':
            self._fit_matrix_factorization()
        
        self.is_fitted = True
        print(f"✅ {self.method} model trained successfully!")
        
    def _fit_user_based(self):
        """Train user-based collaborative filtering"""
        print("   • Computing user-user similarity matrix...")
        
        # Create mean-centered matrix for better similarity calculation
        user_matrix_centered = self.user_item_matrix.sub(self.user_means, axis=0)
        
        # Replace NaN with 0 for similarity calculation
        user_matrix_filled = user_matrix_centered.fillna(0)
        
        # Calculate cosine similarity between users
        self.similarity_matrix = pd.DataFrame(
            cosine_similarity(user_matrix_filled),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print(f"   • Similarity matrix shape: {self.similarity_matrix.shape}")
        
    def _fit_item_based(self):
        """Train item-based collaborative filtering"""
        print("   • Computing item-item similarity matrix...")
        
        # Transpose for item-based filtering
        item_matrix = self.user_item_matrix.T
        item_matrix_centered = item_matrix.sub(self.item_means, axis=0)
        item_matrix_filled = item_matrix_centered.fillna(0)
        
        # Calculate cosine similarity between items
        self.similarity_matrix = pd.DataFrame(
            cosine_similarity(item_matrix_filled),
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        print(f"   • Similarity matrix shape: {self.similarity_matrix.shape}")
        
    def _fit_matrix_factorization(self):
        """Train matrix factorization model"""
        print("   • Training matrix factorization model...")
        
        # Replace NaN with 0 for SVD
        matrix_filled = self.user_item_matrix.fillna(0)
        
        # Apply SVD
        self.svd_model = TruncatedSVD(n_components=min(50, min(matrix_filled.shape) - 1))
        self.user_factors = self.svd_model.fit_transform(matrix_filled)
        self.item_factors = self.svd_model.components_.T
        
        print(f"   • User factors shape: {self.user_factors.shape}")
        print(f"   • Item factors shape: {self.item_factors.shape}")
        
    def predict_rating(self, user_id, item_id):
        """
        Predict rating for a specific user-item pair
        
        Args:
            user_id: User ID
            item_id: Item (movie) ID
            
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return self.global_mean
        
        if self.method == 'user_based':
            return self._predict_user_based(user_id, item_id)
        elif self.method == 'item_based':
            return self._predict_item_based(user_id, item_id)
        elif self.method == 'matrix_factorization':
            return self._predict_matrix_factorization(user_id, item_id)
        
    def _predict_user_based(self, user_id, item_id):
        """Predict using user-based collaborative filtering"""
        # Get similar users who have rated this item
        item_ratings = self.user_item_matrix[item_id]
        rated_users = item_ratings[item_ratings > 0].index
        
        if len(rated_users) == 0:
            return self.global_mean
        
        # Get similarities for users who rated this item
        user_similarities = self.similarity_matrix.loc[user_id, rated_users]
        
        # Get top-k similar users
        top_similar_users = user_similarities.nlargest(min(self.n_neighbors, len(user_similarities)))
        
        if len(top_similar_users) == 0 or top_similar_users.sum() == 0:
            return self.user_means[user_id]
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for similar_user, similarity in top_similar_users.items():
            if similarity > 0:
                rating = self.user_item_matrix.loc[similar_user, item_id]
                user_mean = self.user_means[similar_user]
                
                numerator += similarity * (rating - user_mean)
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means[user_id]
        
        predicted_rating = self.user_means[user_id] + (numerator / denominator)
        
        # Clamp to valid rating range
        return max(0.5, min(5.0, predicted_rating))
        
    def _predict_item_based(self, user_id, item_id):
        """Predict using item-based collaborative filtering"""
        # Get items rated by this user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return self.global_mean
        
        # Get similarities for items rated by this user
        item_similarities = self.similarity_matrix.loc[item_id, rated_items]
        
        # Get top-k similar items
        top_similar_items = item_similarities.nlargest(min(self.n_neighbors, len(item_similarities)))
        
        if len(top_similar_items) == 0 or top_similar_items.sum() == 0:
            return self.item_means[item_id]
        
        # Calculate weighted average
        numerator = 0
        denominator = 0
        
        for similar_item, similarity in top_similar_items.items():
            if similarity > 0:
                rating = self.user_item_matrix.loc[user_id, similar_item]
                
                numerator += similarity * rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.item_means[item_id]
        
        predicted_rating = numerator / denominator
        
        # Clamp to valid rating range
        return max(0.5, min(5.0, predicted_rating))
        
    def _predict_matrix_factorization(self, user_id, item_id):
        """Predict using matrix factorization"""
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Clamp to valid rating range
        return max(0.5, min(5.0, predicted_rating))
        
    def recommend_movies(self, user_id, n_recommendations=None, exclude_rated=True):
        """
        Generate movie recommendations for a user
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations (default: self.n_recommendations)
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if n_recommendations is None:
            n_recommendations = self.n_recommendations
        
        if user_id not in self.user_item_matrix.index:
            # Cold start: recommend popular movies
            return self._get_popular_movies(n_recommendations)
        
        # Get all movies
        all_movies = self.user_item_matrix.columns
        
        # Exclude already rated movies if requested
        if exclude_rated:
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_movies = user_ratings[user_ratings == 0].index
            candidate_movies = unrated_movies
        else:
            candidate_movies = all_movies
        
        if len(candidate_movies) == 0:
            return []
        
        # Predict ratings for all candidate movies
        predictions = []
        for movie_id in candidate_movies:
            predicted_rating = self.predict_rating(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top-N
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def _get_popular_movies(self, n_recommendations):
        """Get popular movies for cold start problem"""
        # Calculate movie popularity (number of ratings * average rating)
        movie_counts = (self.user_item_matrix > 0).sum(axis=0)
        movie_averages = self.user_item_matrix.replace(0, np.nan).mean(axis=0)
        movie_popularity = movie_counts * movie_averages
        
        top_movies = movie_popularity.nlargest(n_recommendations)
        
        return [(movie_id, rating) for movie_id, rating in zip(top_movies.index, movie_averages[top_movies.index])]
    
    def get_similar_users(self, user_id, n_similar=10):
        """
        Get most similar users to a given user
        
        Args:
            user_id: Target user ID
            n_similar: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if not self.is_fitted or self.method != 'user_based':
            return []
        
        if user_id not in self.similarity_matrix.index:
            return []
        
        similar_users = self.similarity_matrix.loc[user_id].nlargest(n_similar + 1)[1:]  # Exclude self
        
        return [(user, similarity) for user, similarity in similar_users.items()]
    
    def get_similar_movies(self, movie_id, n_similar=10):
        """
        Get most similar movies to a given movie
        
        Args:
            movie_id: Target movie ID
            n_similar: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted or self.method != 'item_based':
            return []
        
        if movie_id not in self.similarity_matrix.index:
            return []
        
        similar_movies = self.similarity_matrix.loc[movie_id].nlargest(n_similar + 1)[1:]  # Exclude self
        
        return [(movie, similarity) for movie, similarity in similar_movies.items()]
    
    def evaluate_model(self, test_data):
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with userId, movieId, rating columns
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        print("📊 Evaluating model performance...")
        
        predictions = []
        actual_ratings = []
        
        for _, row in test_data.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            try:
                predicted_rating = self.predict_rating(user_id, movie_id)
                predictions.append(predicted_rating)
                actual_ratings.append(actual_rating)
            except:
                continue
        
        if len(predictions) == 0:
            return {'error': 'No valid predictions made'}
        
        # Calculate metrics
        predictions = np.array(predictions)
        actual_ratings = np.array(actual_ratings)
        
        mae = np.mean(np.abs(predictions - actual_ratings))
        rmse = np.sqrt(np.mean((predictions - actual_ratings) ** 2))
        
        # Calculate coverage
        unique_movies_predicted = len(set(test_data['movieId']))
        total_movies = len(self.user_item_matrix.columns)
        coverage = unique_movies_predicted / total_movies
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'n_predictions': len(predictions),
            'method': self.method
        }
        
        print(f"   • MAE: {mae:.3f}")
        print(f"   • RMSE: {rmse:.3f}")
        print(f"   • Coverage: {coverage:.3f}")
        
        return metrics
    
    def save_model(self, model_name=None):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        if model_name is None:
            model_name = f'{self.method}_model.pkl'
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / model_name
        
        model_data = {
            'method': self.method,
            'n_neighbors': self.n_neighbors,
            'n_recommendations': self.n_recommendations,
            'user_item_matrix': self.user_item_matrix,
            'similarity_matrix': self.similarity_matrix,
            'user_means': self.user_means,
            'item_means': self.item_means,
            'global_mean': self.global_mean,
            'svd_model': self.svd_model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved as {model_path}")
    
    def load_model(self, model_name):
        """Load a previously saved model"""
        model_path = MODELS_DIR / model_name
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.method = model_data['method']
            self.n_neighbors = model_data['n_neighbors']
            self.n_recommendations = model_data['n_recommendations']
            self.user_item_matrix = model_data['user_item_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            self.user_means = model_data['user_means']
            self.item_means = model_data['item_means']
            self.global_mean = model_data['global_mean']
            self.svd_model = model_data['svd_model']
            self.user_factors = model_data['user_factors']
            self.item_factors = model_data['item_factors']
            
            self.is_fitted = True
            print(f"✅ Model loaded from {model_path}")
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data_processing.data_loader import RecoFlixDataLoader
    from data_processing.preprocessor import RecoFlixPreprocessor
    
    print("🎬 Testing RecoFlix Collaborative Filtering Engine...")
    
    # Load and preprocess data
    loader = RecoFlixDataLoader()
    data = loader.load_all_data()
    
    if data:
        preprocessor = RecoFlixPreprocessor()
        
        # Try to load processed data, or create it
        processed_data = preprocessor.load_processed_data()
        if processed_data is None:
            processed_data = preprocessor.fit_transform(
                data['ratings'], data['movies'], data['tags']
            )
        
        user_item_matrix = processed_data['user_item_matrix']
        test_data = processed_data['test_data']
        
        # Test user-based collaborative filtering
        print("\n🔄 Testing User-Based Collaborative Filtering...")
        user_cf = RecoFlixCollaborativeFilter(method='user_based', n_neighbors=30)
        user_cf.fit(user_item_matrix)
        
        # Get sample user and generate recommendations
        sample_user = user_item_matrix.index[0]
        recommendations = user_cf.recommend_movies(sample_user, n_recommendations=5)
        
        print(f"\nTop 5 recommendations for User {sample_user}:")
        for i, (movie_id, rating) in enumerate(recommendations, 1):
            print(f"{i}. Movie {movie_id}: {rating:.2f}")
        
        # Test item-based collaborative filtering
        print("\n🔄 Testing Item-Based Collaborative Filtering...")
        item_cf = RecoFlixCollaborativeFilter(method='item_based', n_neighbors=30)
        item_cf.fit(user_item_matrix)
        
        recommendations = item_cf.recommend_movies(sample_user, n_recommendations=5)
        
        print(f"\nTop 5 item-based recommendations for User {sample_user}:")
        for i, (movie_id, rating) in enumerate(recommendations, 1):
            print(f"{i}. Movie {movie_id}: {rating:.2f}")
        
        # Evaluate models if test data available
        if len(test_data) > 0:
            print("\n📊 Evaluating Models...")
            user_metrics = user_cf.evaluate_model(test_data)
            item_metrics = item_cf.evaluate_model(test_data)
            
        print("\n🎉 Collaborative Filtering Engine test completed!")
        
    else:
        print("❌ Could not load data. Check your data files.")