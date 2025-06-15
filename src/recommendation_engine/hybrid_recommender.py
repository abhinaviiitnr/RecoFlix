"""
RecoFlix Hybrid Recommendation System
Combines collaborative filtering and content-based filtering for superior recommendations
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, RECOMMENDATION_CONFIG

# Import our recommendation engines
try:
    from .collaborative_filtering import RecoFlixCollaborativeFilter
    from .content_based import RecoFlixContentFilter
except ImportError:
    from collaborative_filtering import RecoFlixCollaborativeFilter
    from content_based import RecoFlixContentFilter

class RecoFlixHybridRecommender:
    """
    Hybrid Recommendation System for RecoFlix
    Intelligently combines collaborative and content-based filtering
    """
    
    def __init__(self, 
                 hybrid_method='weighted_average',
                 cf_weight=0.6, 
                 cb_weight=0.4,
                 n_recommendations=10,
                 min_cf_neighbors=10):
        """
        Initialize the hybrid recommender
        
        Args:
            hybrid_method: 'weighted_average', 'switching', or 'mixed'
            cf_weight: Weight for collaborative filtering (0-1)
            cb_weight: Weight for content-based filtering (0-1)
            n_recommendations: Number of recommendations to generate
            min_cf_neighbors: Minimum neighbors for CF to be effective
        """
        self.hybrid_method = hybrid_method
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.n_recommendations = n_recommendations
        self.min_cf_neighbors = min_cf_neighbors
        
        # Component recommendation systems
        self.user_cf = None
        self.item_cf = None
        self.mf_cf = None
        self.content_filter = None
        
        # Data storage
        self.user_item_matrix = None
        self.movie_features = None
        self.user_profiles = None
        self.component_performance = {}
        
        self.is_fitted = False
        
    def fit(self, user_item_matrix, movie_features, user_profiles, test_data=None):
        """
        Train the hybrid recommendation system
        
        Args:
            user_item_matrix: User-item interaction matrix
            movie_features: Movie features DataFrame
            user_profiles: User preference profiles
            test_data: Test data for evaluation (optional)
        """
        print("🚀 Training RecoFlix Hybrid Recommendation System...")
        
        self.user_item_matrix = user_item_matrix
        self.movie_features = movie_features
        self.user_profiles = user_profiles
        
        # Train collaborative filtering models
        print("\n🤖 Training Collaborative Filtering Components...")
        
        # User-based CF
        self.user_cf = RecoFlixCollaborativeFilter(
            method='user_based', 
            n_neighbors=30,
            n_recommendations=self.n_recommendations
        )
        self.user_cf.fit(user_item_matrix)
        
        # Item-based CF
        self.item_cf = RecoFlixCollaborativeFilter(
            method='item_based', 
            n_neighbors=30,
            n_recommendations=self.n_recommendations
        )
        self.item_cf.fit(user_item_matrix)
        
        # Matrix factorization CF
        self.mf_cf = RecoFlixCollaborativeFilter(
            method='matrix_factorization',
            n_recommendations=self.n_recommendations
        )
        self.mf_cf.fit(user_item_matrix)
        
        # Train content-based filtering
        print("\n🎬 Training Content-Based Filtering Component...")
        self.content_filter = RecoFlixContentFilter(n_recommendations=self.n_recommendations)
        self.content_filter.fit(movie_features, user_profiles, user_item_matrix)
        
        # Evaluate individual components if test data provided
        if test_data is not None and len(test_data) > 0:
            print("\n📊 Evaluating Individual Components...")
            self.component_performance = self._evaluate_components(test_data)
        else:
            self.component_performance = {}
        
        self.is_fitted = True
        print("\n✅ Hybrid Recommendation System trained successfully!")
        
    def recommend_movies(self, user_id, n_recommendations=None, explanation=False):
        """
        Generate hybrid movie recommendations for a user
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations (default: self.n_recommendations)
            explanation: Whether to include explanation for recommendations
            
        Returns:
            List of (movie_id, score, explanation) tuples if explanation=True
            List of (movie_id, score) tuples if explanation=False
        """
        if not self.is_fitted:
            raise ValueError("Hybrid system not fitted. Call fit() first.")
        
        if n_recommendations is None:
            n_recommendations = self.n_recommendations
        
        # Determine the best approach for this user
        recommendation_strategy = self._determine_strategy(user_id)
        
        if self.hybrid_method == 'weighted_average':
            recommendations = self._weighted_average_recommendations(user_id, n_recommendations)
        elif self.hybrid_method == 'switching':
            recommendations = self._switching_recommendations(user_id, n_recommendations, recommendation_strategy)
        elif self.hybrid_method == 'mixed':
            recommendations = self._mixed_recommendations(user_id, n_recommendations)
        else:
            recommendations = self._weighted_average_recommendations(user_id, n_recommendations)
        
        # Add explanations if requested
        if explanation:
            explained_recommendations = []
            for movie_id, score in recommendations:
                exp = self._explain_recommendation(user_id, movie_id, recommendation_strategy)
                explained_recommendations.append((movie_id, score, exp))
            return explained_recommendations
        
        return recommendations
        
    def _determine_strategy(self, user_id):
        """
        Determine the best recommendation strategy for a user
        
        Args:
            user_id: Target user ID
            
        Returns:
            Dictionary with strategy information
        """
        strategy = {
            'user_id': user_id,
            'is_new_user': user_id not in self.user_item_matrix.index,
            'cf_viable': False,
            'cb_viable': True,  # Content-based is always viable
            'preferred_method': 'content_based'
        }
        
        if not strategy['is_new_user']:
            user_ratings = self.user_item_matrix.loc[user_id]
            num_ratings = (user_ratings > 0).sum()
            
            # CF is viable if user has enough ratings
            strategy['cf_viable'] = num_ratings >= self.min_cf_neighbors
            strategy['num_ratings'] = num_ratings
            
            # Determine preferred method based on user activity
            if num_ratings >= 20:  # Active user - CF likely better
                strategy['preferred_method'] = 'collaborative'
            elif num_ratings >= 10:  # Moderate user - hybrid
                strategy['preferred_method'] = 'hybrid'
            else:  # Sparse user - content-based
                strategy['preferred_method'] = 'content_based'
        
        return strategy
        
    def _weighted_average_recommendations(self, user_id, n_recommendations):
        """Generate recommendations using weighted average of CF and CB"""
        
        # Get CF recommendations (try multiple methods)
        cf_recommendations = {}
        
        # User-based CF
        try:
            user_cf_recs = self.user_cf.recommend_movies(user_id, n_recommendations * 2)
            for movie_id, score in user_cf_recs:
                cf_recommendations[movie_id] = cf_recommendations.get(movie_id, 0) + score * 0.4
        except:
            pass
        
        # Item-based CF
        try:
            item_cf_recs = self.item_cf.recommend_movies(user_id, n_recommendations * 2)
            for movie_id, score in item_cf_recs:
                cf_recommendations[movie_id] = cf_recommendations.get(movie_id, 0) + score * 0.4
        except:
            pass
        
        # Matrix Factorization CF
        try:
            mf_cf_recs = self.mf_cf.recommend_movies(user_id, n_recommendations * 2)
            for movie_id, score in mf_cf_recs:
                cf_recommendations[movie_id] = cf_recommendations.get(movie_id, 0) + score * 0.2
        except:
            pass
        
        # Get content-based recommendations
        cb_recommendations = {}
        try:
            cb_recs = self.content_filter.recommend_movies(user_id, self.user_item_matrix, n_recommendations * 2)
            for movie_id, score in cb_recs:
                cb_recommendations[movie_id] = score
        except:
            pass
        
        # Combine recommendations with weighted average
        combined_recommendations = {}
        all_movies = set(cf_recommendations.keys()) | set(cb_recommendations.keys())
        
        for movie_id in all_movies:
            cf_score = cf_recommendations.get(movie_id, 0)
            cb_score = cb_recommendations.get(movie_id, 0)
            
            # Normalize scores to 0-1 range for fair combination
            cf_score_norm = min(1.0, max(0.0, cf_score / 5.0)) if cf_score > 0 else 0
            cb_score_norm = min(1.0, max(0.0, cb_score)) if cb_score > 0 else 0
            
            # Calculate weighted average
            combined_score = (cf_score_norm * self.cf_weight + cb_score_norm * self.cb_weight)
            
            if combined_score > 0:
                combined_recommendations[movie_id] = combined_score
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(combined_recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
        
    def _switching_recommendations(self, user_id, n_recommendations, strategy):
        """Generate recommendations by switching between methods based on user characteristics"""
        
        if strategy['preferred_method'] == 'collaborative' and strategy['cf_viable']:
            # Use the best performing CF method
            try:
                return self.item_cf.recommend_movies(user_id, n_recommendations)
            except:
                return self.content_filter.recommend_movies(user_id, self.user_item_matrix, n_recommendations)
        
        elif strategy['preferred_method'] == 'hybrid':
            # Use weighted average for moderate users
            return self._weighted_average_recommendations(user_id, n_recommendations)
        
        else:
            # Use content-based for new or sparse users
            return self.content_filter.recommend_movies(user_id, self.user_item_matrix, n_recommendations)
    
    def _mixed_recommendations(self, user_id, n_recommendations):
        """Generate recommendations by mixing different methods"""
        
        # Allocate recommendations across methods
        cf_count = int(n_recommendations * 0.6)
        cb_count = n_recommendations - cf_count
        
        recommendations = []
        used_movies = set()
        
        # Get CF recommendations
        try:
            cf_recs = self.item_cf.recommend_movies(user_id, cf_count * 2)
            for movie_id, score in cf_recs:
                if movie_id not in used_movies and len([r for r in recommendations if r[0] == movie_id]) == 0:
                    recommendations.append((movie_id, score))
                    used_movies.add(movie_id)
                    if len([r for r in recommendations]) >= cf_count:
                        break
        except:
            pass
        
        # Get content-based recommendations
        try:
            cb_recs = self.content_filter.recommend_movies(user_id, self.user_item_matrix, cb_count * 2)
            for movie_id, score in cb_recs:
                if movie_id not in used_movies:
                    recommendations.append((movie_id, score))
                    used_movies.add(movie_id)
                    if len(recommendations) >= n_recommendations:
                        break
        except:
            pass
        
        # Fill remaining slots if needed
        if len(recommendations) < n_recommendations:
            try:
                additional_recs = self.user_cf.recommend_movies(user_id, n_recommendations)
                for movie_id, score in additional_recs:
                    if movie_id not in used_movies:
                        recommendations.append((movie_id, score))
                        if len(recommendations) >= n_recommendations:
                            break
            except:
                pass
        
        return recommendations[:n_recommendations]
        
    def _explain_recommendation(self, user_id, movie_id, strategy):
        """Generate explanation for why a movie was recommended"""
        
        explanation = {
            'movie_id': movie_id,
            'user_id': user_id,
            'primary_reason': '',
            'secondary_reasons': [],
            'confidence': 0.0
        }
        
        if strategy['is_new_user']:
            explanation['primary_reason'] = "Popular movie for new users"
            explanation['confidence'] = 0.6
        elif strategy['preferred_method'] == 'collaborative':
            explanation['primary_reason'] = "Users with similar preferences also liked this movie"
            explanation['secondary_reasons'].append("Based on collaborative filtering")
            explanation['confidence'] = 0.8
        elif strategy['preferred_method'] == 'content_based':
            explanation['primary_reason'] = "Matches your movie preferences and viewing history"
            explanation['secondary_reasons'].append("Based on movie features and your taste profile")
            explanation['confidence'] = 0.7
        else:
            explanation['primary_reason'] = "Recommended by our hybrid algorithm"
            explanation['secondary_reasons'].append("Combines multiple recommendation techniques")
            explanation['confidence'] = 0.75
            
        return explanation
        
    def get_user_insights(self, user_id):
        """
        Get insights about a user's preferences and recommendation strategy
        
        Args:
            user_id: Target user ID
            
        Returns:
            Dictionary with user insights
        """
        if not self.is_fitted:
            return {}
        
        insights = {
            'user_id': user_id,
            'strategy': self._determine_strategy(user_id),
            'preferences': {},
            'similar_users': [],
            'recommendation_strength': {}
        }
        
        # Get user preferences from profiles
        if user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
            insights['preferences'] = {
                'avg_rating': user_profile.get('avg_rating', 0),
                'num_ratings': user_profile.get('num_ratings', 0),
                'favorite_genre': user_profile.get('favorite_genre', 'Unknown'),
                'genre_preferences': user_profile.get('genre_preferences', {})
            }
        
        # Get similar users if CF is viable
        if insights['strategy']['cf_viable']:
            try:
                similar_users = self.user_cf.get_similar_users(user_id, n_similar=5)
                insights['similar_users'] = similar_users
            except:
                pass
        
        # Assess recommendation method strengths for this user
        insights['recommendation_strength'] = {
            'collaborative_filtering': 0.8 if insights['strategy']['cf_viable'] else 0.3,
            'content_based': 0.7,  # Always reasonably strong
            'hybrid': 0.9 if insights['strategy']['cf_viable'] else 0.6
        }
        
        return insights
        
    def _evaluate_components(self, test_data):
        """Evaluate individual recommendation components"""
        
        performance = {}
        
        # Evaluate collaborative filtering methods
        try:
            user_cf_metrics = self.user_cf.evaluate_model(test_data)
            performance['user_based_cf'] = user_cf_metrics
        except:
            performance['user_based_cf'] = {'error': 'Evaluation failed'}
        
        try:
            item_cf_metrics = self.item_cf.evaluate_model(test_data)
            performance['item_based_cf'] = item_cf_metrics
        except:
            performance['item_based_cf'] = {'error': 'Evaluation failed'}
        
        # Evaluate content-based filtering
        try:
            cb_metrics = self.content_filter.evaluate_model(test_data, self.user_item_matrix)
            performance['content_based'] = cb_metrics
        except:
            performance['content_based'] = {'error': 'Evaluation failed'}
        
        return performance
        
    def evaluate_hybrid_model(self, test_data, sample_size=100):
        """
        Evaluate the hybrid recommendation system
        
        Args:
            test_data: Test dataset
            sample_size: Number of users to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Hybrid system not fitted. Call fit() first.")
        
        print("📊 Evaluating Hybrid Recommendation System...")
        
        # Sample users for evaluation
        test_users = test_data['userId'].unique()
        if len(test_users) > sample_size:
            eval_users = np.random.choice(test_users, sample_size, replace=False)
        else:
            eval_users = test_users
        
        # Metrics tracking
        total_recommendations = 0
        successful_recommendations = 0
        strategy_usage = {'collaborative': 0, 'content_based': 0, 'hybrid': 0}
        user_satisfaction_scores = []
        
        for user_id in eval_users:
            try:
                strategy = self._determine_strategy(user_id)
                strategy_usage[strategy['preferred_method']] += 1
                
                recommendations = self.recommend_movies(user_id, n_recommendations=10)
                
                if recommendations:
                    total_recommendations += len(recommendations)
                    successful_recommendations += 1
                    
                    # Calculate satisfaction score (simplified)
                    avg_score = np.mean([score for _, score in recommendations])
                    user_satisfaction_scores.append(avg_score)
                    
            except Exception as e:
                continue
        
        # Calculate overall metrics
        success_rate = successful_recommendations / len(eval_users) if len(eval_users) > 0 else 0
        avg_satisfaction = np.mean(user_satisfaction_scores) if user_satisfaction_scores else 0
        
        metrics = {
            'success_rate': success_rate,
            'avg_satisfaction_score': avg_satisfaction,
            'total_users_evaluated': len(eval_users),
            'successful_recommendations': successful_recommendations,
            'avg_recommendations_per_user': total_recommendations / successful_recommendations if successful_recommendations > 0 else 0,
            'strategy_distribution': strategy_usage,
            'component_performance': self.component_performance,
            'hybrid_method': self.hybrid_method
        }
        
        print(f"   • Success rate: {success_rate:.3f}")
        print(f"   • Average satisfaction: {avg_satisfaction:.3f}")
        print(f"   • Strategy distribution: {strategy_usage}")
        
        return metrics
        
    def save_model(self, model_name='hybrid_recommender.pkl'):
        """Save the trained hybrid model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / model_name
        
        # Save component models separately for better organization
        self.user_cf.save_model('hybrid_user_cf_model.pkl')
        self.item_cf.save_model('hybrid_item_cf_model.pkl')
        self.mf_cf.save_model('hybrid_mf_cf_model.pkl')
        self.content_filter.save_model('hybrid_content_model.pkl')
        
        # Save hybrid-specific data
        model_data = {
            'hybrid_method': self.hybrid_method,
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'n_recommendations': self.n_recommendations,
            'min_cf_neighbors': self.min_cf_neighbors,
            'component_performance': getattr(self, 'component_performance', {}),
            'user_item_matrix': self.user_item_matrix,
            'movie_features': self.movie_features,
            'user_profiles': self.user_profiles
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Hybrid model saved as {model_path}")
        
    def load_model(self, model_name='hybrid_recommender.pkl'):
        """Load a previously saved hybrid model"""
        model_path = MODELS_DIR / model_name
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load hybrid configuration
            self.hybrid_method = model_data['hybrid_method']
            self.cf_weight = model_data['cf_weight']
            self.cb_weight = model_data['cb_weight']
            self.n_recommendations = model_data['n_recommendations']
            self.min_cf_neighbors = model_data['min_cf_neighbors']
            self.component_performance = model_data.get('component_performance', {})
            self.user_item_matrix = model_data['user_item_matrix']
            self.movie_features = model_data['movie_features']
            self.user_profiles = model_data['user_profiles']
            
            # Load component models
            self.user_cf = RecoFlixCollaborativeFilter(method='user_based')
            self.user_cf.load_model('hybrid_user_cf_model.pkl')
            
            self.item_cf = RecoFlixCollaborativeFilter(method='item_based')
            self.item_cf.load_model('hybrid_item_cf_model.pkl')
            
            self.mf_cf = RecoFlixCollaborativeFilter(method='matrix_factorization')
            self.mf_cf.load_model('hybrid_mf_cf_model.pkl')
            
            self.content_filter = RecoFlixContentFilter()
            self.content_filter.load_model('hybrid_content_model.pkl')
            
            self.is_fitted = True
            print(f"✅ Hybrid model loaded from {model_path}")
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from data_processing.data_loader import RecoFlixDataLoader
    except ImportError:
        from ..data_processing.data_loader import RecoFlixDataLoader
    try:
        from data_processing.preprocessor import RecoFlixPreprocessor
    except ImportError:
        from ..data_processing.preprocessor import RecoFlixPreprocessor
    
    print("🚀 Testing RecoFlix Hybrid Recommendation System...")
    
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
        
        user_item_matrix = processed_data['user_item_matrix']
        movie_features = processed_data['movie_features']
        user_profiles = processed_data['user_profiles']
        test_data = processed_data['test_data']
        
        # Test hybrid recommendation system
        print("\n🔄 Testing Hybrid Recommendation System...")
        
        # Test weighted average method
        hybrid_system = RecoFlixHybridRecommender(
            hybrid_method='weighted_average',
            cf_weight=0.6,
            cb_weight=0.4
        )
        
        hybrid_system.fit(user_item_matrix, movie_features, user_profiles, test_data)
        
        # Get sample user and generate recommendations
        sample_user = user_item_matrix.index[0]
        
        print(f"\n🎯 Hybrid Recommendations for User {sample_user}:")
        recommendations = hybrid_system.recommend_movies(sample_user, n_recommendations=5, explanation=True)
        
        for i, (movie_id, score, explanation) in enumerate(recommendations, 1):
            print(f"{i}. Movie {movie_id}: {score:.3f}")
            print(f"   Why: {explanation['primary_reason']}")
        
        # Get user insights
        insights = hybrid_system.get_user_insights(sample_user)
        print(f"\n👤 User {sample_user} Insights:")
        print(f"• Strategy: {insights['strategy']['preferred_method']}")
        print(f"• CF Viable: {insights['strategy']['cf_viable']}")
        if 'favorite_genre' in insights['preferences']:
            print(f"• Favorite Genre: {insights['preferences']['favorite_genre']}")
        
        # Evaluate hybrid system
        if len(test_data) > 0:
            print("\n📊 Evaluating Hybrid System...")
            metrics = hybrid_system.evaluate_hybrid_model(test_data, sample_size=50)
        
        print("\n🎉 Hybrid Recommendation System test completed!")
        print("🎬 RecoFlix is ready for production!")
        
    else:
        print("❌ Could not load data. Check your data files.")