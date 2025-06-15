"""
RecoFlix Data Loader
Handles loading and initial validation of movie datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_FILES, RECOMMENDATION_CONFIG

class RecoFlixDataLoader:
    """
    Data loading and validation class for RecoFlix
    """
    
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.tags_df = None
        self.links_df = None
        self.is_loaded = False
        
    def load_all_data(self, data_path=None):
        """
        Load all CSV files into pandas DataFrames
        
        Args:
            data_path (str): Path to data directory (optional)
        
        Returns:
            dict: Dictionary containing all loaded DataFrames
        """
        print("🎬 Loading RecoFlix datasets...")
        
        try:
            # Use custom path or default config path
            if data_path:
                base_path = Path(data_path)
                files = {
                    'ratings': base_path / 'ratings.csv',
                    'movies': base_path / 'movies.csv',
                    'tags': base_path / 'tags.csv',
                    'links': base_path / 'links.csv'
                }
            else:
                files = DATA_FILES
            
            # Load each dataset
            self.ratings_df = pd.read_csv(files['ratings'])
            self.movies_df = pd.read_csv(files['movies'])
            self.tags_df = pd.read_csv(files['tags'])
            self.links_df = pd.read_csv(files['links'])
            
            print(f"✅ Ratings loaded: {len(self.ratings_df):,} interactions")
            print(f"✅ Movies loaded: {len(self.movies_df):,} movies")
            print(f"✅ Tags loaded: {len(self.tags_df):,} tags")
            print(f"✅ Links loaded: {len(self.links_df):,} movie links")
            
            self.is_loaded = True
            return self.get_all_dataframes()
            
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find data files. {e}")
            print("💡 Make sure to place your CSV files in the data/raw/ directory")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def get_all_dataframes(self):
        """Return dictionary of all loaded DataFrames"""
        if not self.is_loaded:
            print("⚠️ Data not loaded yet. Call load_all_data() first.")
            return None
            
        return {
            'ratings': self.ratings_df,
            'movies': self.movies_df,
            'tags': self.tags_df,
            'links': self.links_df
        }
    
    def validate_data(self):
        """
        Validate loaded data for recommendation system requirements
        
        Returns:
            dict: Validation results and statistics
        """
        if not self.is_loaded:
            return {"error": "Data not loaded"}
        
        print("\n🔍 Validating RecoFlix data...")
        
        validation_results = {}
        
        # Basic data validation
        validation_results['basic_stats'] = {
            'total_ratings': len(self.ratings_df),
            'unique_users': self.ratings_df['userId'].nunique(),
            'unique_movies': self.ratings_df['movieId'].nunique(),
            'rating_range': (self.ratings_df['rating'].min(), self.ratings_df['rating'].max()),
            'missing_values': {
                'ratings': self.ratings_df.isnull().sum().sum(),
                'movies': self.movies_df.isnull().sum().sum()
            }
        }
        
        # Recommendation system readiness
        min_ratings_user = RECOMMENDATION_CONFIG['min_ratings_per_user']
        min_ratings_movie = RECOMMENDATION_CONFIG['min_ratings_per_movie']
        
        user_counts = self.ratings_df['userId'].value_counts()
        movie_counts = self.ratings_df['movieId'].value_counts()
        
        validation_results['recommendation_readiness'] = {
            'users_with_min_ratings': (user_counts >= min_ratings_user).sum(),
            'movies_with_min_ratings': (movie_counts >= min_ratings_movie).sum(),
            'data_sparsity': len(self.ratings_df) / (self.ratings_df['userId'].nunique() * self.ratings_df['movieId'].nunique()) * 100
        }
        
        # Content-based filtering readiness
        genres_available = (self.movies_df['genres'] != '(no genres listed)').sum()
        validation_results['content_readiness'] = {
            'movies_with_genres': genres_available,
            'genre_coverage': genres_available / len(self.movies_df) * 100,
            'total_tags': len(self.tags_df)
        }
        
        # Print validation summary
        stats = validation_results['basic_stats']
        reco_stats = validation_results['recommendation_readiness']
        content_stats = validation_results['content_readiness']
        
        print(f"📊 Basic Statistics:")
        print(f"   • Total ratings: {stats['total_ratings']:,}")
        print(f"   • Unique users: {stats['unique_users']:,}")
        print(f"   • Unique movies: {stats['unique_movies']:,}")
        print(f"   • Rating range: {stats['rating_range'][0]} - {stats['rating_range'][1]}")
        
        print(f"\n🤖 Recommendation System Readiness:")
        print(f"   • Users with {min_ratings_user}+ ratings: {reco_stats['users_with_min_ratings']:,}")
        print(f"   • Movies with {min_ratings_movie}+ ratings: {reco_stats['movies_with_min_ratings']:,}")
        print(f"   • Data density: {reco_stats['data_sparsity']:.3f}%")
        
        print(f"\n🎬 Content-Based Filtering Readiness:")
        print(f"   • Movies with genres: {content_stats['movies_with_genres']:,} ({content_stats['genre_coverage']:.1f}%)")
        print(f"   • Available tags: {content_stats['total_tags']:,}")
        
        # Validation verdict
        is_ready = (
            reco_stats['users_with_min_ratings'] >= 50 and
            reco_stats['movies_with_min_ratings'] >= 100 and
            content_stats['genre_coverage'] > 80
        )
        
        validation_results['is_ready_for_recommendations'] = is_ready
        
        if is_ready:
            print(f"\n✅ Dataset is READY for RecoFlix recommendation system!")
        else:
            print(f"\n⚠️ Dataset needs preprocessing for optimal performance")
        
        return validation_results
    
    def get_data_overview(self):
        """Get comprehensive data overview for exploration"""
        if not self.is_loaded:
            return None
            
        overview = {}
        
        # Rating distribution
        overview['rating_distribution'] = self.ratings_df['rating'].value_counts().sort_index()
        
        # User activity distribution
        overview['user_activity'] = self.ratings_df['userId'].value_counts().describe()
        
        # Movie popularity distribution  
        overview['movie_popularity'] = self.ratings_df['movieId'].value_counts().describe()
        
        # Genre analysis
        all_genres = []
        for genres in self.movies_df['genres'].dropna():
            if genres != '(no genres listed)':
                all_genres.extend(genres.split('|'))
        overview['genre_distribution'] = pd.Series(all_genres).value_counts()
        
        # Temporal analysis
        if 'timestamp' in self.ratings_df.columns:
            self.ratings_df['datetime'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
            overview['temporal_range'] = {
                'start_date': self.ratings_df['datetime'].min(),
                'end_date': self.ratings_df['datetime'].max(),
                'total_days': (self.ratings_df['datetime'].max() - self.ratings_df['datetime'].min()).days
            }
        
        return overview

# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    loader = RecoFlixDataLoader()
    
    # Load data (you'll need to put your CSV files in data/raw/ directory)
    data = loader.load_all_data()
    
    if data:
        # Validate data
        validation = loader.validate_data()
        
        # Get overview
        overview = loader.get_data_overview()
        
        print("\n🎉 RecoFlix Data Loader test completed successfully!")
        print("📋 Next step: Run data preprocessing")
    else:
        print("❌ Please ensure your CSV files are in the correct directory")