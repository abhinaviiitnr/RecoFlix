"""
RecoFlix - Movie Recommendation System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from config import APP_CONFIG, COLORS, PROCESSED_DATA_DIR
from src.data_processing.data_loader import RecoFlixDataLoader
from src.data_processing.preprocessor import RecoFlixPreprocessor
from src.recommendation_engine.hybrid_recommender import RecoFlixHybridRecommender

# Configure Streamlit page
st.set_page_config(
    page_title=APP_CONFIG['page_title'],
    page_icon=APP_CONFIG['page_icon'],
    layout=APP_CONFIG['layout'],
    initial_sidebar_state=APP_CONFIG['initial_sidebar_state']
)

# Custom CSS for Netflix-like theme
st.markdown("""
<style>
    .main {
        background-color: #141414;
        color: #FFFFFF;
    }
    
    .stApp {
        background-color: #141414;
    }
    
    .css-1d391kg {
        background-color: #141414;
    }
    
    .stSelectbox > div > div {
        background-color: #333333;
        color: #FFFFFF;
    }
    
    .stButton > button {
        background-color: #E50914;
        color: #FFFFFF;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #F40612;
        color: #FFFFFF;
    }
    
    .movie-card {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #444444;
    }
    
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333333;
    }
    
    .recommendation-score {
        color: #FFA500;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .genre-tag {
        background-color: #E50914;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8em;
        margin: 0.1rem;
        display: inline-block;
    }
    
    h1, h2, h3 {
        color: #FFFFFF;
    }
    
    .stMarkdown {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_recoflix_data():
    """Load and cache RecoFlix data"""
    try:
        loader = RecoFlixDataLoader()
        data = loader.load_all_data()
        
        preprocessor = RecoFlixPreprocessor()
        processed_data = preprocessor.load_processed_data()
        
        if processed_data is None:
            with st.spinner("Processing data for the first time..."):
                processed_data = preprocessor.fit_transform(
                    data['ratings'], data['movies'], data['tags']
                )
        
        return data, processed_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_recommendation_system(_processed_data):
    """Load and cache the hybrid recommendation system"""
    try:
        with st.spinner("Loading RecoFlix recommendation engine..."):
            hybrid_system = RecoFlixHybridRecommender(
                hybrid_method='weighted_average',
                cf_weight=0.6,
                cb_weight=0.4
            )
            
            hybrid_system.fit(
                _processed_data['user_item_matrix'],
                _processed_data['movie_features'],
                _processed_data['user_profiles'],
                _processed_data['test_data']
            )
            
            return hybrid_system
    except Exception as e:
        st.error(f"Error loading recommendation system: {e}")
        return None

def get_movie_info(movie_id, movies_df):
    """Get movie information by ID"""
    movie_info = movies_df[movies_df['movieId'] == movie_id]
    if len(movie_info) > 0:
        movie = movie_info.iloc[0]
        return {
            'title': movie['title'],
            'genres': movie['genres'],
            'movieId': movie['movieId']
        }
    return {'title': f'Movie {movie_id}', 'genres': 'Unknown', 'movieId': movie_id}

def format_genres(genres_str):
    """Format genres as HTML tags"""
    if pd.isna(genres_str) or genres_str == '(no genres listed)':
        return '<span class="genre-tag">Unknown</span>'
    
    genres = genres_str.split('|')[:3]  # Show max 3 genres
    genre_tags = [f'<span class="genre-tag">{genre}</span>' for genre in genres]
    return ' '.join(genre_tags)

def display_movie_recommendations(recommendations, movies_df, title="🎬 Your Recommendations"):
    """Display movie recommendations in a nice format"""
    st.markdown(f"### {title}")
    
    if not recommendations:
        st.warning("No recommendations available. Try selecting a different user or check the system.")
        return
    
    # Create columns for movie cards
    cols = st.columns(2)
    
    for i, (movie_id, score) in enumerate(recommendations[:10]):
        movie_info = get_movie_info(movie_id, movies_df)
        
        with cols[i % 2]:
            st.markdown(f"""
            <div class="movie-card">
                <h4>{movie_info['title']}</h4>
                <p>{format_genres(movie_info['genres'])}</p>
                <p class="recommendation-score">⭐ Score: {score:.2f}</p>
                <p><small>Movie ID: {movie_id}</small></p>
            </div>
            """, unsafe_allow_html=True)

def display_user_insights(user_insights):
    """Display user insights and preferences"""
    st.markdown("### 👤 Your Movie Profile")
    
    if not user_insights or 'preferences' not in user_insights:
        st.info("No user profile available.")
        return
    
    prefs = user_insights['preferences']
    strategy = user_insights['strategy']
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{prefs.get('num_ratings', 0)}</h3>
            <p>Movies Rated</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_rating = prefs.get('avg_rating', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_rating:.1f}</h3>
            <p>Average Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        fav_genre = prefs.get('favorite_genre', 'Unknown')
        st.markdown(f"""
        <div class="metric-card">
            <h3>{fav_genre}</h3>
            <p>Favorite Genre</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        user_type = strategy.get('preferred_method', 'Unknown').replace('_', ' ').title()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{user_type}</h3>
            <p>User Type</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Genre preferences chart
    if 'genre_preferences' in prefs and prefs['genre_preferences']:
        st.markdown("#### 🎭 Your Genre Preferences")
        
        genre_prefs = prefs['genre_preferences']
        # Get top 10 genres
        top_genres = dict(sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:10])
        
        if top_genres:
            fig = px.bar(
                x=list(top_genres.keys()),
                y=list(top_genres.values()),
                title="Your Top Genres by Average Rating",
                color=list(top_genres.values()),
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Genres",
                yaxis_title="Average Rating"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main RecoFlix application"""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #E50914; font-size: 3rem; margin-bottom: 0;">🎬 RecoFlix</h1>
        <p style="color: #FFFFFF; font-size: 1.2rem;">Your Personalized Movie Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data, processed_data = load_recoflix_data()
    
    if data is None or processed_data is None:
        st.error("Failed to load RecoFlix data. Please check your data files.")
        st.stop()
    
    # Load recommendation system
    hybrid_system = load_recommendation_system(processed_data)
    
    if hybrid_system is None:
        st.error("Failed to load recommendation system.")
        st.stop()
    
    # Sidebar for user selection and controls
    st.sidebar.markdown("## 🎮 RecoFlix Controls")
    st.sidebar.markdown("---")
    
    # User type selection
    user_type = st.sidebar.radio(
        "🆕 User Type",
        options=["Existing User", "New User (Cold Start)"],
        help="Choose to test as existing user or simulate new user experience"
    )
    
    if user_type == "Existing User":
        # User selection for existing users
        available_users = processed_data['user_item_matrix'].index.tolist()
        selected_user = st.sidebar.selectbox(
            "👤 Select User ID",
            options=available_users,
            index=0,
            help="Choose a user to get personalized recommendations"
        )
        is_new_user = False
    else:
        # New user simulation
        st.sidebar.markdown("### 🆕 New User Simulation")
        st.sidebar.info("Testing how RecoFlix handles users with no rating history")
        
        # Create a fake new user ID that doesn't exist
        max_user_id = max(processed_data['user_item_matrix'].index)
        selected_user = max_user_id + 1000  # Use a clearly new user ID
        is_new_user = True
        
        st.sidebar.markdown(f"**Simulated New User ID:** {selected_user}")
        
        # Optional: Let new user select preferred genres
        st.sidebar.markdown("#### 🎭 Your Genre Preferences (Optional)")
        all_genres = set()
        for genres in data['movies']['genres'].dropna():
            if genres != '(no genres listed)':
                all_genres.update(genres.split('|'))
        
        preferred_genres = st.sidebar.multiselect(
            "Select genres you like:",
            options=sorted(list(all_genres)),
            default=[],
            help="This will influence your recommendations"
        )
    
    # Recommendation settings
    st.sidebar.markdown("### ⚙️ Recommendation Settings")
    
    num_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        help="How many movie recommendations to show"
    )
    
    recommendation_method = st.sidebar.selectbox(
        "Recommendation Method",
        options=['weighted_average', 'switching', 'mixed'],
        index=0,
        help="Choose the hybrid recommendation strategy"
    )
    
    show_explanations = st.sidebar.checkbox(
        "Show Explanations",
        value=True,
        help="Include explanations for why movies were recommended"
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Recommendations", "👤 User Profile", "📊 Analytics", "🔍 Explore"])
    
    with tab1:
        st.markdown("## 🎯 Personalized Movie Recommendations")
        
        # Update hybrid system method if changed
        hybrid_system.hybrid_method = recommendation_method
        
        # Generate recommendations
        with st.spinner("Generating your personalized recommendations..."):
            try:
                if is_new_user:
                    # Handle new user recommendations
                    st.info(f"🆕 Generating recommendations for new user (ID: {selected_user}) using content-based filtering")
                    
                    # For new users, we'll use content-based recommendations with genre preferences
                    if 'preferred_genres' in locals() and preferred_genres:
                        # Filter movies by preferred genres
                        genre_filtered_movies = data['movies'][
                            data['movies']['genres'].str.contains('|'.join(preferred_genres), case=False, na=False)
                        ]
                        
                        if len(genre_filtered_movies) > 0:
                            # Get popular movies from preferred genres
                            popular_movies_in_genres = []
                            for _, movie in genre_filtered_movies.iterrows():
                                movie_id = movie['movieId']
                                if movie_id in processed_data['user_item_matrix'].columns:
                                    # Calculate popularity score
                                    movie_ratings = processed_data['user_item_matrix'][movie_id]
                                    num_ratings = (movie_ratings > 0).sum()
                                    avg_rating = movie_ratings[movie_ratings > 0].mean() if num_ratings > 0 else 0
                                    
                                    if num_ratings >= 10:  # Only consider movies with sufficient ratings
                                        popularity_score = num_ratings * avg_rating / 100  # Normalize
                                        popular_movies_in_genres.append((movie_id, popularity_score))
                            
                            # Sort by popularity and get top recommendations
                            popular_movies_in_genres.sort(key=lambda x: x[1], reverse=True)
                            recommendations = popular_movies_in_genres[:num_recommendations]
                            
                            # Add explanations for new user
                            if show_explanations:
                                explained_recommendations = []
                                for movie_id, score in recommendations:
                                    explanation = {
                                        'primary_reason': f"Popular movie in your preferred genres: {', '.join(preferred_genres)}",
                                        'confidence': 0.7
                                    }
                                    explained_recommendations.append((movie_id, score, explanation))
                                recommendations = explained_recommendations
                        else:
                            st.warning("No movies found for your preferred genres. Showing general popular movies.")
                            recommendations = []
                    
                    if not recommendations or ('preferred_genres' in locals() and not preferred_genres):
                        # Fallback to general popular movies for new users
                        st.info("🔥 Showing trending movies popular among all users")
                        
                        # Get most popular movies overall
                        movie_popularity = []
                        for movie_id in processed_data['user_item_matrix'].columns:
                            movie_ratings = processed_data['user_item_matrix'][movie_id]
                            num_ratings = (movie_ratings > 0).sum()
                            avg_rating = movie_ratings[movie_ratings > 0].mean() if num_ratings > 0 else 0
                            
                            if num_ratings >= 20:  # Popular movies threshold
                                popularity_score = (num_ratings * avg_rating) / 1000  # Normalize
                                movie_popularity.append((movie_id, popularity_score))
                        
                        # Sort and get top movies
                        movie_popularity.sort(key=lambda x: x[1], reverse=True)
                        recommendations = movie_popularity[:num_recommendations]
                        
                        # Add explanations
                        if show_explanations:
                            explained_recommendations = []
                            for movie_id, score in recommendations:
                                explanation = {
                                    'primary_reason': "Popular movie recommended for new users",
                                    'confidence': 0.6
                                }
                                explained_recommendations.append((movie_id, score, explanation))
                            recommendations = explained_recommendations
                    
                    # Display new user recommendations
                    if recommendations:
                        if show_explanations:
                            st.markdown(f"### 🎬 Top {len(recommendations)} Movies for New User")
                            st.markdown("*These recommendations are based on overall popularity and your genre preferences*")
                            
                            for i, (movie_id, score, explanation) in enumerate(recommendations, 1):
                                movie_info = get_movie_info(movie_id, data['movies'])
                                
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"""
                                        <div class="movie-card">
                                            <h4>{i}. {movie_info['title']}</h4>
                                            <p>{format_genres(movie_info['genres'])}</p>
                                            <p><strong>Why recommended:</strong> {explanation['primary_reason']}</p>
                                            <p><small>Confidence: {explanation['confidence']:.1%}</small></p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h3 class="recommendation-score">{score:.3f}</h3>
                                            <p>Popularity Score</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                        else:
                            display_movie_recommendations(recommendations, data['movies'], "🆕 New User Recommendations")
                    else:
                        st.error("Unable to generate recommendations. Please try different genre preferences.")
                
                else:
                    # Existing user recommendations (original logic)
                    if show_explanations:
                        recommendations = hybrid_system.recommend_movies(
                            selected_user, 
                            n_recommendations=num_recommendations,
                            explanation=True
                        )
                        
                        # Display with explanations
                        st.markdown(f"### 🎬 Top {len(recommendations)} Movies for User {selected_user}")
                        
                        for i, (movie_id, score, explanation) in enumerate(recommendations, 1):
                            movie_info = get_movie_info(movie_id, data['movies'])
                            
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="movie-card">
                                        <h4>{i}. {movie_info['title']}</h4>
                                        <p>{format_genres(movie_info['genres'])}</p>
                                        <p><strong>Why recommended:</strong> {explanation['primary_reason']}</p>
                                        <p><small>Confidence: {explanation['confidence']:.1%}</small></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h3 class="recommendation-score">{score:.2f}</h3>
                                        <p>RecoFlix Score</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        recommendations = hybrid_system.recommend_movies(
                            selected_user, 
                            n_recommendations=num_recommendations,
                            explanation=False
                        )
                        display_movie_recommendations(recommendations, data['movies'])
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                # Debug info for troubleshooting
                st.error(f"User ID: {selected_user}, Is New User: {is_new_user}")
    
    with tab2:
        st.markdown("## 👤 User Profile & Insights")
        
        if is_new_user:
            # New user profile display
            st.markdown("### 🆕 New User Profile")
            st.info("This is a simulated new user with no rating history")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>0</h3>
                    <p>Movies Rated</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>N/A</h3>
                    <p>Average Rating</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                fav_genre = preferred_genres[0] if 'preferred_genres' in locals() and preferred_genres else 'None Selected'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{fav_genre}</h3>
                    <p>Preferred Genre</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>New User</h3>
                    <p>User Type</p>
                </div>
                """, unsafe_allow_html=True)
            
            if 'preferred_genres' in locals() and preferred_genres:
                st.markdown("#### 🎭 Your Selected Genre Preferences")
                for i, genre in enumerate(preferred_genres, 1):
                    st.write(f"{i}. **{genre}** (Selected preference)")
            else:
                st.markdown("#### 🎭 No Genre Preferences Selected")
                st.info("Select genre preferences in the sidebar to get more personalized recommendations!")
            
            st.markdown("#### 🔍 New User Recommendation Strategy")
            st.markdown("""
            **How RecoFlix handles new users:**
            - Uses **Content-Based Filtering** (no collaborative data available)
            - Recommends **popular movies** from your preferred genres
            - Falls back to **overall popular movies** if no preferences set
            - As you rate movies, the system will switch to **hybrid recommendations**
            """)
            
        else:
            # Existing user insights (original logic)
            with st.spinner("Analyzing user preferences..."):
                try:
                    user_insights = hybrid_system.get_user_insights(selected_user)
                    display_user_insights(user_insights)
                    
                    # Show user's rating history
                    if selected_user in processed_data['user_item_matrix'].index:
                        user_ratings = processed_data['user_item_matrix'].loc[selected_user]
                        rated_movies = user_ratings[user_ratings > 0]
                        
                        if len(rated_movies) > 0:
                            st.markdown("#### 🎥 Your Recent Ratings")
                            
                            # Show top rated movies
                            top_rated = rated_movies.nlargest(5)
                            
                            for movie_id, rating in top_rated.items():
                                movie_info = get_movie_info(movie_id, data['movies'])
                                col1, col2 = st.columns([4, 1])
                                
                                with col1:
                                    st.write(f"**{movie_info['title']}**")
                                    st.write(format_genres(movie_info['genres']), unsafe_allow_html=True)
                                
                                with col2:
                                    st.metric("Your Rating", f"{rating:.1f}⭐")
                    
                except Exception as e:
                    st.error(f"Error loading user insights: {e}")
    
    with tab3:
        st.markdown("## 📊 RecoFlix Analytics")
        
        # System overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_users = len(processed_data['user_item_matrix'].index)
            st.metric("Total Users", f"{total_users:,}")
        
        with col2:
            total_movies = len(processed_data['user_item_matrix'].columns)
            st.metric("Total Movies", f"{total_movies:,}")
        
        with col3:
            total_ratings = int(processed_data['user_item_matrix'].sum().sum())
            st.metric("Total Ratings", f"{total_ratings:,}")
        
        with col4:
            avg_rating = processed_data['user_item_matrix'].replace(0, np.nan).mean().mean()
            st.metric("Average Rating", f"{avg_rating:.2f}⭐")
        
        # Rating distribution
        st.markdown("#### 📈 Rating Distribution")
        
        all_ratings = processed_data['user_item_matrix'].values.flatten()
        all_ratings = all_ratings[all_ratings > 0]  # Remove zeros
        
        # Create DataFrame for plotly
        rating_df = pd.DataFrame({'rating': all_ratings})
        
        fig = px.histogram(
            rating_df,
            x='rating',
            nbins=10,
            title="Distribution of Movie Ratings",
            color_discrete_sequence=['#E50914']
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title="Rating",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Genre popularity
        st.markdown("#### 🎭 Most Popular Genres")
        
        genre_counts = {}
        for _, movie in data['movies'].iterrows():
            genres = str(movie['genres'])
            if genres != '(no genres listed)' and pd.notna(genres):
                for genre in genres.split('|'):
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        if genre_counts:
            top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig = px.pie(
                values=list(top_genres.values()),
                names=list(top_genres.keys()),
                title="Top 10 Genres by Number of Movies",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## 🔍 Explore Movies")
        
        # Movie search
        st.markdown("#### 🎬 Search Movies")
        search_term = st.text_input("Search for movies by title:", placeholder="Enter movie title...")
        
        if search_term:
            matching_movies = data['movies'][
                data['movies']['title'].str.contains(search_term, case=False, na=False)
            ]
            
            if len(matching_movies) > 0:
                st.markdown(f"Found {len(matching_movies)} movies:")
                
                for _, movie in matching_movies.head(10).iterrows():
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4>{movie['title']}</h4>
                        <p>{format_genres(movie['genres'])}</p>
                        <p><small>Movie ID: {movie['movieId']}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No movies found matching your search.")
        
        # Random movie discovery
        st.markdown("#### 🎲 Discover Random Movies")
        
        if st.button("🎲 Show Me Random Movies", key="random_movies"):
            random_movies = data['movies'].sample(5)
            
            st.markdown("##### Random Movie Selection:")
            for _, movie in random_movies.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <h4>{movie['title']}</h4>
                    <p>{format_genres(movie['genres'])}</p>
                    <p><small>Movie ID: {movie['movieId']}</small></p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #888888;">
        <p>🎬 RecoFlix - Powered by Advanced Machine Learning | Built with ❤️ using Streamlit</p>
        <p><small>Recommendation Engine: Hybrid Collaborative + Content-Based Filtering</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()