import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MoodBite | AI Mood-Based Food Recommender",
    page_icon="🍔",
    layout="wide"
)

# Custom CSS for UI Enhancement
# Custom CSS for UI Enhancement (Dark & Light Mode Friendly)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        text-align: center;
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        color: #1a1a1a !important; /* Forces dark readable text */
    }
    .card h3 {
        color: #111111 !important;
        margin-top: 0;
    }
    .card p {
        color: #333333 !important;
    }
    .card b {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------------------------
# 2. DATA LOADING & ML ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
   return pd.read_csv("food_mood_dataset_cleaned.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Dataset file 'food_mood_dataset.csv' not found. Run the CSV generation script first!")
    st.stop()

def get_recommendations(user_query, df, selected_diet, max_calories):
    # Filter by Diet
    if selected_diet != "All":
        filtered_df = df[df['Veg_NonVeg'] == selected_diet].copy()
    else:
        filtered_df = df.copy()
        
    # Filter by Calories
    filtered_df = filtered_df[filtered_df['Calories'] <= max_calories]
    
    if filtered_df.empty:
        return pd.DataFrame()

    # Content-Based Filtering using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    # Combine Target Mood and Description for semantic search
    corpus = filtered_df['Target_Mood'] + " " + filtered_df['Description'] + " " + filtered_df['Category']
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([user_query])
    
    # Calculate Similarity Score
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    filtered_df['Match_Score'] = (similarity_scores * 100).round(1)
    
    # Sort by score and rating
    results = filtered_df.sort_values(by=['Match_Score', 'Rating'], ascending=False)
    return results

# -----------------------------------------------------------------------------
# 3. USER INTERFACE (UI)
# -----------------------------------------------------------------------------
st.markdown("<h1 class='main-header'>🍽️ MoodBite AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Discover perfect food recommendations tailored to your current emotional state.</p>", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("⚙️ Dietary Preferences")
diet_option = st.sidebar.radio("Dietary Type", ["All", "Veg", "Non-Veg"])
max_cal = st.sidebar.slider("Maximum Calories", min_value=50, max_value=800, value=700, step=50)

# Main Input Section
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. How are you feeling?")
    preset_mood = st.selectbox(
        "Select a preset mood:",
        ["Custom Input", "Stressed 😫", "Sad 😢", "Happy 😊", "Bored 🥱", "Energetic ⚡", "Sick/Tired 🤒", "Angry 😡"]
    )

with col2:
    st.subheader("2. Express your mood")
    if preset_mood == "Custom Input":
        user_input = st.text_input("Type how you feel right now:", "I feel exhausted after work and need something comforting")
    else:
        clean_mood = preset_mood.split(" ")[0]
        user_input = st.text_input("Refine your input or use preset:", f"I am feeling very {clean_mood}")

# Recommendation Trigger
if st.button("🚀 Find My Food", type="primary"):
    with st.spinner("Analyzing your mood using NLP..."):
        recommendations = get_recommendations(user_input, df, diet_option, max_cal)
        
    st.markdown("---")
    st.subheader("✨ Recommended Foods for You")
    
    if recommendations.empty:
        st.warning("No foods matched your specific filters. Try increasing the max calorie limit or changing dietary preference.")
    else:
        top_recs = recommendations.head(4)
        
        for idx, row in top_recs.iterrows():
            st.markdown(f"""
            <div class="card">
                <h3>{row['Food_Name']} <span style="font-size: 0.8em; color: #888;">({row['Category']})</span></h3>
                <p><b>Diet:</b> {'🟢 Veg' if row['Veg_NonVeg'] == 'Veg' else '🔴 Non-Veg'} | 
                   <b>Calories:</b> {row['Calories']} kcal | 
                   <b>Prep Time:</b> {row['Cooking_Time_Min']} mins | 
                   <b>Rating:</b> ⭐ {row['Rating']}/5.0</p>
                <p><b>Why it helps:</b> {row['Description']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.sidebar.subheader("📊 Dataset Overview")
        st.sidebar.write(f"Total dishes available: **{len(df)}**")
        st.sidebar.dataframe(df[['Food_Name', 'Target_Mood', 'Calories']], height=200)