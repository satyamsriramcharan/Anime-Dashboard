import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Anime Industry Analytics",
    page_icon="🎌",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #e94560;
        text-align: center;
    }
    .title-text {
        font-size: 42px;
        font-weight: bold;
        color: #e94560;
        text-align: center;
    }
    .subtitle-text {
        font-size: 16px;
        color: #a0a0a0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Groq Client ───────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ── Load & Clean Data ─────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("animes.csv")

    # Extract year from aired column
    df['year'] = df['aired'].str.extract(r'(\d{4})').astype(float)

    # Clean score
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # Clean episodes
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

    # Clean members
    df['members'] = pd.to_numeric(df['members'], errors='coerce')

    # Clean genre — convert string list to actual list
    df['genre'] = df['genre'].str.strip("[]").str.replace("'", "")

    # Drop rows with no score or year
    df = df.dropna(subset=['score', 'year'])

    # Filter realistic years
    df = df[(df['year'] >= 1960) & (df['year'] <= 2024)]

    return df

df = load_data()

# ── Header ────────────────────────────────────────────
st.markdown('<p class="title-text">🎌 Anime Industry Analytics Dashboard</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Exploring trends, ratings, and growth of anime from 1960 to 2024</p>',
            unsafe_allow_html=True)
st.divider()

# ── KPI Metrics ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Anime", f"{len(df):,}")
with col2:
    st.metric("Average Score", f"{df['score'].mean():.2f}")
with col3:
    st.metric("Total Members", f"{df['members'].sum()/1e6:.1f}M")
with col4:
    st.metric("Years Covered", f"{int(df['year'].min())} – {int(df['year'].max())}")

st.divider()

# ── Sidebar Filters ───────────────────────────────────
st.sidebar.header("🔍 Filters")
year_range = st.sidebar.slider(
    "Select Year Range",
    int(df['year'].min()),
    int(df['year'].max()),
    (2000, 2024)
)
min_score = st.sidebar.slider("Minimum Score", 0.0, 10.0, 6.0, 0.1)

filtered_df = df[
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1]) &
    (df['score'] >= min_score)
]

st.sidebar.markdown(f"**{len(filtered_df):,} anime** match your filters")

# ── Chart 1: Anime Releases Per Year ─────────────────
st.subheader("📈 Anime Releases Per Year")
releases_per_year = df.groupby('year').size().reset_index(name='count')
fig1 = px.line(
    releases_per_year,
    x='year', y='count',
    title="Growth of Anime Industry (Releases Per Year)",
    color_discrete_sequence=['#e94560'],
    template='plotly_dark'
)
fig1.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Anime Released"
)
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Top 10 Highest Rated Anime ──────────────
st.subheader("🏆 Top 10 Highest Rated Anime")
top10 = df.nlargest(10, 'score')[['title', 'score', 'year']].reset_index(drop=True)
fig2 = px.bar(
    top10,
    x='score', y='title',
    orientation='h',
    title="Top 10 Anime by Score",
    color='score',
    color_continuous_scale='reds',
    template='plotly_dark',
    text='score'
)
fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: Most Popular Genres ─────────────────────
st.subheader("🎭 Most Popular Genres")
genre_series = df['genre'].dropna().str.split(', ').explode()
genre_counts = genre_series.value_counts().head(15).reset_index()
genre_counts.columns = ['genre', 'count']
fig3 = px.bar(
    genre_counts,
    x='genre', y='count',
    title="Top 15 Most Common Genres",
    color='count',
    color_continuous_scale='reds',
    template='plotly_dark'
)
fig3.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig3, use_container_width=True)

# ── Chart 4: Score Distribution ───────────────────────
st.subheader("📊 Score Distribution")
fig4 = px.histogram(
    filtered_df,
    x='score',
    nbins=30,
    title=f"Score Distribution ({year_range[0]}–{year_range[1]})",
    color_discrete_sequence=['#e94560'],
    template='plotly_dark'
)
fig4.update_layout(
    xaxis_title="Score",
    yaxis_title="Number of Anime"
)
st.plotly_chart(fig4, use_container_width=True)

# ── Chart 5: Members vs Score ─────────────────────────
st.subheader("⭐ Popularity vs Quality")
scatter_df = filtered_df.dropna(subset=['members']).copy()
scatter_df['members_m'] = scatter_df['members'] / 1e6
fig5 = px.scatter(
    scatter_df,
    x='score', y='members_m',
    hover_name='title',
    title="Score vs Community Size",
    color='score',
    color_continuous_scale='reds',
    template='plotly_dark',
    opacity=0.6
)
fig5.update_layout(
    xaxis_title="Score",
    yaxis_title="Members (Millions)"
)
st.plotly_chart(fig5, use_container_width=True)

# ── AI Insights ───────────────────────────────────────
st.divider()
st.subheader("🤖 AI-Generated Insights")

if st.button("Generate AI Insights", type="primary"):
    with st.spinner("AI is analyzing the anime data..."):

        top_genres = genre_counts.head(5)['genre'].tolist()
        avg_score = df['score'].mean()
        peak_year = releases_per_year.loc[releases_per_year['count'].idxmax(), 'year']
        total = len(df)

        prompt = f"""
You are an anime industry analyst. Based on this data from MyAnimeList:

- Total anime in database: {total}
- Average score: {avg_score:.2f}/10
- Peak release year: {int(peak_year)}
- Top 5 genres: {', '.join(top_genres)}
- Year range analyzed: {year_range[0]} to {year_range[1]}
- Anime matching filters: {len(filtered_df)}

Give 4 interesting insights about the anime industry trends in this format:

**INSIGHT 1 — [Title]:**
(2-3 sentences)

**INSIGHT 2 — [Title]:**
(2-3 sentences)

**INSIGHT 3 — [Title]:**
(2-3 sentences)

**INSIGHT 4 — [Title]:**
(2-3 sentences)

Keep it analytical and interesting for someone who loves anime.
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        insights = response.choices[0].message.content
        st.markdown(insights)

st.divider()
st.caption("Built with Python, Streamlit, Plotly & Groq AI | Data: MyAnimeList | Project by Sriram Charan")