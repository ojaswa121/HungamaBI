import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
import re
import hashlib
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
import spacy

# Set deterministic language detection
DetectorFactory.seed = 0

# Language mapping for readable names
LANG_CODE_TO_NAME = {
    'en': 'English', 'hi': 'Hindi', 'pa': 'Punjabi', 'mr': 'Marathi', 'gu': 'Gujarati', 'bn': 'Bengali',
    'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'kn': 'Kannada', 'ur': 'Urdu', 'sa': 'Sanskrit',
    'ne': 'Nepali', 'or': 'Odia', 'as': 'Assamese', 'kok': 'Konkani', 'sd': 'Sindhi', 'other': 'Other',
}

# === Configuration ===
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
POSTGRES_URI = 'postgresql://hungama:Hungama%402025%24%24@10.0.0.228:5432/youtube_scraper'

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Could not load spaCy model. Some features may be limited.")
    nlp = None

st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
st.title("Hungama BI : Smart Data Analyzer — YouTube Revenue Validation")

# ---------------- Context Layers ----------------
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'analysis_context' not in st.session_state:
    st.session_state.analysis_context = {}
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
if 'youtube_data' not in st.session_state:
    st.session_state.youtube_data = None

def add_to_conversation_history(query, response, insights=None, charts_generated=None):
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query': query,
        'response': response,
        'insights': insights or [],
        'charts_generated': charts_generated or [],
        'session_id': st.session_state.session_id
    }
    st.session_state.conversation_history.append(entry)
    if len(st.session_state.conversation_history) > 10:
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]

def update_analysis_context(key_insights):
    for insight in key_insights:
        category = insight.get('category', 'general')
        if category not in st.session_state.analysis_context:
            st.session_state.analysis_context[category] = []
        st.session_state.analysis_context[category].append({
            'insight': insight.get('text', ''),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'confidence': insight.get('confidence', 0.5)
        })

def get_conversation_context():
    if not st.session_state.conversation_history:
        return ""
    context = "\n📝 **Previous Conversation Context:**\n"
    for i, entry in enumerate(st.session_state.conversation_history[-5:], 1):
        context += f"\n**Q{i}:** {entry['query']}\n"
        context += f"**A{i}:** {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}\n"
        if entry['insights']:
            context += f"**Key Insights:** {', '.join([insight.get('text', '')[:50] for insight in entry['insights']])}\n"
    if st.session_state.analysis_context:
        context += "\n🧠 **Key Analysis Context:**\n"
        for category, insights in st.session_state.analysis_context.items():
            latest_insight = insights[-1] if insights else {}
            context += f"- **{category.title()}:** {latest_insight.get('insight', '')[:100]}\n"
    return context

def extract_insights_from_response(response):
    insights = []
    if not response:
        return insights
    response_lower = response.lower()
    if any(word in response_lower for word in ['revenue', 'income', 'earnings']):
        insights.append({'category': 'revenue', 'text': 'Revenue analysis discussed', 'confidence': 0.8})
    if any(word in response_lower for word in ['trend', 'growth', 'decline']):
        insights.append({'category': 'trends', 'text': 'Trend analysis provided', 'confidence': 0.7})
    if any(word in response_lower for word in ['recommend', 'suggest', 'should']):
        insights.append({'category': 'recommendations', 'text': 'Recommendations provided', 'confidence': 0.9})
    if any(word in response_lower for word in ['month', 'seasonal', 'quarterly']):
        insights.append({'category': 'temporal', 'text': 'Temporal analysis conducted', 'confidence': 0.6})
    return insights

# === Database & Queries ===
@st.cache_resource
def get_engine():
    return create_engine(POSTGRES_URI)

@st.cache_data(ttl=300)
def get_table_names():
    try:
        engine = get_engine()
        df = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'", engine)
        return df['table_name'].tolist()
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return []

@st.cache_data(ttl=300)
def load_data(table):
    try:
        engine = get_engine()
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        
        # Process data similar to first version
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0)
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
        df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0)
        df["title"] = df["title"].fillna("")
        df["description"] = df["description"].fillna("")
        
        # Add derived columns
        df["Year"] = df["published_at"].dt.year
        df["Month"] = df["published_at"].dt.strftime('%B')
        df["Date"] = df["published_at"].dt.date
        df["Time"] = df["published_at"].dt.strftime('%H:%M:%S')
        df["Day Name"] = df["published_at"].dt.day_name()
        
        # Add record label detection
        known_labels = [
            "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
            "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"
        ]
        
        def detect_label(row):
            if "channel_name" in row and pd.notna(row["channel_name"]):
                return row["channel_name"]
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            for label in known_labels:
                if label.lower() in text:
                    return label
            return "Other"
            
        df["Record Label"] = df.apply(detect_label, axis=1)
        
        # Add artist extraction
        def extract_artists(row):
            text = f"{row.get('title', '')} {row.get('description', '')}"
            hashtags = re.findall(r"#(\w+)", text)
            artist_patterns = re.findall(r"(singer[s]*|sung by|featuring|starring|artist[s]*):? ([\w\s&,.']+)", text, re.I)
            artists = []
            for match in artist_patterns:
                parts = re.split(r",|&|and", match[1])
                artists.extend([x.strip() for x in parts if len(x.strip()) > 2 and not x.strip().isdigit()])
            artist_hashtags = [tag for tag in hashtags if any(
                key in tag.lower() for key in [
                    'singh', 'khan', 'rao', 'kaur', 'ali', 'kapoor', 'chopra', 'anwar', 'amod', 'pooja',
                    'kk', 'pritam', 'sunidhi', 'arijit', 'shilpa', 'ayushmann', 'akshay', 'kumar', 'katrina',
                    'kaif', 'neha', 'benny', 'nakkash', 'aditi', 'anupam', 'harshdeep', 'loh', 'amod'
                ]
            ) or len(tag) > 5]
            all_artists = set(a.title() for a in artists if a) | set(artist_hashtags)
            return ", ".join(sorted(all_artists)) if all_artists else ""
            
        df["Artists"] = df.apply(extract_artists, axis=1)
        
        # Add language detection
        def guess_language(row):
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            hashtags = re.findall(r"#(\w+)", text)
            for tag in hashtags:
                for code, name in LANG_CODE_TO_NAME.items():
                    if code != 'other' and code in tag:
                        return name
                    if name.lower() in tag:
                        return name
            try:
                detected = detect(text)
                return LANG_CODE_TO_NAME.get(detected, 'Other')
            except:
                return 'Other'
                
        df["Language"] = df.apply(guess_language, axis=1)
        
        return df[df["view_count"] > 0]
        
    except Exception as e:
        st.error(f"Error loading data from {table}: {str(e)}")
        return pd.DataFrame()

def detect_months_and_confidence(text):
    if not text or not nlp:
        return [], 0.0
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    found = {token.text for token in nlp(text.lower()) if token.text in months}
    confidence = len(found) / 12
    return list(found), confidence

def render_visuals_from_keywords(text, videos, monthly, top_videos):
    if not text:
        return []
    charts_generated = []
    text_lower = text.lower()
    months, conf = detect_months_and_confidence(text)
    
    if conf >= 0.25:
        st.subheader("📈 Monthly Revenue Trend")
        st.caption(f"Confidence: {conf:.2f} — Months detected: {', '.join(months)}")
        fig = px.line(monthly, x="Month", y="Estimated Revenue INR", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(monthly)
        charts_generated.append("Monthly Revenue Trend")

    if "performance" in text_lower or "comparison" in text_lower:
        st.subheader("📊 Performance Comparison")
        fig = px.scatter(videos, x="view_count", y="like_count", color="Language", hover_name="title")
        st.plotly_chart(fig, use_container_width=True)
        charts_generated.append("Performance Scatter Plot")
        
    return charts_generated

def generate_cxo_forecasting_prompt(user_query, label_videos, monthly, est_total, actual_total, rpm):
    if label_videos.empty:
        return "No video data available for analysis."

    label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
    video_sample = label_videos[["title", "view_count", "like_count", "comment_count", "Estimated Revenue INR", "Month", "RPV_Estimated"]].head(10)
    monthly_sample = monthly.head(10) if not monthly.empty else pd.DataFrame()
    accuracy_str = f"{est_total / actual_total:.2%}" if actual_total else "N/A"
    conversation_context = get_conversation_context()

    return f"""
You are a **Content Intelligence and Business Forecasting Analyst** embedded in a strategic forecasting tool for CXOs and business leaders.

Your role is to:

- Analyze streaming content data (YouTube & other sources)
- Understand metadata such as **view count**, **publish time**, **content duration**, **engagement**, and **broad descriptions**
- Integrate structured inputs like uploaded Revenue sheets and stream performance metrics
- Forecast revenue trends based on smart defaults and user prompts

🧠 **Context Awareness**:

Maintain a structured memory of ongoing conversations. For each user session:

- Tie your analysis and responses back to the original label, context, and data unless explicitly told otherwise
- Support parallel comparisons of different labels or investments within the same session
- If new sheets or data are uploaded, validate the changes, flag differences, and ask for confirmation before switching context

📌 **Key Responsibilities**:

- Answer user queries using ONLY available, verifiable data
- Ask clarifying questions when required data is missing
- Suggest likely next steps, KPIs to track, or decision points
- Flag inconsistencies or suspicious data patterns

🧮 **Financial Forecasting Rules** (to be applied unless overridden):

- Default revenue split: 75% to label, 25% to Hungama
- Default investment period: 5 years
- Default streaming split: 60-70% video, 30-40% audio
- Post-investment: Expect 3 months of revenue dip → 3 months of stagnation → 15% revenue uplift from month 7
- Default revenue growth YoY: 20%
- Compute breakeven point and exit timeline against investment

🔍 **Current Input Data Snapshot**:

{conversation_context}

▶️ **Top Content Sample**:

{video_sample.to_json(orient="records", indent=2)}

📆 **Monthly Revenue Overview**:

{monthly_sample.to_json(orient="records", indent=2)}

📊 **Business Summary**:

- RPM: ₹{rpm}
- Estimated Revenue: ₹{est_total:,.2f}
- Actual Reported Revenue: ₹{actual_total:,.2f}
- Accuracy: {accuracy_str}
- Total Videos: {len(label_videos)}

---

### 💬 Current CXO Question:

"{user_query}"

---

### Instructions:

1. Build answers step-by-step, using available data.
2. Reference any uploaded Excel or stream sheets.
3. Validate revenue projection logic before giving outputs.
4. Recommend only grounded, specific actions.
5. Flag data gaps or anomalies where relevant.
6. Maintain continuity and context memory throughout.
7. Format your answer clearly using Markdown (e.g., bullet points, tables).
8. Do **not** hallucinate. If you don't have data, say so.
9. Be formal and concise unless the CXO shifts the tone.
"""

def get_deepseek_analysis(prompt, api_key):
    if not api_key or not prompt:
        return "", "Missing API key or prompt"
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://youtube-analytics.streamlit.app",
            "X-Title": "YouTube Analytics Dashboard"
        }
        payload = {
            "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], None
        return "", f"API Error {response.status_code}: {response.text}"
    except Exception as e:
        return "", str(e)

# === Sidebar for Context Management ===
with st.sidebar:
    st.header("🧠 Context Memory")
    if st.button("🔄 Clear Conversation History"):
        st.session_state.conversation_history = []
        st.session_state.analysis_context = {}
        st.success("Context cleared!")
        
    st.subheader("📊 Session Stats")
    st.metric("Queries Asked", len(st.session_state.conversation_history))
    st.metric("Session ID", st.session_state.session_id)
    
    if st.session_state.conversation_history:
        st.subheader("💬 Recent Queries")
        for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
            with st.expander(f"Query {len(st.session_state.conversation_history) - 3 + i}"):
                st.write(f"**Q:** {entry['query'][:100]}...")
                st.write(f"**Time:** {entry['timestamp']}")
                if entry['charts_generated']:
                    st.write(f"**Charts:** {', '.join(entry['charts_generated'])}")

    st.subheader("📦 Data Source")
    tables = get_table_names()
    if tables:
        selected_table = st.selectbox("Select Table", tables)
    else:
        st.error("No database tables found")
        st.stop()

# === Main App ===
st.header("📊 YouTube Analytics Dashboard")

# Load YouTube data from PostgreSQL
if st.session_state.youtube_data is None:
    with st.spinner("Loading data from PostgreSQL..."):
        st.session_state.youtube_data = load_data(selected_table)

df = st.session_state.youtube_data

if df.empty:
    st.error("No data loaded from database")
    st.stop()

st.success(f"✅ Loaded {len(df)} records from `{selected_table}`")

# File upload and data processing
uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])
actual_total = 0

if uploaded_file is not None:
    try:
        revenue_df = pd.read_csv(uploaded_file)
        if "Store Name" in revenue_df.columns:
            yt_row = revenue_df[revenue_df["Store Name"].str.lower().str.contains("youtube", na=False)]
            actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")

# Label selection with "All" option
label_options = ["All"] + sorted(df["Record Label"].unique())
selected_label = st.selectbox(
    "🎙️ Choose Record Label",
    label_options,
    index=0
)

rpm = st.number_input(
    "💸 RPM (Revenue per Million Views)", 
    min_value=500, 
    value=125000,
    help="Revenue per 1 million views in INR"
)

# Filter videos for selected label or all
if selected_label == "All":
    label_videos = df.copy()
else:
    label_videos = df[df["Record Label"] == selected_label].copy()

label_videos = label_videos.dropna(subset=["view_count"])
label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
est_total = label_videos["Estimated Revenue INR"].sum()

# Process temporal data
if "published_at" in label_videos:
    label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
    label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
    monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
else:
    monthly_revenue = pd.DataFrame()
    
label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
top_rpv = label_videos.nlargest(10, "RPV_Estimated")[
    ["title", "view_count", "like_count", "comment_count", "Estimated Revenue INR", "RPV_Estimated"]
]

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Estimated Revenue", f"₹{est_total:,.0f}")
with col2:
    st.metric("Actual Revenue", f"₹{actual_total:,.0f}")
with col3:
    st.metric("Accuracy", f"{(est_total / actual_total):.2%}" if actual_total else "N/A")

# Enhanced filtering options
with st.expander("🔍 Advanced Filters", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        # Artist filtering
        artist_options = sorted({a.strip() for aa in label_videos["Artists"].dropna() 
                              for a in aa.split(",") if a.strip()})
        selected_artists = st.multiselect("🎤 Filter by Artist(s)", artist_options)
        
        # Language filtering
        lang_options = sorted(label_videos["Language"].dropna().unique())
        selected_langs = st.multiselect("🌐 Filter by Language(s)", lang_options)
        
    with col2:
        # Date range filtering
        if "published_at" in label_videos:
            valid_dates = label_videos["published_at"].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                date_range = st.date_input(
                    "📅 Filter by Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    label_videos = label_videos[
                        (label_videos["published_at"].dt.date >= date_range[0]) & 
                        (label_videos["published_at"].dt.date <= date_range[1])
                    ]
        
        # View count filtering
        if not label_videos.empty:
            min_views = int(label_videos["view_count"].min())
            max_views = int(label_videos["view_count"].max())
            view_range = st.slider(
                "👀 Filter by View Count",
                min_value=min_views,
                max_value=max_views,
                value=(min_views, max_views)
            )
            label_videos = label_videos[
                (label_videos["view_count"] >= view_range[0]) & 
                (label_videos["view_count"] <= view_range[1])
            ]

    # Apply other filters
    if selected_artists:
        regex = "|".join([re.escape(a) for a in selected_artists])
        label_videos = label_videos[label_videos["Artists"].str.contains(regex, case=False, na=False)]
        
    if selected_langs:
        label_videos = label_videos[label_videos["Language"].isin(selected_langs)]

# === Charts ===
st.subheader("📈 Monthly Revenue Trend")
if not monthly_revenue.empty:
    fig_line = px.line(monthly_revenue, x="Month", y="Estimated Revenue INR", title="Monthly Estimated Revenue", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

top10 = label_videos.sort_values("Estimated Revenue INR", ascending=False).head(10)
if not top10.empty:
    fig_bar = px.bar(top10, x="title", y="Estimated Revenue INR", title="Top 10 Revenue-Generating Videos")
    fig_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Sticky Chat Input Section ---
st.subheader("Ask a Business Intelligence Question")

# Suggested questions as selectbox above chat input
suggested_questions = [
    "What is the average view count per video?",
    "Which video has the highest engagement rate?",
    "How does revenue vary by language?",
    "What is the monthly revenue trend?",
    "Which artists generate the most revenue?",
    "Which label generates the highest revenue per month?",
    "How many videos would it take to recover an ₹8 Cr investment?",
    "What's the ROI if we invest ₹5 Cr into our top channel?",
]
selected_suggestion = st.selectbox(
    "💡 Or select a suggested question:",
    [""] + suggested_questions
)

# Sticky chat input at the bottom
user_query = st.chat_input(
    "Type your business intelligence question here...",
    key="chat_query"
)

# If a suggestion is selected, show it as info if chat input is empty
if selected_suggestion and not user_query:
    st.info(f"Suggested: {selected_suggestion}")

# Show chat history (user question and answer) like ChatGPT
if st.session_state.conversation_history:
    st.markdown("### 💬 Chat History")
    for entry in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.markdown(entry["query"])
        with st.chat_message("assistant"):
            st.markdown(entry["response"])

# Quick action buttons
if st.session_state.conversation_history:
    st.write("**Quick Actions:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Deep dive into this"):
            user_query = "Can you provide a deeper analysis of the previous insight?"
    with col2:
        if st.button("🔍 What's next?"):
            user_query = "Based on the previous analysis, what should be the next steps?"
    with col3:
        if st.button("📊 Compare trends"):
            user_query = "How do these findings compare to industry benchmarks?"

if user_query:
    with st.spinner("Analyzing with Hungama BI..."):
        full_prompt = generate_cxo_forecasting_prompt(
            user_query, label_videos, monthly_revenue, est_total, actual_total, rpm
        )
        
        if st.session_state.get("debug_mode", False):
            with st.expander("🔧 Debug: View Full Prompt"):
                st.code(full_prompt)
        
        response, error = get_deepseek_analysis(full_prompt, API_KEY)

        if error:
            st.error(f"Analysis Error: {error}")
        else:
            # Show the latest user question and answer in chat format
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.chat_message("assistant"):
                st.markdown(response)
            
            charts_generated = render_visuals_from_keywords(
                response, label_videos, monthly_revenue, top_rpv
            )
            
            insights = extract_insights_from_response(response)
            update_analysis_context(insights)
            add_to_conversation_history(user_query, response, insights, charts_generated)
            
            if len(st.session_state.conversation_history) > 1:
                with st.expander("🔗 Context Connection"):
                    st.write("This analysis builds upon previous insights:")
                    for insight in insights:
                        st.write(f"- {insight['category'].title()}: {insight['text']}")

# Data exploration section
with st.expander("📋 Explore Video Data", expanded=False):
    show_cols = [
        "video_id", "title", "Record Label", "Artists", "Language", "Year",
        "Month", "Date", "Time", "Day Name", "view_count", "like_count", "comment_count",
        "duration", "published_at", "Estimated Revenue INR", "RPV_Estimated"
    ]
    show_cols = [col for col in show_cols if col in label_videos.columns]
    
    st.dataframe(
        label_videos[show_cols].sort_values("view_count", ascending=False),
        use_container_width=True,
        height=400
    )
    
    st.download_button(
        "📥 Export Video Data", 
        label_videos.to_csv(index=False), 
        f"{selected_label}_videos.csv"
    )

# Conversation history export
if st.session_state.conversation_history:
    conversation_export = pd.DataFrame(st.session_state.conversation_history)
    st.download_button(
        "📥 Export Conversation History",
        conversation_export.to_csv(index=False),
        f"conversation_history_{st.session_state.session_id}.csv"
    )

# Debug mode toggle (hidden)
if st.sidebar.checkbox("🔧 Debug Mode", False, key="debug_mode"):
    st.sidebar.write("Debug options enabled")
    if st.session_state.get("youtube_data") is not None:
        st.sidebar.write(f"Data shape: {st.session_state.youtube_data.shape}")
        st.sidebar.write(f"Database table: {selected_table}")


