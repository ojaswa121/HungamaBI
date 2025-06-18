# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import pymongo
# from pymongo import MongoClient

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
# MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
# MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "youtube_data")
# MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "videos")

# # Load NLP model
# nlp = spacy.load("en_core_web_sm")

# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Revenue Validation")

# # Initialize session state for context memory
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

# def add_to_conversation_history(query, response, insights=None, charts_generated=None):
#     """Add query and response to conversation history with metadata"""
#     entry = {
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'query': query,
#         'response': response,
#         'insights': insights or [],
#         'charts_generated': charts_generated or [],
#         'session_id': st.session_state.session_id
#     }
#     st.session_state.conversation_history.append(entry)
    
#     # Keep only last 10 conversations to manage memory
#     if len(st.session_state.conversation_history) > 10:
#         st.session_state.conversation_history = st.session_state.conversation_history[-10:]

# def update_analysis_context(key_insights):
#     """Update persistent analysis context with key findings"""
#     for insight in key_insights:
#         category = insight.get('category', 'general')
#         if category not in st.session_state.analysis_context:
#             st.session_state.analysis_context[category] = []
#         st.session_state.analysis_context[category].append({
#             'insight': insight.get('text', ''),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'confidence': insight.get('confidence', 0.5)
#         })

# def get_conversation_context():
#     """Generate conversation context for AI prompt"""
#     if not st.session_state.conversation_history:
#         return ""
    
#     context = "\n📝 **Previous Conversation Context:**\n"
#     for i, entry in enumerate(st.session_state.conversation_history[-5:], 1):  # Last 5 conversations
#         context += f"\n**Q{i}:** {entry['query']}\n"
#         context += f"**A{i}:** {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}\n"
#         if entry['insights']:
#             context += f"**Key Insights:** {', '.join([insight.get('text', '')[:50] for insight in entry['insights']])}\n"
    
#     # Add persistent context
#     if st.session_state.analysis_context:
#         context += "\n🧠 **Key Analysis Context:**\n"
#         for category, insights in st.session_state.analysis_context.items():
#             latest_insight = insights[-1] if insights else {}
#             context += f"- **{category.title()}:** {latest_insight.get('insight', '')[:100]}\n"
    
#     return context

# def extract_insights_from_response(response):
#     """Extract key insights from AI response for context building"""
#     insights = []
    
#     # Simple keyword-based insight extraction
#     response_lower = response.lower()
    
#     if any(word in response_lower for word in ['revenue', 'income', 'earnings']):
#         insights.append({'category': 'revenue', 'text': 'Revenue analysis discussed', 'confidence': 0.8})
    
#     if any(word in response_lower for word in ['trend', 'growth', 'decline']):
#         insights.append({'category': 'trends', 'text': 'Trend analysis provided', 'confidence': 0.7})
    
#     if any(word in response_lower for word in ['recommend', 'suggest', 'should']):
#         insights.append({'category': 'recommendations', 'text': 'Recommendations provided', 'confidence': 0.9})
    
#     if any(word in response_lower for word in ['month', 'seasonal', 'quarterly']):
#         insights.append({'category': 'temporal', 'text': 'Temporal analysis conducted', 'confidence': 0.6})
    
#     return insights

# def connect_to_mongodb():
#     """Connect to MongoDB and return the collection"""
#     try:
#         client = MongoClient(MONGODB_URI)
#         db = client[MONGODB_DATABASE]
#         collection = db[MONGODB_COLLECTION]
        
#         # Test the connection
#         collection.find_one()
#         return collection, None
#     except Exception as e:
#         return None, str(e)

# @st.cache_data(ttl=300, hash_funcs={MongoClient: lambda x: None})  # Cache for 5 minutes, ignore MongoDB objects
# def load_youtube_data_from_mongodb():
#     """Load YouTube data from MongoDB and return as DataFrame"""
#     try:
#         client = MongoClient(MONGODB_URI)
#         db = client[MONGODB_DATABASE]
#         collection = db[MONGODB_COLLECTION]
        
#         # Test connection
#         collection.find_one()
        
#         # Fetch all documents from MongoDB
#         cursor = collection.find({})
#         data = list(cursor)
        
#         # Close the connection
#         client.close()
        
#         if not data:
#             st.warning("No data found in MongoDB collection")
#             return pd.DataFrame()
        
#         # Convert to DataFrame
#         df = pd.DataFrame(data)
        
#         # Remove MongoDB's _id field if present
#         if '_id' in df.columns:
#             df = df.drop('_id', axis=1)
        
#         # Known record labels for detection
#         known_labels = ["T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#                         "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"]
        
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#             for label in known_labels:
#                 if label.lower() in text:
#                     return label
#             return "Other"
        
#         # Add record label detection
#         df["Record Label"] = df.apply(detect_label, axis=1)
        
#         return df
        
#     except Exception as e:
#         st.error(f"Error loading data from MongoDB: {str(e)}")
#         return pd.DataFrame()

# def get_mongodb_stats():
#     """Get statistics about the MongoDB collection"""
#     try:
#         client = MongoClient(MONGODB_URI)
#         db = client[MONGODB_DATABASE]
#         collection = db[MONGODB_COLLECTION]
        
#         stats = {
#             "total_documents": collection.count_documents({}),
#             "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
        
#         # Get sample document to show structure
#         sample_doc = collection.find_one({})
#         if sample_doc:
#             stats["sample_fields"] = list(sample_doc.keys())
        
#         # Close the connection
#         client.close()
        
#         return stats
#     except Exception as e:
#         return {"error": str(e)}

# def detect_months_and_confidence(text):
#     months = [
#         "january", "february", "march", "april", "may", "june",
#         "july", "august", "september", "october", "november", "december"
#     ]
#     found = {token.text for token in nlp(text.lower()) if token.text in months}
#     confidence = len(found) / 12
#     return list(found), confidence

# def render_visuals_from_keywords(text, videos, monthly, top_videos):
#     text_lower = text.lower()
#     charts_generated = []
    
#     months, conf = detect_months_and_confidence(text)
#     if conf >= 0.25:
#         st.subheader("📈 Monthly Revenue Trend")
#         st.caption(f"Confidence: {conf:.2f} — Months detected: {', '.join(months)}")
#         fig = px.line(monthly, x="Month", y="Estimated Revenue INR", markers=True)
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(monthly)
#         charts_generated.append("Monthly Revenue Trend")

#     if "top" in text_lower or "rpv" in text_lower:
#         st.subheader("🏆 Top Videos by RPV")
#         fig = px.bar(top_videos, x="RPV_Estimated", y="title", orientation="h")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(top_videos)
#         charts_generated.append("Top Videos by RPV")
    
#     return charts_generated

# def generate_enhanced_prompt(user_query, label_videos, monthly, est_total, actual_total, rpm):
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     video_sample = label_videos[["title", "view_count", "Estimated Revenue INR", "Month", "RPV_Estimated"]].head(10)
#     monthly_sample = monthly.head(10)

#     accuracy_str = f"{est_total / actual_total:.2%}" if actual_total else "N/A"
    
#     # Get conversation context
#     conversation_context = get_conversation_context()

#     return f"""
# You are a senior Business Intelligence analyst embedded inside a live dashboard with context memory.

# {conversation_context}

# ▶️ **Current Video Data Sample (Top 10):**
# {video_sample.to_json(orient="records", indent=2)}

# 📆 **Monthly Revenue (Top 10):**
# {monthly_sample.to_json(orient="records", indent=2)}

# 📊 **Current Business Summary**:
# - RPM: ₹{rpm}
# - Total Estimated Revenue: ₹{est_total:,.2f}
# - Actual Reported Revenue: ₹{actual_total:,.2f}
# - Accuracy: {accuracy_str}
# - Total Videos: {len(label_videos)}

# ---

# ### Current User Question:
# "{user_query}"

# ---

# IMPORTANT INSTRUCTIONS:
# 1. **Reference Previous Context**: Build upon previous questions and insights when relevant
# 2. **Provide Continuity**: If this question relates to previous queries, acknowledge and extend the analysis
# 3. **Use Context**: Reference previous findings, trends, or recommendations when applicable
# 4. **Avoid Repetition**: Don't repeat identical insights from previous responses
# 5. **Progressive Analysis**: Deepen the analysis based on conversation history

# Respond with data-driven insights only. Use real numbers from the input. Highlight metrics. 
# Recommend specific actions. Do not hallucinate. Use markdown formatting with sections and bullet points.

# If this is a follow-up question, explicitly connect it to previous analysis and build upon it.
# """

# def get_mistral_analysis(prompt, api_key):
#     try:
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 1500
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}"
#     except Exception as e:
#         return "", str(e)

# # === Sidebar for Context Management and MongoDB Stats ===
# with st.sidebar:
#     st.header("🧠 Context Memory")
    
#     if st.button("🔄 Clear Conversation History"):
#         st.session_state.conversation_history = []
#         st.session_state.analysis_context = {}
#         st.success("Context cleared!")
    
#     st.subheader("📊 Session Stats")
#     st.metric("Queries Asked", len(st.session_state.conversation_history))
#     st.metric("Session ID", st.session_state.session_id)
    
#     # MongoDB Connection Status
#     st.subheader("🗄️ MongoDB Status")
#     mongo_stats = get_mongodb_stats()
    
#     if "error" in mongo_stats:
#         st.error(f"MongoDB Error: {mongo_stats['error']}")
#     else:
#         st.success("✅ Connected to MongoDB")
#         st.metric("Total Videos", mongo_stats.get("total_documents", 0))
#         st.caption(f"Last updated: {mongo_stats.get('last_updated', 'Unknown')}")
        
#         if "sample_fields" in mongo_stats:
#             with st.expander("📋 Available Fields"):
#                 for field in mongo_stats["sample_fields"]:
#                     st.write(f"• {field}")
    
#     # Refresh data button
#     if st.button("🔄 Refresh MongoDB Data"):
#         st.cache_data.clear()
#         st.rerun()
    
#     if st.session_state.conversation_history:
#         st.subheader("💬 Recent Queries")
#         for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
#             with st.expander(f"Query {len(st.session_state.conversation_history) - 3 + i}"):
#                 st.write(f"**Q:** {entry['query'][:100]}...")
#                 st.write(f"**Time:** {entry['timestamp']}")
#                 if entry['charts_generated']:
#                     st.write(f"**Charts:** {', '.join(entry['charts_generated'])}")

# # === Main App ===
# st.info("🗄️ Loading data from MongoDB...")

# # Load data from MongoDB instead of JSON
# youtube_df = load_youtube_data_from_mongodb()

# if youtube_df.empty:
#     st.error("⚠️ No data available from MongoDB. Please check your connection and data.")
#     st.stop()

# st.success(f"✅ Loaded {len(youtube_df)} videos from MongoDB")

# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
    
#     # Check if we have the required columns
#     if "Record Label" not in youtube_df.columns:
#         st.error("❌ 'Record Label' column not found in MongoDB data. Please check your data structure.")
#         st.stop()
    
#     selected_label = st.selectbox("🎙️ Choose Record Label", sorted(youtube_df["Record Label"].unique()))
#     rpm = st.number_input("💸 RPM (Revenue per Million Views)", min_value=500, value=1200)

#     label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#     label_videos = label_videos.dropna(subset=["view_count"])
#     label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#     est_total = label_videos["Estimated Revenue INR"].sum()

#     if "Store Name" in df.columns:
#         yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#         actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
#     else:
#         actual_total = 0

#     # Handle different date column names that might come from MongoDB
#     date_columns = ['published_at', 'publishedAt', 'upload_date', 'date', 'created_at']
#     date_column = None
    
#     for col in date_columns:
#         if col in label_videos.columns:
#             date_column = col
#             break
    
#     if date_column:
#         label_videos[date_column] = pd.to_datetime(label_videos[date_column], errors="coerce")
#         label_videos["Month"] = label_videos[date_column].dt.to_period("M").astype(str)
#     else:
#         st.warning("⚠️ No date column found. Monthly analysis will be limited.")
#         label_videos["Month"] = "Unknown"
    
#     monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     top_rpv = label_videos.nlargest(10, "RPV_Estimated")[["title", "view_count", "Estimated Revenue INR", "RPV_Estimated"]]

#     # Display metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Estimated Revenue", f"₹{est_total:,.0f}")
#     with col2:
#         st.metric("Actual Revenue", f"₹{actual_total:,.0f}")
#     with col3:
#         st.metric("Accuracy", f"{(est_total / actual_total):.2%}" if actual_total else "N/A")

#     # Enhanced query interface
#     st.subheader("🧠 Ask a Business Intelligence Question")
    
#     # Show context hint
#     if st.session_state.conversation_history:
#         last_query = st.session_state.conversation_history[-1]['query']
#         st.info(f"💡 **Context Available** - Last query: '{last_query[:50]}...' - Ask follow-up questions for deeper insights!")
    
#     user_query = st.text_area("Your question:", placeholder="Ask about trends, comparisons, recommendations, or follow up on previous insights...")
    
#     # Quick suggestion buttons based on context
#     if st.session_state.conversation_history:
#         st.write("**Quick Follow-ups:**")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             if st.button("📈 Deep dive into this"):
#                 user_query = f"Can you provide a deeper analysis of the previous insight?"
#         with col2:
#             if st.button("🔍 What's next?"):
#                 user_query = f"Based on the previous analysis, what should be the next steps?"
#         with col3:
#             if st.button("📊 Compare trends"):
#                 user_query = f"How do these findings compare to industry benchmarks?"
    
#     if user_query:
#         with st.spinner("🧠 Mistral AI is analyzing with context..."):
#             full_prompt = generate_enhanced_prompt(user_query, label_videos, monthly_revenue, est_total, actual_total, rpm)
#             response, error = get_mistral_analysis(full_prompt, API_KEY)
            
#             if error:
#                 st.error(error)
#             else:
#                 st.markdown("### 🧠 Mistral Insight")
#                 st.markdown(response)
                
#                 # Generate visuals and track them
#                 charts_generated = render_visuals_from_keywords(response, label_videos, monthly_revenue, top_rpv)
                
#                 # Extract insights and update context
#                 insights = extract_insights_from_response(response)
#                 update_analysis_context(insights)
                
#                 # Add to conversation history
#                 add_to_conversation_history(user_query, response, insights, charts_generated)
                
#                 # Show connection to previous context
#                 if len(st.session_state.conversation_history) > 1:
#                     with st.expander("🔗 Context Connection"):
#                         st.write("This analysis builds upon:")
#                         for insight in insights:
#                             st.write(f"- {insight['category'].title()}: {insight['text']}")

#     st.download_button("📥 Export Video Data", label_videos.to_csv(index=False), "videos.csv")
    
#     # Export conversation history
#     if st.session_state.conversation_history:
#         conversation_export = pd.DataFrame(st.session_state.conversation_history)
#         st.download_button(
#             "📥 Export Conversation History", 
#             conversation_export.to_csv(index=False), 
#             f"conversation_history_{st.session_state.session_id}.csv"
#         )

# else:
#     st.info("📁 Upload a revenue CSV to get started.")
    
#     # Show welcome message with context capabilities and MongoDB info
#     st.markdown("""
#     ### 🧠 Context Memory Features:
#     - **Conversation History**: Remembers your previous questions and insights
#     - **Progressive Analysis**: Each query builds upon previous findings
#     - **Context Awareness**: AI understands the flow of your analysis
#     - **Session Persistence**: Maintains context throughout your session
#     - **Export Conversations**: Download your analysis journey
    
#     ### 🗄️ MongoDB Integration:
#     - **Real-time Data**: Fetches fresh data from your MongoDB collection
#     - **Automatic Refresh**: Data updates every 5 minutes
#     - **Connection Monitoring**: Shows MongoDB connection status
#     - **Field Detection**: Automatically detects available data fields
#     """)

###########################################################################################


# import streamlit as st
# import pandas as pd
# import openai
# import plotly.express as px
# import plotly.graph_objects as go
# import re
# from dateutil.parser import parse
# import io
# import contextlib

# # OpenAI (Ollama local) config
# openai.api_base = "http://localhost:11434/v1"
# openai.api_key = "ollama"

# st.set_page_config(page_title="AskCSV Pro", layout="wide")
# st.title("📊 AskCSV Pro — Intent-Aware LLM Data Analyst")
# st.markdown("Upload a CSV and ask for insights, summaries, tables, groupings, or visualizations.")

# def detect_date_columns(df):
#     date_cols = []
#     for col in df.columns:
#         try:
#             _ = parse(col, fuzzy=False)
#             date_cols.append(col)
#         except:
#             continue
#     return date_cols

# def query_is_textual_only(query):
#     query = query.lower()
#     return (
#         any(k in query for k in ["analysis", "summary", "insight", "understand", "describe", "overview"]) and
#         not any(k in query for k in ["plot", "chart", "graph", "table", "group", "sort", "draw", "filter", "code"])
#     )

# def auto_patch_code(original_code, df_columns):
#     if "subplots" in original_code or "ax.bar" in original_code:
#         if "Age" in df_columns and "Chronic Medical Conditions" in df_columns:
#             return '''
# df_grouped = df.groupby("Age")["Chronic Medical Conditions"].sum().reset_index()

# fig = px.bar(
#     df_grouped,
#     x="Age",
#     y="Chronic Medical Conditions",
#     title="Chronic Medical Conditions by Age",
#     labels={"Chronic Medical Conditions": "Number of Conditions"}
# )
# '''
#     return original_code

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
# query = st.text_input("Ask a question or request analysis")

# if "last_query" not in st.session_state:
#     st.session_state.last_query = ""

# if query:
#     st.session_state.last_query = query

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("📄 Data Preview")
#     st.dataframe(df.head(), use_container_width=True)

#     date_cols = detect_date_columns(df)
#     long_df = None
#     if date_cols:
#         id_vars = [col for col in df.columns if col not in date_cols]
#         long_df = df.melt(id_vars=id_vars, value_vars=date_cols, var_name="Date", value_name="Revenue")
#         try:
#             long_df["Date"] = pd.to_datetime(long_df["Date"], format="%m/%d/%Y")
#         except:
#             long_df["Date"] = pd.to_datetime(long_df["Date"], errors='coerce')

#     if st.session_state.last_query:
#         df_used = long_df if long_df is not None else df
#         intent_text_only = query_is_textual_only(st.session_state.last_query)

#         # ✅ Preset override: revenue trends
#         if "revenue" in st.session_state.last_query.lower():
#             revenue_cols = [col for col in df_used.columns if "revenue" in col.lower() or "fy" in col.lower()]
#             date_col = "Date" if "Date" in df_used.columns else None

#             if date_col and revenue_cols:
#                 df_trend = df_used[[date_col] + revenue_cols].copy().dropna()
#                 df_trend = df_trend.sort_values(by=date_col)

#                 fig = px.line(
#                     df_trend.melt(id_vars=[date_col], var_name="Source", value_name="Revenue"),
#                     x=date_col,
#                     y="Revenue",
#                     color="Source",
#                     title="Revenue Trends Over Time"
#                 )
#                 st.subheader("📈 Chart")
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.stop()

#         with st.spinner("Analyzing..."):
#             prompt = f"""
# You are a Python and Plotly data analyst working with a DataFrame named `df`, which comes from a user-uploaded CSV.

# ### USER REQUEST:
# {st.session_state.last_query}

# ### DATA PREVIEW:
# {df_used.head(3).to_markdown(index=False)}

# ### COLUMNS:
# {', '.join(df_used.columns)}

# ### INSTRUCTIONS:
# - Only respond to this specific query.
# - Use Plotly Express if charting. No matplotlib, seaborn, or custom functions like 'create_line_chart'.
# - If plotting trends over time, assume 'Date' is the x-axis.
# - If the user says 'revenue', look for columns with 'Revenue' or 'FY' in the name.
# - Assign the chart to a variable named `fig`.
# - Do NOT use pd.read_csv or fake data.
# """

#             try:
#                 response = openai.ChatCompletion.create(
#                     model="gemma:2b",
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=0.2,
#                 )
#                 reply = response.choices[0].message["content"]
#                 st.success("LLM Response:")
#                 st.markdown(reply)

#                 code_blocks = re.findall(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)

#                 if code_blocks and not intent_text_only:
#                     code = code_blocks[0]

#                     if any(term in code for term in ["sns.", "plt.", "matplotlib", "go.subplots", "go.scatter", "ax.bar"]):
#                         st.warning("⚠️ Detected unsupported code. Attempting auto-patch...")
#                         code = auto_patch_code(code, df_used.columns)

#                     code = re.sub(r"pd\.read_csv\(.*?\)", "# file read removed", code)
#                     code = re.sub(r"import .*? as .*?\n", "", code)
#                     code = re.sub(r"fig,\s*ax\s*=\s*", "fig = ", code)
#                     code = code.replace("fig.show()", "")
#                     code = re.sub(r"xaxis_label\s*=\s*['\"].*?['\"],?", "", code)
#                     code = re.sub(r"yaxis_label\s*=\s*['\"].*?['\"],?", "", code)

#                     exec_locals = {"df": df_used.copy(), "px": px, "go": go, "pd": pd}
#                     stdout = io.StringIO()
#                     try:
#                         with contextlib.redirect_stdout(stdout):
#                             exec(code, {}, exec_locals)
#                     except Exception as exec_err:
#                         st.warning(f"⚠️ Code execution error: {exec_err}")
#                         st.info("Try using this example format:")
#                         st.code('''
# df_grouped = df.groupby("Age")["Chronic Medical Conditions"].sum().reset_index()

# fig = px.bar(
#     df_grouped,
#     x="Age",
#     y="Chronic Medical Conditions",
#     title="Chronic Medical Conditions by Age",
#     labels={"Chronic Medical Conditions": "Number of Conditions"}
# )
#                         ''')

#                     text_output = stdout.getvalue().strip()
#                     fig = exec_locals.get("fig")
#                     result_df = exec_locals.get("result_df")

#                     if text_output:
#                         st.subheader("🧠 Output")
#                         st.code(text_output)

#                     if isinstance(result_df, pd.DataFrame):
#                         st.subheader("📋 Table")
#                         st.dataframe(result_df, use_container_width=True)

#                     if fig:
#                         st.subheader("📈 Chart")
#                         st.plotly_chart(fig, use_container_width=True)

#                 elif intent_text_only:
#                     st.info("🧠 Insight-only mode: No code or chart shown as prompt didn’t request it.")
#                 else:
#                     st.warning("⚠️ No code block returned. Try asking to plot, group, or show a table.")

#             except Exception as e:
#                 st.error(f"❌ LLM processing error: {e}")

#         if st.button("🔁 Retry This Prompt"):
#             st.rerun()



# # old one without LLM 

# import streamlit as st
# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")

# import pandas as pd
# import numpy as np
# import plotly.express as px
# import json
# import warnings
# import requests
# from openai import OpenAI
# import time
# warnings.filterwarnings("ignore")

# # Initialize OpenAI client to connect to Ollama
# @st.cache_resource
# def init_ollama_client():
#     """Initialize OpenAI client for Ollama local server"""
#     try:
#         client = OpenAI(
#             base_url="http://localhost:11434/v1",  # Ollama OpenAI-compatible endpoint
#             api_key="ollama"  # Ollama doesn't require a real API key
#         )
#         return client
#     except Exception as e:
#         st.error(f"Failed to initialize Ollama client: {e}")
#         return None

# # Initialize session state
# if "query_history" not in st.session_state:
#     st.session_state.query_history = []
# if "ollama_client" not in st.session_state:
#     st.session_state.ollama_client = init_ollama_client()

# # --- Load YouTube Metadata from local JSON ---
# @st.cache_data
# def load_youtube_metadata():
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)

#         if isinstance(data, dict) and "videos" in data:
#             df = pd.DataFrame(data["videos"])
#         else:
#             df = pd.DataFrame(data)

#         # Known labels for detection
#         KNOWN_LABELS = [
#             "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#             "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"
#         ]

#         # Detect labels in title/description
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#             for label in KNOWN_LABELS:
#                 if label.lower() in text:
#                     return label
#             return "Other"

#         df["Record Label"] = df.apply(detect_label, axis=1)
#         return df

#     except Exception as e:
#         st.error(f"❌ Error loading YouTube metadata: {e}")
#         return pd.DataFrame()

# def check_ollama_status():
#     """Check if Ollama server is running"""
#     try:
#         response = requests.get("http://localhost:11434/api/version", timeout=5)
#         return response.status_code == 200
#     except:
#         return False

# def get_ollama_analysis(prompt, model="mistral"):
#     """Get analysis from Ollama using OpenAI-compatible API"""
#     if not st.session_state.ollama_client:
#         return None, "Ollama client not initialized"
    
#     try:
#         response = st.session_state.ollama_client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a business intelligence analyst specializing in YouTube revenue analysis. Provide clear, data-driven insights with specific recommendations."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=1500
#         )
#         return response.choices[0].message.content, None
#     except Exception as e:
#         return None, f"Ollama error: {str(e)}"

# def generate_business_prompt(user_query, label_videos, monthly_revenue, est_total, actual_total, rpm, selected_label):
#     """Generate business intelligence prompt"""
#     # Calculate key metrics
#     total_videos = len(label_videos)
#     avg_views = label_videos['view_count'].mean() if not label_videos.empty else 0
#     top_video_views = label_videos['view_count'].max() if not label_videos.empty else 0
#     accuracy = (est_total / actual_total * 100) if actual_total > 0 else 0
    
#     # Get top performing videos
#     top_videos = label_videos.nlargest(5, 'view_count')[['title', 'view_count', 'Estimated Revenue INR']]
    
#     # Monthly trend summary
#     monthly_summary = ""
#     if not monthly_revenue.empty:
#         monthly_summary = f"Monthly data shows {len(monthly_revenue)} months of activity, with peak revenue of ₹{monthly_revenue['Estimated Revenue INR'].max():,.0f}"
    
#     prompt = f"""
#     Analyze the YouTube revenue data for {selected_label} and answer: "{user_query}"

#     KEY METRICS:
#     - Record Label: {selected_label}
#     - Total Videos: {total_videos:,}
#     - Estimated Revenue: ₹{est_total:,.0f}
#     - Actual Revenue: ₹{actual_total:,.0f}
#     - Revenue Accuracy: {accuracy:.1f}%
#     - RPM (Revenue per Million): ₹{rpm:,}
#     - Average Views per Video: {avg_views:,.0f}
#     - Top Video Views: {top_video_views:,.0f}

#     TOP PERFORMING VIDEOS:
#     {top_videos.to_string(index=False) if not top_videos.empty else "No video data available"}

#     MONTHLY PERFORMANCE:
#     {monthly_summary}

#     PREVIOUS QUESTIONS:
#     {chr(10).join(f"- {q}" for q in st.session_state.query_history[-3:]) if st.session_state.query_history else "None"}

#     Please provide:
#     1. Direct answer to the user's question
#     2. Key insights from the data
#     3. Specific recommendations for improving revenue
#     4. Suggestions for visualizations that would be helpful
    
#     Keep the response concise and actionable.
#     """
#     return prompt

# def render_suggested_visualizations(analysis_text, label_videos, monthly_revenue):
#     """Render visualizations based on AI analysis suggestions"""
#     if label_videos.empty:
#         return
    
#     text_lower = analysis_text.lower()
    
#     # Top performing videos
#     if any(keyword in text_lower for keyword in ["top videos", "best performing", "highest views", "performance"]):
#         st.subheader("🔥 Top Performing Videos")
#         top_videos = label_videos.nlargest(10, 'view_count')
#         fig = px.bar(top_videos, x='view_count', y='title', orientation='h',
#                     title="Top 10 Videos by Views")
#         st.plotly_chart(fig, use_container_width=True)
    
#     # Revenue vs Views correlation
#     if any(keyword in text_lower for keyword in ["correlation", "relationship", "views vs revenue"]):
#         st.subheader("📊 Views vs Revenue Correlation")
#         fig = px.scatter(label_videos, x='view_count', y='Estimated Revenue INR',
#                         title="Views vs Estimated Revenue",
#                         hover_data=['title'])
#         st.plotly_chart(fig, use_container_width=True)
    
#     # Monthly trends
#     if any(keyword in text_lower for keyword in ["monthly", "trend", "over time", "seasonal"]) and not monthly_revenue.empty:
#         st.subheader("📈 Monthly Revenue Trends")
#         fig = px.line(monthly_revenue, x='Month', y='Estimated Revenue INR',
#                      title="Monthly Revenue Trend", markers=True)
#         st.plotly_chart(fig, use_container_width=True)
    
#     # Revenue distribution
#     if any(keyword in text_lower for keyword in ["distribution", "histogram", "spread"]):
#         st.subheader("📊 Revenue Distribution")
#         fig = px.histogram(label_videos, x='Estimated Revenue INR', nbins=20,
#                           title="Revenue Distribution Across Videos")
#         st.plotly_chart(fig, use_container_width=True)

# # Load backend metadata
# youtube_df = load_youtube_metadata()

# # --- UI Header ---
# st.title("🧠 Smart Data Analyzer Pro — YouTube Revenue Validation")
# st.markdown("Upload a **Revenue CSV file** below and compare estimated vs reported revenue for any record label.")

# # Sidebar for Ollama status and settings
# with st.sidebar:
#     st.header("🤖 AI Analysis Settings")
    
#     # Check Ollama status
#     ollama_status = check_ollama_status()
#     if ollama_status:
#         st.success("✅ Ollama is running")
#     else:
#         st.error("❌ Ollama not detected")
#         st.info("Start Ollama with: `ollama serve`")
    
#     # Model selection
#     available_models = ["mistral", "llama2", "codellama", "neural-chat"]
#     selected_model = st.selectbox("Select AI Model", available_models)
    
#     # Analysis options
#     auto_analyze = st.checkbox("Auto-analyze data", value=True)
#     show_suggestions = st.checkbox("Show visualization suggestions", value=True)
    
#     st.markdown("---")
#     st.markdown("**Recent Questions:**")
#     for i, q in enumerate(reversed(st.session_state.query_history[-3:]), 1):
#         st.caption(f"{i}. {q[:50]}...")

# # --- File Upload ---
# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# # --- If CSV is uploaded ---
# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        
#         # Show data preview
#         with st.expander("📋 Data Preview"):
#             st.dataframe(df.head(), use_container_width=True)

#         if not youtube_df.empty and "view_count" in youtube_df.columns:
#             st.subheader("🎙️ Select Record Label for Analysis")

#             # Create columns for better layout
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 # Unique label selection
#                 unique_labels = youtube_df["Record Label"].unique().tolist()
#                 selected_label = st.selectbox("Choose a record label:", sorted(unique_labels))
            
#             with col2:
#                 rpm = st.number_input("💸 Revenue per Million Views (INR)", 
#                                     min_value=500, max_value=500000, value=1200, step=100)

#             # Filter videos for selected label
#             label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#             label_videos = label_videos.dropna(subset=["view_count"])
#             label_videos["view_count"] = pd.to_numeric(label_videos["view_count"], errors='coerce')
#             label_videos = label_videos.dropna(subset=["view_count"])
#             label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#             est_total = label_videos["Estimated Revenue INR"].sum()

#             # Match against uploaded CSV actuals
#             actual_total = 0
#             if "Store Name" in df.columns:
#                 yt_row = df[df["Store Name"].str.lower() == "youtube"]
#                 if not yt_row.empty and "Annual Revenue in INR" in df.columns:
#                     actual_total = yt_row["Annual Revenue in INR"].values[0]
#                     accuracy = est_total / actual_total if actual_total else 0
#                     verdict = "✅ Close Match" if 0.8 <= accuracy <= 1.2 else "❌ Significant Difference"

#                     # Display metrics
#                     col1, col2, col3 = st.columns(3)
#                     col1.metric(f"📊 Total Videos", f"{len(label_videos):,}")
#                     col2.metric(f"Estimated Revenue ({selected_label})", f"₹{est_total:,.0f}")
#                     col3.metric("Reported Revenue (CSV)", f"₹{actual_total:,.0f}")
                    
#                     st.markdown(f"### 🔍 Analysis: {verdict} — Accuracy: {accuracy:.1%}")

#                     # Chart comparison
#                     comparison_data = pd.DataFrame({
#                         'Source': [f'Estimated ({selected_label})', 'Actual (CSV)'],
#                         'Revenue': [est_total, actual_total]
#                     })
                    
#                     fig = px.bar(comparison_data, x='Source', y='Revenue',
#                                 title=f"📊 Revenue Comparison for {selected_label}",
#                                 color='Source')
#                     st.plotly_chart(fig, use_container_width=True)

#                     # Monthly breakdown
#                     if "published_at" in label_videos.columns:
#                         label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#                         label_videos = label_videos.dropna(subset=["published_at"])
#                         label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#                         monthly_revenue = (
#                             label_videos.groupby("Month")["Estimated Revenue INR"]
#                             .sum().reset_index().sort_values("Month")
#                         )

#                         # Monthly chart
#                         if not monthly_revenue.empty:
#                             st.subheader(f"📆 Monthly Revenue Trend: {selected_label}")
#                             fig_month = px.bar(monthly_revenue, x="Month", y="Estimated Revenue INR",
#                                              title=f"Monthly Revenue Estimate: {selected_label}")
#                             st.plotly_chart(fig_month, use_container_width=True)
#                     else:
#                         monthly_revenue = pd.DataFrame()

#                     # AI Analysis Section
#                     st.markdown("---")
#                     st.subheader("🧠 AI-Powered Business Intelligence")
                    
#                     # Auto-analyze if enabled
#                     if auto_analyze and ollama_status:
#                         with st.spinner("🤖 Analyzing data..."):
#                             auto_prompt = generate_business_prompt(
#                                 "Provide a comprehensive analysis of this YouTube revenue data with key insights and recommendations",
#                                 label_videos, monthly_revenue, est_total, actual_total, rpm, selected_label
#                             )
#                             analysis, error = get_ollama_analysis(auto_prompt, selected_model)
                            
#                             if analysis:
#                                 st.markdown("### 📊 Automated Analysis")
#                                 st.markdown(analysis)
                                
#                                 if show_suggestions:
#                                     render_suggested_visualizations(analysis, label_videos, monthly_revenue)
#                             elif error:
#                                 st.error(f"Analysis failed: {error}")
                    
#                     # Interactive Q&A
#                     st.markdown("### 💬 Ask Questions About Your Data")
#                     user_question = st.text_area(
#                         "What would you like to know about this revenue data?",
#                         placeholder="e.g., Which videos generate the most revenue per view? What's the seasonal trend?",
#                         height=100
#                     )
                    
#                     if user_question and st.button("🔍 Analyze", type="primary"):
#                         if ollama_status:
#                             st.session_state.query_history.append(user_question)
#                             st.session_state.query_history = st.session_state.query_history[-10:]  # Keep last 10
                            
#                             with st.spinner("🤖 Analyzing your question..."):
#                                 prompt = generate_business_prompt(
#                                     user_question, label_videos, monthly_revenue, 
#                                     est_total, actual_total, rpm, selected_label
#                                 )
#                                 analysis, error = get_ollama_analysis(prompt, selected_model)
                                
#                                 if analysis:
#                                     st.markdown("### 🎯 Analysis Results")
#                                     st.markdown(analysis)
                                    
#                                     if show_suggestions:
#                                         render_suggested_visualizations(analysis, label_videos, monthly_revenue)
#                                 elif error:
#                                     st.error(f"Analysis failed: {error}")
#                         else:
#                             st.error("❌ Ollama is not running. Please start Ollama server first.")

#                     # Data export section
#                     with st.expander(f"📄 Export {selected_label} Data"):
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             if st.button("📥 Download Video Data"):
#                                 csv_data = label_videos[["title", "view_count", "Estimated Revenue INR", "Month"]].to_csv(index=False)
#                                 st.download_button(
#                                     "Download CSV",
#                                     csv_data,
#                                     f"{selected_label}_video_data.csv",
#                                     "text/csv"
#                                 )
                        
#                         with col2:
#                             if st.button("📊 Download Monthly Data") and not monthly_revenue.empty:
#                                 monthly_csv = monthly_revenue.to_csv(index=False)
#                                 st.download_button(
#                                     "Download Monthly CSV",
#                                     monthly_csv,
#                                     f"{selected_label}_monthly_revenue.csv",
#                                     "text/csv"
#                                 )
                        
#                         # Show detailed data
#                         st.dataframe(
#                             label_videos[["title", "view_count", "Estimated Revenue INR", "Month"]],
#                             use_container_width=True
#                         )
#                 else:
#                     st.warning("⚠️ YouTube revenue not found in your uploaded CSV or missing 'Annual Revenue in INR' column.")
#             else:
#                 st.warning("⚠️ 'Store Name' column is missing in your CSV.")
#         else:
#             st.warning("⚠️ YouTube metadata not loaded or missing 'view_count' column.")
#     except Exception as e:
#         st.error(f"❌ Error processing file: {e}")
#         st.exception(e)
# else:
#     st.info("📁 Upload a revenue CSV file to get started.")
    
#     # Show instructions
#     st.markdown("""
#     ### 📋 How to Use:
#     1. **Start Ollama**: Run `ollama serve` in your terminal
#     2. **Install Models**: Run `ollama pull mistral` (or your preferred model)
#     3. **Upload CSV**: Your CSV should have columns like 'Store Name' and 'Annual Revenue in INR'
#     4. **Ask Questions**: Use the AI assistant to analyze your YouTube revenue data
    
#     ### 📊 Required Files:
#     - `youtube_metadata.json` - Contains video metadata with view counts
#     - Revenue CSV file with YouTube data
#     """)












# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import json
# import warnings
# import requests

# warnings.filterwarnings("ignore")

# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Revenue Validation")

# @st.cache_data
# def load_youtube_metadata():
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)

#         df = pd.DataFrame(data["videos"]) if isinstance(data, dict) and "videos" in data else pd.DataFrame(data)

#         KNOWN_LABELS = [
#             "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#             "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"
#         ]

#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#             for label in KNOWN_LABELS:
#                 if label.lower() in text:
#                     return label
#             return "Other"

#         df["Record Label"] = df.apply(detect_label, axis=1)
#         return df

#     except Exception as e:
#         st.error(f"❌ Error loading YouTube metadata: {e}")
#         return pd.DataFrame()

# youtube_df = load_youtube_metadata()

# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
#         st.dataframe(df.head(), use_container_width=True)

#         if not youtube_df.empty and "view_count" in youtube_df.columns:
#             st.subheader("🎙️ Select Record Label for Analysis")
#             selected_label = st.selectbox("Choose a record label:", sorted(youtube_df["Record Label"].unique()))
#             rpm = st.number_input("💸 Revenue per Million Views (INR)", min_value=500, value=1200)

#             label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#             label_videos = label_videos.dropna(subset=["view_count"])
#             label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#             est_total = label_videos["Estimated Revenue INR"].sum()

#             if "Store Name" in df.columns:
#                 yt_row = df[df["Store Name"].str.lower() == "youtube"]
#                 if not yt_row.empty:
#                     actual_total = yt_row["Annual Revenue in INR"].values[0]
#                     accuracy = est_total / actual_total if actual_total else 0
#                     verdict = "✅ Match" if 0.8 <= accuracy <= 1.2 else "❌ Mismatch"

#                     col1, col2 = st.columns(2)
#                     col1.metric(f"Estimated Revenue ({selected_label})", f"₹{est_total:,.0f}")
#                     col2.metric("Reported Revenue (CSV)", f"₹{actual_total:,.0f}")
#                     st.markdown(f"### 🔍 Verdict: {verdict} — Accuracy: {accuracy:.2%}")

#                     fig = px.bar(
#                         x=[f"Estimated ({selected_label})", "Actual (CSV)"],
#                         y=[est_total, actual_total],
#                         labels={"x": "Source", "y": "Revenue (INR)"},
#                         title=f"📊 Revenue Comparison for {selected_label}"
#                     )
#                     st.plotly_chart(fig, use_container_width=True)

#                     label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#                     label_videos = label_videos.dropna(subset=["published_at"])
#                     label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#                     monthly_revenue = (
#                         label_videos.groupby("Month")["Estimated Revenue INR"]
#                         .sum().reset_index().sort_values("Month")
#                     )

#                     st.subheader(f"📆 Monthly Estimated Revenue: {selected_label}")
#                     st.dataframe(monthly_revenue, use_container_width=True)

#                     fig_month = px.bar(
#                         monthly_revenue,
#                         x="Month",
#                         y="Estimated Revenue INR",
#                         title=f"📈 Monthly Revenue Estimate: {selected_label}",
#                         labels={"Estimated Revenue INR": "INR"}
#                     )
#                     st.plotly_chart(fig_month, use_container_width=True)

#                     with st.expander(f"📄 {selected_label} Video Metadata"):
#                         st.dataframe(
#                             label_videos[["title", "view_count", "Estimated Revenue INR", "Month"]],
#                             use_container_width=True
#                         )

#                     # --- Smart Query Section ---
#                     st.subheader("💬 Ask a Business Intelligence Question")

#                     suggested_queries = [
#                         "Why is there a mismatch between estimated and actual revenue?",
#                         "Which months had the highest revenue gap?",
#                         "List top 5 videos by estimated revenue.",
#                         "What is the average revenue per video?",
#                         "Compare monthly estimated revenue and highlight anomalies.",
#                         "Which videos contributed the least to total revenue?"
#                     ]

#                     selected_query = st.selectbox("📌 Try a suggested question:", ["(Choose one)"] + suggested_queries)
#                     query = st.text_area("Or type your own question below:", value=selected_query if selected_query != "(Choose one)" else "")

#                     if query:
#                         with st.spinner("🧠 Thinking with Mistral..."):
#                             try:
#                                 # Clean prompt inputs
#                                 label_videos["title"] = label_videos["title"].astype(str).str.slice(0, 100)
#                                 video_data_json = label_videos[["title", "view_count", "Estimated Revenue INR", "Month"]].head(10).to_json(orient="records")
#                                 revenue_data_json = df.head(10).to_json(orient="records")

#                                 prompt = f"""
# You are a business intelligence expert with access to two datasets:

# YouTube Video Data:
# {video_data_json}

# CSV Revenue Data:
# {revenue_data_json}

# User's Question:
# \"\"\"{query}\"\"\"

# Analyze revenue estimation vs reported values. Identify gaps, explain mismatches, and perform logical or Excel-style operations. Back your analysis with data where relevant.
# """

#                                 response = requests.post(
#                                     "http://127.0.0.1:11434/api/generate",
#                                     json={
#                                         "model": "mistral",  # Change to "mistral:instruct" if needed
#                                         "prompt": prompt.strip(),
#                                         "stream": False
#                                     },
#                                     timeout=60
#                                 )

#                                 if response.status_code == 200:
#                                     result = response.json()
#                                     st.markdown(f"### 🧠 Insight:\n{result['response']}")
#                                 else:
#                                     st.error(f"⚠️ Mistral error: {response.status_code}")
#                                     st.code(response.text, language='json')
#                             except Exception as e:
#                                 st.error(f"❌ Failed to connect to Mistral: {e}")
#                 else:
#                     st.warning("⚠️ YouTube revenue not found in your uploaded CSV.")
#             else:
#                 st.warning("⚠️ 'Store Name' column missing in your CSV.")
#         else:
#             st.warning("⚠️ YouTube metadata not loaded or missing 'view_count'.")
#     except Exception as e:
#         st.error(f"❌ Error processing file: {e}")
# else:
#     st.info("Upload a revenue CSV file to get started.")







# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import json
# import requests
# from dotenv import load_dotenv
# import os
# from datetime import datetime

# # Load env variables
# load_dotenv()
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Revenue Validation")

# @st.cache_data
# def load_youtube_metadata():
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)
#         df = pd.DataFrame(data["videos"]) if isinstance(data, dict) and "videos" in data else pd.DataFrame(data)
#         labels = ["T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#                   "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"]
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#             for label in labels:
#                 if label.lower() in text:
#                     return label
#             return "Other"
#         df["Record Label"] = df.apply(detect_label, axis=1)
#         return df
#     except Exception as e:
#         st.error(f"❌ Failed to load YouTube metadata: {e}")
#         return pd.DataFrame()

# def get_mistral_analysis(prompt, api_key, max_retries=3):
#     if not api_key:
#         return generate_fallback_analysis(prompt), "❌ OpenRouter API key not provided in .env"
#     for _ in range(max_retries):
#         try:
#             headers = {
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json"
#             }
#             payload = {
#                 "model": "mistralai/mistral-7b-instruct:free",
#                 "messages": [
#                     {"role": "system", "content": "You are a business intelligence analyst."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 "temperature": 0.7,
#                 "max_tokens": 1500,
#                 "top_p": 0.9
#             }
#             response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#             if response.status_code == 200:
#                 content = response.json()["choices"][0]["message"]["content"]
#                 return content.strip(), None
#         except Exception as e:
#             last_error = str(e)
#     return generate_fallback_analysis(prompt), f"Failed after retries: {last_error}"

# def generate_fallback_analysis(prompt):
#     return f"""
# 🧠 **Fallback Summary for Prompt**:
# > {prompt}

# - Revenue likely fluctuates with seasonal demand.
# - Top-performing videos should be studied for strategy replication.
# - Review actual-vs-estimate regularly to fine-tune RPM.
# """

# def generate_business_context(videos, est_total, actual_total, label, monthly):
#     top = videos.loc[videos["Estimated Revenue INR"].idxmax()] if not videos.empty else {}
#     low = videos.loc[videos["Estimated Revenue INR"].idxmin()] if not videos.empty else {}
#     return f"""
# **Label:** {label}
# **Estimated Revenue:** ₹{est_total:,.2f}
# **Actual Revenue:** ₹{actual_total:,.2f}
# **Top Video:** {top.get('title', 'N/A')} (₹{top.get('Estimated Revenue INR', 0):,.0f})
# **Lowest Video:** {low.get('title', 'N/A')} (₹{low.get('Estimated Revenue INR', 0):,.0f})
# **Active Months:** {len(monthly)}
# """

# # Load metadata
# youtube_df = load_youtube_metadata()
# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.success(f"✅ Uploaded {len(df)} rows")

#     st.subheader("🎙️ Select Record Label")
#     selected_label = st.selectbox("Choose a record label:", sorted(youtube_df["Record Label"].unique()))
#     rpm = st.number_input("💸 Revenue per Million Views", value=1200, min_value=500)

#     label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#     label_videos = label_videos.dropna(subset=["view_count"])
#     label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#     est_total = label_videos["Estimated Revenue INR"].sum()

#     yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)] if "Store Name" in df.columns else pd.DataFrame()
#     actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0

#     # Monthly data
#     label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#     label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#     monthly = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()

#     # KPIs
#     st.subheader("📊 Key Metrics")
#     col1, col2, col3, col4 = st.columns(4)
#     with col1: st.metric("Estimated Revenue", f"₹{est_total:,.0f}")
#     with col2: st.metric("Actual Revenue", f"₹{actual_total:,.0f}")
#     with col3: st.metric("Accuracy", f"{(est_total / actual_total):.1%}" if actual_total else "N/A")
#     with col4: st.metric("Variance", f"₹{abs(est_total - actual_total):,.0f}")

#     # Graphs
#     tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "🏆 Top Videos", "📈 Monthly Trend", "💰 Distribution"])

#     with tab1:
#         fig = go.Figure()
#         fig.add_bar(name="Estimated", x=["YouTube"], y=[est_total])
#         fig.add_bar(name="Actual", x=["YouTube"], y=[actual_total])
#         fig.update_layout(title="Estimated vs Actual Revenue", barmode="group")
#         st.plotly_chart(fig, use_container_width=True)

#     with tab2:
#         top_videos = label_videos.nlargest(10, "Estimated Revenue INR")[["title", "view_count", "Estimated Revenue INR"]]
#         fig = px.bar(top_videos, x="Estimated Revenue INR", y="title", orientation="h", title="Top 10 Revenue Videos")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(top_videos)

#     with tab3:
#         fig = px.line(monthly, x="Month", y="Estimated Revenue INR", title="Monthly Revenue Trend", markers=True)
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(monthly)

#     with tab4:
#         fig = px.histogram(label_videos, x="Estimated Revenue INR", nbins=20, title="Revenue Distribution")
#         st.plotly_chart(fig, use_container_width=True)

#     # Smart Prompt
#     st.subheader("🧠 Ask a Business Intelligence Question")
#     query = st.text_area("Example: What seasonal trends do you observe?")
#     if query:
#         if not OPENROUTER_API_KEY:
#             st.error("❌ No API key found in .env")
#         else:
#             with st.spinner("🔍 Mistral AI is analyzing..."):
#                 context = generate_business_context(label_videos, est_total, actual_total, selected_label, monthly)
#                 prompt = f"{context}\n\nQuestion: {query}"
#                 result, err = get_mistral_analysis(prompt, OPENROUTER_API_KEY)
#                 if err:
#                     st.warning(err)
#                 st.markdown("### 🎯 Insight:")
#                 st.markdown(result)

#     # Export
#     st.subheader("📤 Export")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.download_button("📋 Download Video Data (CSV)", label_videos.to_csv(index=False), "video_data.csv", "text/csv")
#     with col2:
#         summary = {
#             "label": selected_label,
#             "estimated": est_total,
#             "actual": actual_total,
#             "accuracy": (est_total / actual_total) if actual_total else None
#         }
#         st.download_button("💾 Download Summary (JSON)", json.dumps(summary, indent=2), "summary.json", "application/json")
# else:
#     st.info("📁 Upload a CSV to begin.")


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import json
# import requests
# import spacy
# from prophet import Prophet
# from prophet.plot import plot_components_plotly
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
# nlp = spacy.load("en_core_web_sm")

# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Revenue Validation")

# # Initialize session state
# if "query_history" not in st.session_state:
#     st.session_state.query_history = []

# @st.cache_data
# def load_youtube_metadata():
#     with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#         data = json.load(f)
#     df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)

#     KNOWN_LABELS = [
#         "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#         "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"
#     ]

#     def detect_label(row):
#         text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#         for label in KNOWN_LABELS:
#             if label.lower() in text:
#                 return label
#         return "Other"

#     df["Record Label"] = df.apply(detect_label, axis=1)
#     return df

# def generate_prompt(user_query, label_videos, monthly, est_total, actual_total, rpm, history=None):
#     history_text = ""
#     if history:
#         history_text = "\n\nPrevious Questions:\n" + "\n".join(f"- {q}" for q in history)

#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     video_sample = label_videos[["title", "view_count", "Estimated Revenue INR", "Month", "RPV_Estimated"]].head(10)
#     monthly_sample = monthly.head(10)

#     return f"""
# You are a business analyst with access to structured YouTube revenue data.

# Here is real data:

# ▶️ Video data (top 10 rows):
# {video_sample.to_json(orient="records", indent=2)}

# 📅 Monthly revenue (last 10 rows):
# {monthly_sample.to_json(orient="records", indent=2)}

# Total revenue: ₹{est_total:,.2f}
# Actual reported: ₹{actual_total:,.2f}
# RPM: ₹{rpm}

# {history_text}

# ---

# User question: "{user_query}"

# Respond using real numbers above. Do NOT hallucinate tables or charts. Suggest what should be plotted or tabulated, but do not make up fake structures. My app will create the visuals based on your suggestion.
# """

# def get_mistral_analysis(prompt, api_key):
#     try:
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 1500
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}"
#     except Exception as e:
#         return "", str(e)

# def prophet_forecast(monthly_df, forecast_months=6, show_components=False):
#     df = monthly_df.copy()
#     df = df.rename(columns={"Month": "ds", "Estimated Revenue INR": "y"})
#     df["ds"] = pd.to_datetime(df["ds"])

#     model = Prophet()
#     model.add_country_holidays(country_name='IN')  # Optional: Add Indian holidays
#     model.fit(df)

#     future = model.make_future_dataframe(periods=forecast_months, freq='M')
#     forecast = model.predict(future)

#     forecast_filtered = forecast[forecast["ds"] > df["ds"].max()]
#     forecast_filtered = forecast_filtered[["ds", "yhat"]].rename(columns={"ds": "Month", "yhat": "Forecasted Revenue INR"})

#     line_fig = px.line(forecast_filtered, x="Month", y="Forecasted Revenue INR", title=f"🔮 Prophet Forecast: Next {forecast_months} Months")

#     component_fig = None
#     if show_components:
#         component_fig = plot_components_plotly(model, forecast)

#     return line_fig, component_fig

# def render_visuals_from_keywords(text, videos, monthly, top_videos, forecast_months):
#     text_lower = text.lower()

#     if any(k in text_lower for k in ["monthly", "season", "trend", "timeline", "growth"]):
#         st.subheader("📈 Revenue Trend")
#         fig = px.area(monthly, x="Month", y="Estimated Revenue INR", title="Revenue Over Time", markers=True)
#         st.plotly_chart(fig, use_container_width=True)

#     if any(k in text_lower for k in ["top", "highest", "rank", "best", "rpv"]):
#         st.subheader("🏆 Top Videos by RPV")
#         fig = px.bar(top_videos, x="RPV_Estimated", y="title", orientation="h")
#         st.plotly_chart(fig, use_container_width=True)

#     if any(k in text_lower for k in ["distribution", "spread", "variance", "range"]):
#         st.subheader("📊 Revenue Distribution")
#         fig = px.histogram(videos, x="Estimated Revenue INR", nbins=20)
#         st.plotly_chart(fig, use_container_width=True)

#     if any(k in text_lower for k in ["correlation", "relationship", "compare", "views vs revenue"]):
#         st.subheader("📉 Views vs Revenue Correlation")
#         fig = px.scatter(videos, x="view_count", y="Estimated Revenue INR", size="RPV_Estimated")
#         st.plotly_chart(fig, use_container_width=True)

#     if any(k in text_lower for k in ["forecast", "projection", "next 6 months", "future"]):
#         st.subheader(f"🔮 Revenue Forecast ({forecast_months} Months)")
#         forecast_fig, components_fig = prophet_forecast(monthly, forecast_months, show_components=True)
#         st.plotly_chart(forecast_fig, use_container_width=True)

#         if components_fig:
#             st.subheader("📊 Prophet Forecast Components")
#             st.plotly_chart(components_fig, use_container_width=True)

#     if any(k in text_lower for k in ["share", "portion", "ratio", "split"]):
#         st.subheader("🥧 Revenue Share (Top 5 Videos)")
#         top_5 = videos.nlargest(5, "Estimated Revenue INR")
#         fig = px.pie(top_5, names="title", values="Estimated Revenue INR")
#         st.plotly_chart(fig, use_container_width=True)

# # --- Main App Logic ---

# youtube_df = load_youtube_metadata()
# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     selected_label = st.selectbox("Choose Record Label", sorted(youtube_df["Record Label"].unique()))
#     rpm = st.number_input("💸 RPM (INR)", min_value=500, value=225000)
#     forecast_months = st.selectbox("📆 Forecast Horizon (Months)", [3, 6, 12], index=1)

#     label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#     label_videos = label_videos.dropna(subset=["view_count"])
#     label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#     est_total = label_videos["Estimated Revenue INR"].sum()

#     actual_total = 0
#     if "Store Name" in df.columns:
#         yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#         if not yt_row.empty:
#             actual_total = yt_row["Annual Revenue in INR"].values[0]

#     label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#     label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#     monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     top_rpv = label_videos.nlargest(10, "RPV_Estimated")[["title", "view_count", "Estimated Revenue INR", "RPV_Estimated"]]

#     st.metric("Estimated Revenue", f"₹{est_total:,.0f}")
#     st.metric("Actual Revenue", f"₹{actual_total:,.0f}")
#     st.metric("Accuracy", f"{(est_total / actual_total):.2%}" if actual_total else "N/A")

#     user_query = st.text_area("🧠 Ask a Business Intelligence Question")

#     if user_query:
#         st.session_state.query_history.append(user_query)
#         st.session_state.query_history = st.session_state.query_history[-5:]

#         with st.spinner("Mistral AI is analyzing..."):
#             prompt = generate_prompt(
#                 user_query,
#                 label_videos,
#                 monthly_revenue,
#                 est_total,
#                 actual_total,
#                 rpm,
#                 history=st.session_state.query_history[:-1]
#             )
#             response, error = get_mistral_analysis(prompt, API_KEY)
#             if error:
#                 st.error(error)
#             else:
#                 st.markdown("### 🧠 Mistral Insight")
#                 st.markdown(response)
#                 render_visuals_from_keywords(response, label_videos, monthly_revenue, top_rpv, forecast_months)

#     if st.session_state.query_history:
#         st.markdown("#### 📚 Recent Questions")
#         for q in reversed(st.session_state.query_history):
#             st.markdown(f"- {q}")

#     st.download_button("📥 Export Video Data", label_videos.to_csv(index=False), "videos.csv")
# else:
#     st.info("📁 Upload a revenue CSV to begin.")

###########   Working ok........#############

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# # Load NLP model
# nlp = spacy.load("en_core_web_sm")

# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Revenue Validation")

# @st.cache_data
# def load_youtube_metadata():
#     with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#         data = json.load(f)
#     df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
#     known_labels = ["T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#                     "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"]
    
#     def detect_label(row):
#         text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#         for label in known_labels:
#             if label.lower() in text:
#                 return label
#         return "Other"
    
#     df["Record Label"] = df.apply(detect_label, axis=1)
#     return df

# def detect_months_and_confidence(text):
#     months = [
#         "january", "february", "march", "april", "may", "june",
#         "july", "august", "september", "october", "november", "december"
#     ]
#     found = {token.text for token in nlp(text.lower()) if token.text in months}
#     confidence = len(found) / 12
#     return list(found), confidence

# def render_visuals_from_keywords(text, videos, monthly, top_videos):
#     text_lower = text.lower()
#     months, conf = detect_months_and_confidence(text)
#     if conf >= 0.25:
#         st.subheader("📈 Monthly Revenue Trend")
#         st.caption(f"Confidence: {conf:.2f} — Months detected: {', '.join(months)}")
#         fig = px.line(monthly, x="Month", y="Estimated Revenue INR", markers=True)
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(monthly)

#     if "top" in text_lower or "rpv" in text_lower:
#         st.subheader("🏆 Top Videos by RPV")
#         fig = px.bar(top_videos, x="RPV_Estimated", y="title", orientation="h")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(top_videos)

# def generate_prompt(user_query, label_videos, monthly, est_total, actual_total, rpm):
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     video_sample = label_videos[["title", "view_count", "Estimated Revenue INR", "Month", "RPV_Estimated"]].head(10)
#     monthly_sample = monthly.head(10)

#     accuracy_str = f"{est_total / actual_total:.2%}" if actual_total else "N/A"

#     return f"""
# You are a senior Business Intelligence analyst embedded inside a live dashboard.

# You have access to the following structured data:

# ▶️ **Video Data Sample (Top 10):**
# {video_sample.to_json(orient="records", indent=2)}

# 📆 **Monthly Revenue (Top 10):**
# {monthly_sample.to_json(orient="records", indent=2)}

# 📊 **Business Summary**:
# - RPM: ₹{rpm}
# - Total Estimated Revenue: ₹{est_total:,.2f}
# - Actual Reported Revenue: ₹{actual_total:,.2f}
# - Accuracy: {accuracy_str}
# - Total Videos: {len(label_videos)}

# ---

# ### User Question:
# "{user_query}"

# ---

# Respond with data-driven insights only. Use real numbers from the input. Highlight metrics. Recommend specific actions. Do not hallucinate.
# Use markdown formatting with sections and bullet points.
# """

# def get_mistral_analysis(prompt, api_key):
#     try:
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 1500
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}"
#     except Exception as e:
#         return "", str(e)

# # === Main App ===
# youtube_df = load_youtube_metadata()
# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     selected_label = st.selectbox("🎙️ Choose Record Label", sorted(youtube_df["Record Label"].unique()))
#     rpm = st.number_input("💸 RPM (Revenue per Million Views)", min_value=500, value=1200)

#     label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#     label_videos = label_videos.dropna(subset=["view_count"])
#     label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#     est_total = label_videos["Estimated Revenue INR"].sum()

#     if "Store Name" in df.columns:
#         yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#         actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
#     else:
#         actual_total = 0

#     label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#     label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#     monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     top_rpv = label_videos.nlargest(10, "RPV_Estimated")[["title", "view_count", "Estimated Revenue INR", "RPV_Estimated"]]

#     st.metric("Estimated Revenue", f"₹{est_total:,.0f}")
#     st.metric("Actual Revenue", f"₹{actual_total:,.0f}")
#     st.metric("Accuracy", f"{(est_total / actual_total):.2%}" if actual_total else "N/A")

#     user_query = st.text_area("🧠 Ask a Business Intelligence Question")
#     if user_query:
#         with st.spinner("Mistral AI is analyzing..."):
#             full_prompt = generate_prompt(user_query, label_videos, monthly_revenue, est_total, actual_total, rpm)
#             response, error = get_mistral_analysis(full_prompt, API_KEY)
#             if error:
#                 st.error(error)
#             else:
#                 st.markdown("### 🧠 Mistral Insight")
#                 st.markdown(response)
#                 render_visuals_from_keywords(response, label_videos, monthly_revenue, top_rpv)

#     st.download_button("📥 Export Video Data", label_videos.to_csv(index=False), "videos.csv")
# else:
#     st.info("📁 Upload a revenue CSV to get started.")

#################################

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import json
# import requests
# import spacy
# from prophet import Prophet
# from prophet.plot import plot_components_plotly
# import os
# import time
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# # Load NLP model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     st.warning("SpaCy model 'en_core_web_sm' not found. Install it with: python -m spacy download en_core_web_sm")
#     nlp = None

# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Revenue Validation")

# # Initialize session state
# if "query_history" not in st.session_state:
#     st.session_state.query_history = []

# @st.cache_data
# def load_youtube_metadata():
#     """Load YouTube metadata with error handling"""
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)
#         df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
#     except FileNotFoundError:
#         st.error("youtube_metadata.json file not found. Please ensure the file exists in the same directory.")
#         return pd.DataFrame()
#     except json.JSONDecodeError as e:
#         st.error(f"Error parsing JSON file: {e}")
#         return pd.DataFrame()

#     KNOWN_LABELS = [
#         "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#         "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"
#     ]

#     def detect_label(row):
#         text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#         for label in KNOWN_LABELS:
#             if label.lower() in text:
#                 return label
#         return "Other"

#     if not df.empty:
#         df["Record Label"] = df.apply(detect_label, axis=1)
#     return df

# def generate_prompt(user_query, label_videos, monthly, est_total, actual_total, rpm, history=None):
#     """Generate enhanced prompt for AI analysis"""
#     history_text = ""
#     if history:
#         history_text = "\n\nPrevious Questions:\n" + "\n".join(f"- {q}" for q in history[-3:])  # Limit to last 3

#     if label_videos.empty:
#         return f"User question: '{user_query}'\n\nNo video data available. Please provide insights on YouTube revenue analysis in general."

#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / (label_videos["view_count"] + 1e-9)
#     video_sample = label_videos[["title", "view_count", "Estimated Revenue INR", "Month", "RPV_Estimated"]].head(5)
#     monthly_sample = monthly.head(5) if not monthly.empty else pd.DataFrame()

#     # Calculate metrics safely
#     avg_views = label_videos['view_count'].mean() if not label_videos.empty else 0
    
#     return f"""
# You are a business analyst with access to structured YouTube revenue data.

# Here is real data:

# ▶️ Video data (top 5 rows):
# {video_sample.to_json(orient="records", indent=2) if not video_sample.empty else "No video data"}

# 📅 Monthly revenue (last 5 rows):
# {monthly_sample.to_json(orient="records", indent=2) if not monthly_sample.empty else "No monthly data"}

# 📊 Key Metrics:
# - Total estimated revenue: ₹{est_total:,.2f}
# - Actual reported revenue: ₹{actual_total:,.2f}
# - RPM (Revenue Per Mille): ₹{rpm:,.0f}
# - Total videos analyzed: {len(label_videos)}
# - Average views per video: {avg_views:,.0f}

# {history_text}

# ---

# User question: "{user_query}"

# Provide a concise, data-driven analysis using the real numbers above. Focus on actionable insights and suggest specific visualizations that would help answer the question.
# """

# def create_requests_session():
#     """Create a requests session with retry strategy"""
#     session = requests.Session()
#     retry_strategy = Retry(
#         total=3,
#         backoff_factor=1,
#         status_forcelist=[429, 500, 502, 503, 504],
#     )
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     session.mount("http://", adapter)
#     session.mount("https://", adapter)
#     return session

# def get_ollama_analysis(prompt: str, model_name: str = "mistral", timeout: int = 120):
#     """Get analysis from Ollama with improved error handling and timeout"""
#     try:
#         session = create_requests_session()
#         url = "http://localhost:11434/api/generate"
#         headers = {"Content-Type": "application/json"}
#         payload = {
#             "model": model_name,
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "temperature": 0.7,
#                 "top_p": 0.9,
#                 "max_tokens": 1000
#             }
#         }
        
#         # Check if Ollama is running
#         try:
#             health_check = session.get("http://localhost:11434/api/version", timeout=5)
#             if health_check.status_code != 200:
#                 return "", "Ollama service is not responding. Please ensure Ollama is running."
#         except requests.exceptions.RequestException:
#             return "", "Cannot connect to Ollama. Please ensure Ollama is installed and running on localhost:11434"
        
#         response = session.post(url, json=payload, headers=headers, timeout=timeout)

#         if response.status_code == 200:
#             result = response.json()
#             return result.get("response", ""), None
#         else:
#             return "", f"Ollama Error {response.status_code}: {response.text}"
            
#     except requests.exceptions.Timeout:
#         return "", f"Request timed out after {timeout} seconds. Try reducing the data size or increasing timeout."
#     except requests.exceptions.ConnectionError:
#         return "", "Connection error. Please check if Ollama is running on localhost:11434"
#     except Exception as e:
#         return "", f"Unexpected error: {str(e)}"

# def get_fallback_analysis(user_query, label_videos, monthly_revenue, est_total, actual_total):
#     """Provide fallback analysis when Ollama is not available"""
#     # Calculate metrics safely
#     accuracy = (est_total/actual_total*100) if actual_total > 0 else 0
#     avg_views = label_videos['view_count'].mean() if not label_videos.empty else 0
#     max_views = label_videos['view_count'].max() if not label_videos.empty else 0
#     peak_revenue = monthly_revenue['Estimated Revenue INR'].max() if not monthly_revenue.empty else 0
    
#     analysis = f"""
#     ## 📊 Quick Analysis for: "{user_query}"
    
#     **Revenue Overview:**
#     - Estimated Total Revenue: ₹{est_total:,.0f}
#     - Actual Revenue: ₹{actual_total:,.0f}
#     - Accuracy: {accuracy:.1f}% {"" if actual_total > 0 else "(No actual data)"}
    
#     **Content Performance:**
#     - Total Videos: {len(label_videos)}
#     - Average Views: {avg_views:,.0f}
#     - Top Video Views: {max_views:,.0f}
    
#     **Monthly Trends:**
#     - Active Months: {len(monthly_revenue)}
#     - Peak Month Revenue: ₹{peak_revenue:,.0f}
    
#     *Note: Ollama AI analysis unavailable. Basic metrics shown above.*
#     """
#     return analysis

# def prophet_forecast(monthly_df, forecast_months=6, show_components=False):
#     """Generate Prophet forecast with error handling"""
#     try:
#         if monthly_df.empty or len(monthly_df) < 2:
#             st.warning("Insufficient data for forecasting (need at least 2 data points)")
#             return None, None
            
#         df = monthly_df.copy()
#         df = df.rename(columns={"Month": "ds", "Estimated Revenue INR": "y"})
#         df["ds"] = pd.to_datetime(df["ds"])
#         df = df.sort_values("ds")

#         model = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=False,
#             daily_seasonality=False
#         )
        
#         try:
#             model.add_country_holidays(country_name='IN')
#         except:
#             pass  # Continue without holidays if not available
            
#         model.fit(df)

#         future = model.make_future_dataframe(periods=forecast_months, freq='M')
#         forecast = model.predict(future)

#         forecast_filtered = forecast[forecast["ds"] > df["ds"].max()]
#         forecast_filtered = forecast_filtered[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
#             columns={"ds": "Month", "yhat": "Forecasted Revenue INR", 
#                     "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}
#         )

#         line_fig = px.line(forecast_filtered, x="Month", y="Forecasted Revenue INR", 
#                           title=f"🔮 Prophet Forecast: Next {forecast_months} Months")
#         line_fig.add_scatter(x=forecast_filtered["Month"], y=forecast_filtered["Lower Bound"], 
#                            mode='lines', name='Lower Bound', line=dict(dash='dash'))
#         line_fig.add_scatter(x=forecast_filtered["Month"], y=forecast_filtered["Upper Bound"], 
#                            mode='lines', name='Upper Bound', line=dict(dash='dash'))

#         component_fig = None
#         if show_components:
#             try:
#                 component_fig = plot_components_plotly(model, forecast)
#             except:
#                 st.warning("Could not generate forecast components")

#         return line_fig, component_fig
        
#     except Exception as e:
#         st.error(f"Forecasting error: {str(e)}")
#         return None, None

# def render_visuals_from_keywords(text, videos, monthly, top_videos, forecast_months):
#     """Render visualizations based on keywords in the analysis"""
#     if videos.empty:
#         st.warning("No video data available for visualization")
#         return
        
#     text_lower = text.lower()

#     if any(k in text_lower for k in ["monthly", "season", "trend", "timeline", "growth", "over time"]):
#         if not monthly.empty:
#             st.subheader("📈 Revenue Trend Over Time")
#             fig = px.area(monthly, x="Month", y="Estimated Revenue INR", 
#                          title="Monthly Revenue Trend", markers=True)
#             fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (INR)")
#             st.plotly_chart(fig, use_container_width=True)

#     if any(k in text_lower for k in ["top", "highest", "rank", "best", "rpv", "performance"]):
#         if not top_videos.empty:
#             st.subheader("🏆 Top Videos by Revenue Per View (RPV)")
#             top_display = top_videos.head(10).copy()
#             top_display["Short Title"] = top_display["title"].str.slice(0, 40) + "..."
#             fig = px.bar(top_display, x="RPV_Estimated", y="Short Title", 
#                         orientation="h", title="Top 10 Videos by RPV")
#             st.plotly_chart(fig, use_container_width=True)

#     if any(k in text_lower for k in ["distribution", "spread", "variance", "range", "histogram"]):
#         st.subheader("📊 Revenue Distribution")
#         fig = px.histogram(videos, x="Estimated Revenue INR", nbins=20, 
#                           title="Revenue Distribution Across Videos")
#         st.plotly_chart(fig, use_container_width=True)

#     if any(k in text_lower for k in ["correlation", "relationship", "compare", "views vs revenue", "scatter"]):
#         st.subheader("📉 Views vs Revenue Correlation")
#         fig = px.scatter(videos, x="view_count", y="Estimated Revenue INR", 
#                         size="RPV_Estimated", title="Views vs Revenue Relationship",
#                         hover_data=["title"])
#         st.plotly_chart(fig, use_container_width=True)

#     if any(k in text_lower for k in ["forecast", "projection", "future", "predict"]):
#         if not monthly.empty:
#             st.subheader(f"🔮 Revenue Forecast ({forecast_months} Months)")
#             forecast_fig, components_fig = prophet_forecast(monthly, forecast_months, show_components=True)
#             if forecast_fig:
#                 st.plotly_chart(forecast_fig, use_container_width=True)
#                 if components_fig:
#                     st.subheader("📊 Forecast Components")
#                     st.plotly_chart(components_fig, use_container_width=True)

#     if any(k in text_lower for k in ["share", "portion", "ratio", "split", "pie"]):
#         st.subheader("🥧 Revenue Share (Top 10 Videos)")
#         top_10 = videos.nlargest(10, "Estimated Revenue INR").copy()
#         top_10["Short Title"] = top_10["title"].str.slice(0, 30) + "..."
#         fig = px.pie(top_10, names="Short Title", values="Estimated Revenue INR",
#                     title="Revenue Share Distribution")
#         st.plotly_chart(fig, use_container_width=True)

# # --- Main App Logic ---

# # Sidebar for configuration
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     ollama_timeout = st.number_input("Ollama Timeout (seconds)", min_value=30, max_value=300, value=120)
#     use_fallback = st.checkbox("Use fallback analysis if Ollama fails", value=True)
    
#     st.header("📊 Data Info")
#     if st.button("🔄 Refresh Data"):
#         st.cache_data.clear()
#         st.rerun()

# youtube_df = load_youtube_metadata()
# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         st.success(f"✅ Loaded CSV with {len(df)} rows")
#     except Exception as e:
#         st.error(f"Error reading file: {e}")
#         st.stop()

#     # Configuration section
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         label_options = ["All"] + sorted(youtube_df["Record Label"].unique()) if not youtube_df.empty else ["All"]
#         selected_label = st.selectbox("🏷️ Choose Record Label", label_options)
    
#     with col2:
#         rpm = st.number_input("💸 RPM (Revenue Per Mille INR)", 
#                              min_value=100, max_value=1000000, value=225000, step=1000)
    
#     with col3:
#         forecast_months = st.selectbox("📆 Forecast Horizon", [3, 6, 12], index=1)

#     # Process data
#     if not youtube_df.empty:
#         if selected_label != "All":
#             label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#         else:
#             label_videos = youtube_df.copy()

#         label_videos = label_videos.dropna(subset=["view_count"])
#         label_videos["view_count"] = pd.to_numeric(label_videos["view_count"], errors='coerce')
#         label_videos = label_videos.dropna(subset=["view_count"])
        
#         label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#         est_total = label_videos["Estimated Revenue INR"].sum()

#         # Get actual revenue from uploaded CSV
#         actual_total = 0
#         if "Store Name" in df.columns and "Annual Revenue in INR" in df.columns:
#             yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#             if not yt_row.empty:
#                 actual_total = yt_row["Annual Revenue in INR"].values[0]

#         # Create monthly aggregation
#         if "published_at" in label_videos.columns:
#             label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#             label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#             monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
#             monthly_revenue = monthly_revenue.sort_values("Month")
#         else:
#             monthly_revenue = pd.DataFrame()

#         # Calculate RPV
#         label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / (label_videos["view_count"] + 1e-9)
#         top_rpv = label_videos.nlargest(10, "RPV_Estimated")[["title", "view_count", "Estimated Revenue INR", "RPV_Estimated"]]

#         # Display metrics
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("📊 Total Videos", f"{len(label_videos):,}")
#         with col2:
#             st.metric("💰 Estimated Revenue", f"₹{est_total:,.0f}")
#         with col3:
#             st.metric("📈 Actual Revenue", f"₹{actual_total:,.0f}")
#         with col4:
#             accuracy = (est_total / actual_total * 100) if actual_total > 0 else 0
#             st.metric("🎯 Accuracy", f"{accuracy:.1f}%")

#         # Query interface
#         st.markdown("---")
#         user_query = st.text_area("🧠 Ask a Business Intelligence Question", 
#                                  placeholder="e.g., How has YouTube revenue changed over the months? Which videos perform best?",
#                                  height=100)

#         if user_query:
#             st.session_state.query_history.append(user_query)
#             st.session_state.query_history = st.session_state.query_history[-5:]  # Keep last 5

#             with st.spinner("🧠 Analyzing your question..."):
#                 prompt = generate_prompt(
#                     user_query,
#                     label_videos,
#                     monthly_revenue,
#                     est_total,
#                     actual_total,
#                     rpm,
#                     history=st.session_state.query_history[:-1]
#                 )
                
#                 response, error = get_ollama_analysis(prompt, model_name="mistral", timeout=ollama_timeout)
                
#                 if error:
#                     st.error(f"🚫 Ollama Error: {error}")
#                     if use_fallback:
#                         st.info("📋 Using fallback analysis...")
#                         response = get_fallback_analysis(user_query, label_videos, monthly_revenue, est_total, actual_total)
#                 else:
#                     st.success("✅ Analysis completed successfully!")

#                 if response:
#                     st.markdown("### 🧠 AI Analysis")
#                     st.markdown(response)
                    
#                     # Generate visualizations based on the response
#                     render_visuals_from_keywords(response, label_videos, monthly_revenue, top_rpv, forecast_months)

#         # Query history
#         if st.session_state.query_history:
#             with st.expander("📚 Recent Questions"):
#                 for i, q in enumerate(reversed(st.session_state.query_history), 1):
#                     st.markdown(f"{i}. {q}")

#         # Export functionality
#         st.markdown("---")
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("📥 Export Video Data"):
#                 csv_data = label_videos.to_csv(index=False)
#                 st.download_button("Download CSV", csv_data, "youtube_analysis.csv", "text/csv")
#         with col2:
#             if st.button("📊 Export Summary Report"):
#                 summary = f"""
# # YouTube Revenue Analysis Report

# ## Overview
# - **Record Label**: {selected_label}
# - **Total Videos**: {len(label_videos):,}
# - **Estimated Revenue**: ₹{est_total:,.0f}
# - **Actual Revenue**: ₹{actual_total:,.0f}
# - **RPM Used**: ₹{rpm:,}

# ## Top Performing Videos (by RPV)
# {top_rpv.to_string(index=False)}

# ## Monthly Revenue Trend
# {monthly_revenue.to_string(index=False) if not monthly_revenue.empty else "No monthly data available"}
#                 """
#                 st.download_button("Download Report", summary, "revenue_report.md", "text/markdown")
#     else:
#         st.warning("⚠️ No YouTube metadata loaded. Please check your youtube_metadata.json file.")
        
# else:
#     st.info("📁 Please upload a revenue CSV file to begin analysis.")
#     st.markdown("""
#     ### 📋 Expected CSV Format:
#     - Should contain columns: `Store Name`, `Annual Revenue in INR`
#     - YouTube row should have 'youtube' in the Store Name
    
#     ### 📂 Required Files:
#     - `youtube_metadata.json` - Contains video metadata
#     - Revenue CSV file (uploaded through interface)
#     """)

##### Completely Working with context layers

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# # Load NLP model
# nlp = spacy.load("en_core_web_sm")

# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Revenue Validation")

# # Initialize session state for context memory
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

# def add_to_conversation_history(query, response, insights=None, charts_generated=None):
#     """Add query and response to conversation history with metadata"""
#     entry = {
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'query': query,
#         'response': response,
#         'insights': insights or [],
#         'charts_generated': charts_generated or [],
#         'session_id': st.session_state.session_id
#     }
#     st.session_state.conversation_history.append(entry)
    
#     # Keep only last 10 conversations to manage memory
#     if len(st.session_state.conversation_history) > 10:
#         st.session_state.conversation_history = st.session_state.conversation_history[-10:]

# def update_analysis_context(key_insights):
#     """Update persistent analysis context with key findings"""
#     for insight in key_insights:
#         category = insight.get('category', 'general')
#         if category not in st.session_state.analysis_context:
#             st.session_state.analysis_context[category] = []
#         st.session_state.analysis_context[category].append({
#             'insight': insight.get('text', ''),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'confidence': insight.get('confidence', 0.5)
#         })

# def get_conversation_context():
#     """Generate conversation context for AI prompt"""
#     if not st.session_state.conversation_history:
#         return ""
    
#     context = "\n📝 **Previous Conversation Context:**\n"
#     for i, entry in enumerate(st.session_state.conversation_history[-5:], 1):  # Last 5 conversations
#         context += f"\n**Q{i}:** {entry['query']}\n"
#         context += f"**A{i}:** {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}\n"
#         if entry['insights']:
#             context += f"**Key Insights:** {', '.join([insight.get('text', '')[:50] for insight in entry['insights']])}\n"
    
#     # Add persistent context
#     if st.session_state.analysis_context:
#         context += "\n🧠 **Key Analysis Context:**\n"
#         for category, insights in st.session_state.analysis_context.items():
#             latest_insight = insights[-1] if insights else {}
#             context += f"- **{category.title()}:** {latest_insight.get('insight', '')[:100]}\n"
    
#     return context

# def extract_insights_from_response(response):
#     """Extract key insights from AI response for context building"""
#     insights = []
    
#     # Simple keyword-based insight extraction
#     response_lower = response.lower()
    
#     if any(word in response_lower for word in ['revenue', 'income', 'earnings']):
#         insights.append({'category': 'revenue', 'text': 'Revenue analysis discussed', 'confidence': 0.8})
    
#     if any(word in response_lower for word in ['trend', 'growth', 'decline']):
#         insights.append({'category': 'trends', 'text': 'Trend analysis provided', 'confidence': 0.7})
    
#     if any(word in response_lower for word in ['recommend', 'suggest', 'should']):
#         insights.append({'category': 'recommendations', 'text': 'Recommendations provided', 'confidence': 0.9})
    
#     if any(word in response_lower for word in ['month', 'seasonal', 'quarterly']):
#         insights.append({'category': 'temporal', 'text': 'Temporal analysis conducted', 'confidence': 0.6})
    
#     return insights

# @st.cache_data
# def load_youtube_metadata():
#     with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#         data = json.load(f)
#     df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
#     known_labels = ["T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#                     "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"]
    
#     def detect_label(row):
#         text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#         for label in known_labels:
#             if label.lower() in text:
#                 return label
#         return "Other"
    
#     df["Record Label"] = df.apply(detect_label, axis=1)
#     return df

# def detect_months_and_confidence(text):
#     months = [
#         "january", "february", "march", "april", "may", "june",
#         "july", "august", "september", "october", "november", "december"
#     ]
#     found = {token.text for token in nlp(text.lower()) if token.text in months}
#     confidence = len(found) / 12
#     return list(found), confidence

# def render_visuals_from_keywords(text, videos, monthly, top_videos):
#     text_lower = text.lower()
#     charts_generated = []
    
#     months, conf = detect_months_and_confidence(text)
#     if conf >= 0.25:
#         st.subheader("📈 Monthly Revenue Trend")
#         st.caption(f"Confidence: {conf:.2f} — Months detected: {', '.join(months)}")
#         fig = px.line(monthly, x="Month", y="Estimated Revenue INR", markers=True)
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(monthly)
#         charts_generated.append("Monthly Revenue Trend")

#     if "top" in text_lower or "rpv" in text_lower:
#         st.subheader("🏆 Top Videos by RPV")
#         fig = px.bar(top_videos, x="RPV_Estimated", y="title", orientation="h")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(top_videos)
#         charts_generated.append("Top Videos by RPV")
    
#     return charts_generated

# def generate_enhanced_prompt(user_query, label_videos, monthly, est_total, actual_total, rpm):
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     video_sample = label_videos[["title", "view_count", "Estimated Revenue INR", "Month", "RPV_Estimated"]].head(10)
#     monthly_sample = monthly.head(10)

#     accuracy_str = f"{est_total / actual_total:.2%}" if actual_total else "N/A"
    
#     # Get conversation context
#     conversation_context = get_conversation_context()

#     return f"""
# You are a senior Business Intelligence analyst embedded inside a live dashboard with context memory.

# {conversation_context}

# ▶️ **Current Video Data Sample (Top 10):**
# {video_sample.to_json(orient="records", indent=2)}

# 📆 **Monthly Revenue (Top 10):**
# {monthly_sample.to_json(orient="records", indent=2)}

# 📊 **Current Business Summary**:
# - RPM: ₹{rpm}
# - Total Estimated Revenue: ₹{est_total:,.2f}
# - Actual Reported Revenue: ₹{actual_total:,.2f}
# - Accuracy: {accuracy_str}
# - Total Videos: {len(label_videos)}

# ---

# ### Current User Question:
# "{user_query}"

# ---

# IMPORTANT INSTRUCTIONS:
# 1. **Reference Previous Context**: Build upon previous questions and insights when relevant
# 2. **Provide Continuity**: If this question relates to previous queries, acknowledge and extend the analysis
# 3. **Use Context**: Reference previous findings, trends, or recommendations when applicable
# 4. **Avoid Repetition**: Don't repeat identical insights from previous responses
# 5. **Progressive Analysis**: Deepen the analysis based on conversation history

# Respond with data-driven insights only. Use real numbers from the input. Highlight metrics. 
# Recommend specific actions. Do not hallucinate. Use markdown formatting with sections and bullet points.

# If this is a follow-up question, explicitly connect it to previous analysis and build upon it.
# """

# def get_mistral_analysis(prompt, api_key):
#     try:
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 1500
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}"
#     except Exception as e:
#         return "", str(e)

# # === Sidebar for Context Management ===
# with st.sidebar:
#     st.header("🧠 Context Memory")
    
#     if st.button("🔄 Clear Conversation History"):
#         st.session_state.conversation_history = []
#         st.session_state.analysis_context = {}
#         st.success("Context cleared!")
    
#     st.subheader("📊 Session Stats")
#     st.metric("Queries Asked", len(st.session_state.conversation_history))
#     st.metric("Session ID", st.session_state.session_id)
    
#     if st.session_state.conversation_history:
#         st.subheader("💬 Recent Queries")
#         for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
#             with st.expander(f"Query {len(st.session_state.conversation_history) - 3 + i}"):
#                 st.write(f"**Q:** {entry['query'][:100]}...")
#                 st.write(f"**Time:** {entry['timestamp']}")
#                 if entry['charts_generated']:
#                     st.write(f"**Charts:** {', '.join(entry['charts_generated'])}")

# # === Main App ===
# youtube_df = load_youtube_metadata()
# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     selected_label = st.selectbox("🎙️ Choose Record Label", sorted(youtube_df["Record Label"].unique()))
#     rpm = st.number_input("💸 RPM (Revenue per Million Views)", min_value=500, value=125000)

#     label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#     label_videos = label_videos.dropna(subset=["view_count"])
#     label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#     est_total = label_videos["Estimated Revenue INR"].sum()

#     if "Store Name" in df.columns:
#         yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#         actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
#     else:
#         actual_total = 0

#     label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#     label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#     monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     top_rpv = label_videos.nlargest(10, "RPV_Estimated")[["title", "view_count", "Estimated Revenue INR", "RPV_Estimated"]]

#     # Display metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Estimated Revenue", f"₹{est_total:,.0f}")
#     with col2:
#         st.metric("Actual Revenue", f"₹{actual_total:,.0f}")
#     with col3:
#         st.metric("Accuracy", f"{(est_total / actual_total):.2%}" if actual_total else "N/A")

#     # Enhanced query interface
#     st.subheader("🧠 Ask a Business Intelligence Question")
    
#     # Show context hint
#     if st.session_state.conversation_history:
#         last_query = st.session_state.conversation_history[-1]['query']
#         st.info(f"💡 **Context Available** - Last query: '{last_query[:50]}...' - Ask follow-up questions for deeper insights!")
    
#     user_query = st.text_area("Your question:", placeholder="Ask about trends, comparisons, recommendations, or follow up on previous insights...")
    
#     # Quick suggestion buttons based on context
#     if st.session_state.conversation_history:
#         st.write("**Quick Follow-ups:**")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             if st.button("📈 Deep dive into this"):
#                 user_query = f"Can you provide a deeper analysis of the previous insight?"
#         with col2:
#             if st.button("🔍 What's next?"):
#                 user_query = f"Based on the previous analysis, what should be the next steps?"
#         with col3:
#             if st.button("📊 Compare trends"):
#                 user_query = f"How do these findings compare to industry benchmarks?"
    
#     if user_query:
#         with st.spinner("🧠 Mistral AI is analyzing with context..."):
#             full_prompt = generate_enhanced_prompt(user_query, label_videos, monthly_revenue, est_total, actual_total, rpm)
#             response, error = get_mistral_analysis(full_prompt, API_KEY)
            
#             if error:
#                 st.error(error)
#             else:
#                 st.markdown("### 🧠 Mistral Insight")
#                 st.markdown(response)
                
#                 # Generate visuals and track them
#                 charts_generated = render_visuals_from_keywords(response, label_videos, monthly_revenue, top_rpv)
                
#                 # Extract insights and update context
#                 insights = extract_insights_from_response(response)
#                 update_analysis_context(insights)
                
#                 # Add to conversation history
#                 add_to_conversation_history(user_query, response, insights, charts_generated)
                
#                 # Show connection to previous context
#                 if len(st.session_state.conversation_history) > 1:
#                     with st.expander("🔗 Context Connection"):
#                         st.write("This analysis builds upon:")
#                         for insight in insights:
#                             st.write(f"- {insight['category'].title()}: {insight['text']}")

#     st.download_button("📥 Export Video Data", label_videos.to_csv(index=False), "videos.csv")
    
#     # Export conversation history
#     if st.session_state.conversation_history:
#         conversation_export = pd.DataFrame(st.session_state.conversation_history)
#         st.download_button(
#             "📥 Export Conversation History", 
#             conversation_export.to_csv(index=False), 
#             f"conversation_history_{st.session_state.session_id}.csv"
#         )

# else:
#     st.info("📁 Upload a revenue CSV to get started.")
    
#     # Show welcome message with context capabilities
#     st.markdown("""
#     ### 🧠 Context Memory Features:
#     - **Conversation History**: Remembers your previous questions and insights
#     - **Progressive Analysis**: Each query builds upon previous findings
#     - **Context Awareness**: AI understands the flow of your analysis
#     - **Session Persistence**: Maintains context throughout your session
#     - **Export Conversations**: Download your analysis journey
#     """)

###### Smart Data Analyzer Pro — Full App with Context, Derived Columns, and LLM
###### Smart Data Analyzer Pro — Full App (Fast Version, No spaCy NER in Artist Extraction)

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import re
# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 0  # For deterministic langdetect

# # Language mapping for readable names
# LANG_CODE_TO_NAME = {
#     'en': 'English', 'hi': 'Hindi', 'pa': 'Punjabi', 'mr': 'Marathi', 'gu': 'Gujarati', 'bn': 'Bengali',
#     'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'kn': 'Kannada', 'ur': 'Urdu', 'sa': 'Sanskrit',
#     'ne': 'Nepali', 'or': 'Odia', 'as': 'Assamese', 'kok': 'Konkani', 'sd': 'Sindhi', 'other': 'Other',
# }

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# # Load NLP model (optional; still used for month detection)
# nlp = spacy.load("en_core_web_sm")

# st.set_page_config(page_title="Smart Data Analyzer Pro", layout="wide")
# st.title("Hungama BI - Smart Data Analyzer ")

# # ---------------- Context Layers ----------------
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

# def add_to_conversation_history(query, response, insights=None, charts_generated=None):
#     entry = {
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'query': query,
#         'response': response,
#         'insights': insights or [],
#         'charts_generated': charts_generated or [],
#         'session_id': st.session_state.session_id
#     }
#     st.session_state.conversation_history.append(entry)
#     if len(st.session_state.conversation_history) > 10:
#         st.session_state.conversation_history = st.session_state.conversation_history[-10:]

# def update_analysis_context(key_insights):
#     for insight in key_insights:
#         category = insight.get('category', 'general')
#         if category not in st.session_state.analysis_context:
#             st.session_state.analysis_context[category] = []
#         st.session_state.analysis_context[category].append({
#             'insight': insight.get('text', ''),
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'confidence': insight.get('confidence', 0.5)
#         })

# def get_conversation_context():
#     if not st.session_state.conversation_history:
#         return ""
#     context = "\n📝 **Previous Conversation Context:**\n"
#     for i, entry in enumerate(st.session_state.conversation_history[-5:], 1):
#         context += f"\n**Q{i}:** {entry['query']}\n"
#         context += f"**A{i}:** {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}\n"
#         if entry['insights']:
#             context += f"**Key Insights:** {', '.join([insight.get('text', '')[:50] for insight in entry['insights']])}\n"
#     if st.session_state.analysis_context:
#         context += "\n🧠 **Key Analysis Context:**\n"
#         for category, insights in st.session_state.analysis_context.items():
#             latest_insight = insights[-1] if insights else {}
#             context += f"- **{category.title()}:** {latest_insight.get('insight', '')[:100]}\n"
#     return context

# def extract_insights_from_response(response):
#     insights = []
#     response_lower = response.lower()
#     if any(word in response_lower for word in ['revenue', 'income', 'earnings']):
#         insights.append({'category': 'revenue', 'text': 'Revenue analysis discussed', 'confidence': 0.8})
#     if any(word in response_lower for word in ['trend', 'growth', 'decline']):
#         insights.append({'category': 'trends', 'text': 'Trend analysis provided', 'confidence': 0.7})
#     if any(word in response_lower for word in ['recommend', 'suggest', 'should']):
#         insights.append({'category': 'recommendations', 'text': 'Recommendations provided', 'confidence': 0.9})
#     if any(word in response_lower for word in ['month', 'seasonal', 'quarterly']):
#         insights.append({'category': 'temporal', 'text': 'Temporal analysis conducted', 'confidence': 0.6})
#     return insights

# @st.cache_data
# def load_youtube_metadata():
#     with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#         data = json.load(f)
#     df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
#     if df.empty:
#         return df

#     known_labels = [
#         "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#         "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"
#     ]
#     def detect_label(row):
#         if "channel_name" in row and row["channel_name"]:
#             return row["channel_name"]
#         text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#         for label in known_labels:
#             if label.lower() in text:
#                 return label
#         return "Other"
#     df["Record Label"] = df.apply(detect_label, axis=1)

#     # Published at (datetime and derived)
#     df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
#     df["Year"] = df["published_at"].dt.year
#     df["Month"] = df["published_at"].dt.strftime('%B')
#     df["Date"] = df["published_at"].dt.date
#     df["Time"] = df["published_at"].dt.strftime('%H:%M:%S')
#     df["Day Name"] = df["published_at"].dt.day_name()
#     print(df[["published_at", "Year", "Month", "Date", "Time", "Day Name"]])
#     # Artists (fast! no spaCy, just regex and hashtags)
#     def extract_artists(row):
#         text = f"{row.get('title', '')} {row.get('description', '')}"
#         hashtags = re.findall(r"#(\w+)", text)
#         artist_patterns = re.findall(r"(singer[s]*|sung by|featuring|starring|artist[s]*):? ([\w\s&,.']+)", text, re.I)
#         artists = []
#         for match in artist_patterns:
#             parts = re.split(r",|&|and", match[1])
#             artists.extend([x.strip() for x in parts if len(x.strip()) > 2 and not x.strip().isdigit()])
#         artist_hashtags = [tag for tag in hashtags if any(
#             key in tag.lower() for key in [
#                 'singh', 'khan', 'rao', 'kaur', 'ali', 'kapoor', 'chopra', 'anwar', 'amod', 'pooja',
#                 'kk', 'pritam', 'sunidhi', 'arijit', 'shilpa', 'ayushmann', 'akshay', 'kumar', 'katrina',
#                 'kaif', 'neha', 'benny', 'nakkash', 'aditi', 'anupam', 'harshdeep', 'loh', 'amod'
#             ]
#         ) or len(tag) > 5]
#         all_artists = set(a.title() for a in artists if a) | set(artist_hashtags)
#         return ", ".join(sorted(all_artists)) if all_artists else ""
#     df["Artists"] = df.apply(extract_artists, axis=1)

#     # Language (hashtags then langdetect, readable)
#     def guess_language(row):
#         text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#         hashtags = re.findall(r"#(\w+)", text)
#         for tag in hashtags:
#             for code, name in LANG_CODE_TO_NAME.items():
#                 if code != 'other' and code in tag:
#                     return name
#                 if name.lower() in tag:
#                     return name
#         try:
#             detected = detect(text)
#             return LANG_CODE_TO_NAME.get(detected, 'Other')
#         except:
#             return 'Other'
#     df["Language"] = df.apply(guess_language, axis=1)

#     return df

# def detect_months_and_confidence(text):
#     months = [
#         "january", "february", "march", "april", "may", "june",
#         "july", "august", "september", "october", "november", "december"
#     ]
#     found = {token.text for token in nlp(text.lower()) if token.text in months}
#     confidence = len(found) / 12
#     return list(found), confidence

# def render_visuals_from_keywords(text, videos, monthly, top_videos):
#     text_lower = text.lower()
#     charts_generated = []
#     months, conf = detect_months_and_confidence(text)
#     if conf >= 0.25:
#         st.subheader("📈 Monthly Revenue Trend")
#         st.caption(f"Confidence: {conf:.2f} — Months detected: {', '.join(months)}")
#         fig = px.line(monthly, x="Month", y="Estimated Revenue INR", markers=True)
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(monthly)
#         charts_generated.append("Monthly Revenue Trend")
#     if "top" in text_lower or "rpv" in text_lower:
#         st.subheader("🏆 Top Videos by RPV")
#         fig = px.bar(top_videos, x="RPV_Estimated", y="title", orientation="h")
#         st.plotly_chart(fig, use_container_width=True)
#         st.dataframe(top_videos)
#         charts_generated.append("Top Videos by RPV")
#     return charts_generated

# def generate_enhanced_prompt(user_query, label_videos, monthly, est_total, actual_total, rpm):
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     video_sample = label_videos[["title", "view_count", "Estimated Revenue INR", "Month", "RPV_Estimated"]].head(10)
#     monthly_sample = monthly.head(10)
#     accuracy_str = f"{est_total / actual_total:.2%}" if actual_total else "N/A"
#     conversation_context = get_conversation_context()
#     return f"""
# You are a senior Business Intelligence analyst embedded inside a live dashboard with context memory.

# {conversation_context}

# ▶️ **Current Video Data Sample (Top 10):**
# {video_sample.to_json(orient="records", indent=2)}

# 📆 **Monthly Revenue (Top 10):**
# {monthly_sample.to_json(orient="records", indent=2)}

# 📊 **Current Business Summary**:
# - RPM: ₹{rpm}
# - Total Estimated Revenue: ₹{est_total:,.2f}
# - Actual Reported Revenue: ₹{actual_total:,.2f}
# - Accuracy: {accuracy_str}
# - Total Videos: {len(label_videos)}

# ---

# ### Current User Question:
# "{user_query}"

# ---

# IMPORTANT INSTRUCTIONS:
# 1. **Reference Previous Context**: Build upon previous questions and insights when relevant
# 2. **Provide Continuity**: If this question relates to previous queries, acknowledge and extend the analysis
# 3. **Use Context**: Reference previous findings, trends, or recommendations when applicable
# 4. **Avoid Repetition**: Don't repeat identical insights from previous responses
# 5. **Progressive Analysis**: Deepen the analysis based on conversation history

# Respond with data-driven insights only. Use real numbers from the input. Highlight metrics. 
# Recommend specific actions. Do not hallucinate. Use markdown formatting with sections and bullet points.

# If this is a follow-up question, explicitly connect it to previous analysis and build upon it.
# """

# def get_mistral_analysis(prompt, api_key):
#     try:
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 1500
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}"
#     except Exception as e:
#         return "", str(e)

# # === Sidebar for Context Management ===
# with st.sidebar:
#     st.header("🧠 Context Memory")
#     if st.button("🔄 Clear Conversation History"):
#         st.session_state.conversation_history = []
#         st.session_state.analysis_context = {}
#         st.success("Context cleared!")
#     st.subheader("📊 Session Stats")
#     st.metric("Queries Asked", len(st.session_state.conversation_history))
#     st.metric("Session ID", st.session_state.session_id)
#     if st.session_state.conversation_history:
#         st.subheader("💬 Recent Queries")
#         for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
#             with st.expander(f"Query {len(st.session_state.conversation_history) - 3 + i}"):
#                 st.write(f"**Q:** {entry['query'][:100]}...")
#                 st.write(f"**Time:** {entry['timestamp']}")
#                 if entry['charts_generated']:
#                     st.write(f"**Charts:** {', '.join(entry['charts_generated'])}")

# # === Main App ===
# youtube_df = load_youtube_metadata()
# uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     selected_label = st.selectbox("🎙️ Choose Record Label", sorted(youtube_df["Record Label"].unique()))
#     rpm = st.number_input("💸 RPM (Revenue per Million Views)", min_value=500, value=125000)

#     label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#     label_videos = label_videos.dropna(subset=["view_count"])
#     label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
#     est_total = label_videos["Estimated Revenue INR"].sum()

#     if "Store Name" in df.columns:
#         yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#         actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
#     else:
#         actual_total = 0

#     label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#     label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#     monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
#     label_videos["RPV_Estimated"] = label_videos["Estimated Revenue INR"] / label_videos["view_count"]
#     top_rpv = label_videos.nlargest(10, "RPV_Estimated")[["title", "view_count", "Estimated Revenue INR", "RPV_Estimated"]]

#     # Display metrics
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Estimated Revenue", f"₹{est_total:,.0f}")
#     with col2:
#         st.metric("Actual Revenue", f"₹{actual_total:,.0f}")
#     with col3:
#         st.metric("Accuracy", f"{(est_total / actual_total):.2%}" if actual_total else "N/A")

#     # Show derived columns for quick filter (optional UI, or comment out if not needed)
#     with st.expander("🔎 Filter videos by derived columns (Artists, Language, Year, Month, Day)"):
#         artist_options = sorted({a.strip() for aa in label_videos["Artists"].dropna() for a in aa.split(",") if a.strip()})
#         lang_options = sorted(label_videos["Language"].dropna().unique())
#         year_options = sorted(label_videos["Year"].dropna().unique())
#         month_options = sorted(label_videos["Month"].dropna().unique())
#         day_options = sorted(label_videos["Day Name"].dropna().unique())

#         selected_artists = st.multiselect("🎤 Artist(s)", artist_options)
#         selected_langs = st.multiselect("🌐 Language(s)", lang_options)
#         selected_years = st.multiselect("📅 Year(s)", year_options)
#         selected_months = st.multiselect("🗓️ Month(s)", month_options)
#         selected_days = st.multiselect("📆 Day(s) of Week", day_options)

#         # Filter based on selections
#         if selected_artists:
#             regex = "|".join([re.escape(a) for a in selected_artists])
#             label_videos = label_videos[label_videos["Artists"].str.contains(regex, case=False, na=False)]
#         if selected_langs:
#             label_videos = label_videos[label_videos["Language"].isin(selected_langs)]
#         if selected_years:
#             label_videos = label_videos[label_videos["Year"].isin(selected_years)]
#         if selected_months:
#             label_videos = label_videos[label_videos["Month"].isin(selected_months)]
#         if selected_days:
#             label_videos = label_videos[label_videos["Day Name"].isin(selected_days)]

#     # Enhanced query interface
#     st.subheader("🧠 Ask a Business Intelligence Question")
#     if st.session_state.conversation_history:
#         last_query = st.session_state.conversation_history[-1]['query']
#         st.info(f"💡 **Context Available** - Last query: '{last_query[:50]}...' - Ask follow-up questions for deeper insights!")
#     user_query = st.text_area(
#         "Your question:",
#         placeholder="Ask about trends, comparisons, recommendations, languages, artists, months, or follow up on previous insights..."
#     )

#     # Quick suggestion buttons based on context
#     if st.session_state.conversation_history:
#         st.write("**Quick Follow-ups:**")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             if st.button("📈 Deep dive into this"):
#                 user_query = "Can you provide a deeper analysis of the previous insight?"
#         with col2:
#             if st.button("🔍 What's next?"):
#                 user_query = "Based on the previous analysis, what should be the next steps?"
#         with col3:
#             if st.button("📊 Compare trends"):
#                 user_query = "How do these findings compare to industry benchmarks?"

#     if user_query:
#         with st.spinner("🧠 Hungama BI is analyzing with context..."):
#             full_prompt = generate_enhanced_prompt(
#                 user_query, label_videos, monthly_revenue, est_total, actual_total, rpm
#             )
#             response, error = get_mistral_analysis(full_prompt, API_KEY)

#             if error:
#                 st.error(error)
#             else:
#                 st.markdown("### 🧠 Hungama BI Insight")
#                 st.markdown(response)
#                 charts_generated = render_visuals_from_keywords(
#                     response, label_videos, monthly_revenue, top_rpv
#                 )
#                 insights = extract_insights_from_response(response)
#                 update_analysis_context(insights)
#                 add_to_conversation_history(user_query, response, insights, charts_generated)
#                 if len(st.session_state.conversation_history) > 1:
#                     with st.expander("🔗 Context Connection"):
#                         st.write("This analysis builds upon:")
#                         for insight in insights:
#                             st.write(f"- {insight['category'].title()}: {insight['text']}")

#     # Show video table with all derived columns for reference and download
#     with st.expander("📋 Show All Video Data (with derived columns)"):
#         show_cols = [
#             "video_id", "title", "Record Label", "Artists", "Language", "Year",
#             "Month", "Date", "Time", "Day Name", "view_count", "like_count", "comment_count",
#             "duration", "published_at", "Estimated Revenue INR"
#         ]
#         show_cols = [col for col in show_cols if col in label_videos.columns]
#         st.dataframe(label_videos[show_cols], use_container_width=True)

#     st.download_button(
#         "📥 Export Video Data", label_videos.to_csv(index=False), "videos.csv"
#     )

#     if st.session_state.conversation_history:
#         conversation_export = pd.DataFrame(st.session_state.conversation_history)
#         st.download_button(
#             "📥 Export Conversation History",
#             conversation_export.to_csv(index=False),
#             f"conversation_history_{st.session_state.session_id}.csv"
#         )

# else:
#     st.info("📁 Upload a revenue CSV to get started.")
#     st.markdown("""
#     ### 🧠 Context Memory Features:
#     - **Conversation History**: Remembers your previous questions and insights
#     - **Progressive Analysis**: Each query builds upon previous findings
#     - **Context Awareness**: AI understands the flow of your analysis
#     - **Session Persistence**: Maintains context throughout your session
#     - **Export Conversations**: Download your analysis journey
#     """)

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import requests
import spacy
from dotenv import load_dotenv
import os
from datetime import datetime
import hashlib
import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For deterministic langdetect

# Language mapping for readable names
LANG_CODE_TO_NAME = {
    'en': 'English', 'hi': 'Hindi', 'pa': 'Punjabi', 'mr': 'Marathi', 'gu': 'Gujarati', 'bn': 'Bengali',
    'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'kn': 'Kannada', 'ur': 'Urdu', 'sa': 'Sanskrit',
    'ne': 'Nepali', 'or': 'Odia', 'as': 'Assamese', 'kok': 'Konkani', 'sd': 'Sindhi', 'other': 'Other',
}

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

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

@st.cache_data
def load_youtube_metadata():
    try:
        with open("youtube_metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "videos" in data:
            df = pd.DataFrame(data["videos"])
        else:
            df = pd.DataFrame(data)
            
        if df.empty:
            st.warning("No video data found in the JSON file.")
            return df

        # --- Robust published_at extraction ---
        def extract_published_at(row):
            val = row.get("published_at")
            if isinstance(val, dict) and "$date" in val:
                return val["$date"]
            return val
        df["published_at"] = df.apply(extract_published_at, axis=1)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["Year"] = df["published_at"].dt.year
        df["Month"] = df["published_at"].dt.strftime('%B')
        df["Date"] = df["published_at"].dt.date
        df["Time"] = df["published_at"].dt.strftime('%H:%M:%S')
        df["Day Name"] = df["published_at"].dt.day_name()

        known_labels = [
            "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
            "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"
        ]
        
        def detect_label(row):
            if "channel_name" in row and row["channel_name"]:
                return row["channel_name"]
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            for label in known_labels:
                if label.lower() in text:
                    return label
            return "Other"
            
        df["Record Label"] = df.apply(detect_label, axis=1)

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

        return df
        
    except Exception as e:
        st.error(f"Error loading YouTube metadata: {str(e)}")
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
        
    if "top" in text_lower or "rpv" in text_lower:
        st.subheader("🏆 Top Videos by RPV")
        fig = px.bar(top_videos, x="RPV_Estimated", y="title", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_videos)
        charts_generated.append("Top Videos by RPV")
        
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
8. Do **not** hallucinate. If you don’t have data, say so.
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
            "temperature": 0.7,
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

# === Main App ===
st.header("📊 YouTube Analytics Dashboard")

# Load YouTube data
if st.session_state.youtube_data is None:
    st.session_state.youtube_data = load_youtube_metadata()

# File upload and data processing
uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.stop()
    
    # Label selection with "All" option
    if not st.session_state.youtube_data.empty:
        label_options = ["All"] + sorted(st.session_state.youtube_data["Record Label"].unique())
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
            label_videos = st.session_state.youtube_data.copy()
        else:
            label_videos = st.session_state.youtube_data[
                st.session_state.youtube_data["Record Label"] == selected_label
            ].copy()
        
        label_videos = label_videos.dropna(subset=["view_count"])
        label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
        est_total = label_videos["Estimated Revenue INR"].sum()

        # Get actual revenue from uploaded CSV
        if "Store Name" in df.columns:
            yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
            actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
        else:
            actual_total = 0

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

        # Query interface
        st.subheader("Ask a Business Intelligence Question")
        
        # Suggested questions based on data
        suggested_questions = [
            "What is the average view count per video?",
            "Which video has the highest engagement rate?",
            "How does revenue vary by language?",
            "What is the monthly revenue trend?",
            "Which artists generate the most revenue?"
        ]
        
        selected_suggestion = st.selectbox(
            "💡 Or select a suggested question:",
            [""] + suggested_questions
        )
        
        user_query = st.text_area(
            "Your question:",
            value=selected_suggestion if selected_suggestion else "",
            placeholder="Ask about trends, comparisons, recommendations...",
            height=100
        )

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
                    st.markdown("### AI Analysis")
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

    else:
        st.warning("No YouTube data available. Please check your JSON file.")

else:
    st.info("📁 Upload a revenue CSV to get started.")
    st.markdown("""
    ### 🧠 Key Features:
    - **Comprehensive Analytics**: View counts, engagement metrics, revenue estimation
    - **Context-Aware AI**: Remembers previous questions and builds on them
    - **Advanced Filtering**: Filter by artists, languages, date ranges
    - **Visualizations**: Automatic generation of relevant charts
    - **Data Export**: Download filtered data and conversation history
    
    ### 📊 Sample Questions to Try:
    - "What is our top performing video by revenue per view?"
    - "How does engagement vary by language?"
    - "Show me the monthly revenue trend for the past year"
    - "Which artists generate the most views?"
    """)

# Debug mode toggle (hidden)
if st.sidebar.checkbox("🔧 Debug Mode", False, key="debug_mode"):
    st.sidebar.write("Debug options enabled")
    if st.session_state.get("youtube_data") is not None:
        st.sidebar.write(f"Data shape: {st.session_state.youtube_data.shape}")


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import warnings
# warnings.filterwarnings('ignore')

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# # Load NLP model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     st.error("Please install spacy English model: python -m spacy download en_core_web_sm")
#     nlp = None

# st.set_page_config(page_title="Smart Data Analyzer Pro - Label Performance Predictor", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Label Performance & Revenue Validation")

# # Initialize session state
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
# if 'label_performance_data' not in st.session_state:
#     st.session_state.label_performance_data = None
# if 'revenue_comparison_data' not in st.session_state:
#     st.session_state.revenue_comparison_data = None

# def calculate_label_performance_metrics(youtube_df):
#     """Calculate comprehensive performance metrics for each label"""
    
#     # Group by record label and calculate metrics
#     label_metrics = youtube_df.groupby('Record Label').agg({
#         'view_count': ['sum', 'mean', 'count', 'std'],
#         'like_count': ['sum', 'mean'],
#         'comment_count': ['sum', 'mean'],
#         'published_at': ['min', 'max']
#     }).round(2)
    
#     # Flatten column names
#     label_metrics.columns = ['_'.join(col).strip() for col in label_metrics.columns]
#     label_metrics = label_metrics.reset_index()
    
#     # Calculate additional performance metrics
#     label_metrics['total_videos'] = label_metrics['view_count_count']
#     label_metrics['avg_views_per_video'] = label_metrics['view_count_mean']
#     label_metrics['total_views'] = label_metrics['view_count_sum']
#     label_metrics['engagement_rate'] = (label_metrics['like_count_sum'] + label_metrics['comment_count_sum']) / label_metrics['view_count_sum'] * 100
    
#     # Calculate consistency score (lower std deviation relative to mean = more consistent)
#     label_metrics['consistency_score'] = 100 - (label_metrics['view_count_std'] / label_metrics['view_count_mean'] * 100).fillna(0)
#     label_metrics['consistency_score'] = label_metrics['consistency_score'].clip(0, 100)
    
#     # Calculate performance score (weighted combination of metrics)
#     scaler = StandardScaler()
#     metrics_for_scoring = ['total_views', 'avg_views_per_video', 'engagement_rate', 'consistency_score', 'total_videos']
    
#     # Handle missing values
#     for col in metrics_for_scoring:
#         label_metrics[col] = label_metrics[col].fillna(label_metrics[col].median())
    
#     # Normalize metrics for scoring
#     normalized_metrics = scaler.fit_transform(label_metrics[metrics_for_scoring])
    
#     # Calculate weighted performance score
#     weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Adjust weights as needed
#     label_metrics['performance_score'] = np.dot(normalized_metrics, weights)
    
#     # Normalize performance score to 0-100 scale
#     min_score = label_metrics['performance_score'].min()
#     max_score = label_metrics['performance_score'].max()
#     label_metrics['performance_score'] = ((label_metrics['performance_score'] - min_score) / (max_score - min_score) * 100).round(2)
    
#     # Add performance tier
#     label_metrics['performance_tier'] = pd.cut(label_metrics['performance_score'], 
#                                              bins=[0, 33, 66, 100], 
#                                              labels=['Bronze', 'Silver', 'Gold'])
    
#     # Sort by performance score
#     label_metrics = label_metrics.sort_values('performance_score', ascending=False)
    
#     return label_metrics

# def predict_revenue_for_labels(youtube_df, rpm_value):
#     """Predict revenue for all labels based on views and RPM"""
#     label_revenue = youtube_df.groupby('Record Label').agg({
#         'view_count': 'sum',
#         'published_at': 'count'
#     }).rename(columns={'published_at': 'video_count'}).reset_index()
    
#     # Calculate estimated revenue
#     label_revenue['estimated_revenue_inr'] = (label_revenue['view_count'] / 1_000_000) * rpm_value
#     label_revenue['avg_revenue_per_video'] = label_revenue['estimated_revenue_inr'] / label_revenue['video_count']
    
#     # Sort by estimated revenue
#     label_revenue = label_revenue.sort_values('estimated_revenue_inr', ascending=False)
    
#     return label_revenue

# def compare_actual_vs_estimated_revenue(revenue_df, estimated_revenue, selected_label):
#     """Compare actual vs estimated revenue for selected label"""
#     comparison_data = {}
    
#     # Try to find matching revenue data
#     if 'Store Name' in revenue_df.columns:
#         # Look for YouTube revenue data
#         youtube_rows = revenue_df[revenue_df['Store Name'].str.lower().str.contains('youtube', na=False)]
        
#         if not youtube_rows.empty:
#             actual_revenue = youtube_rows['Annual Revenue in INR'].sum()
#             comparison_data['actual_revenue'] = actual_revenue
#             comparison_data['estimated_revenue'] = estimated_revenue
#             comparison_data['accuracy_percentage'] = (estimated_revenue / actual_revenue * 100) if actual_revenue > 0 else 0
#             comparison_data['variance'] = estimated_revenue - actual_revenue
#             comparison_data['variance_percentage'] = ((estimated_revenue - actual_revenue) / actual_revenue * 100) if actual_revenue > 0 else 0
    
#     # Try to find label-specific revenue data
#     label_specific_rows = revenue_df[revenue_df.apply(lambda x: selected_label.lower() in str(x).lower(), axis=1)]
    
#     if not label_specific_rows.empty and 'Annual Revenue in INR' in revenue_df.columns:
#         label_actual_revenue = label_specific_rows['Annual Revenue in INR'].sum()
#         comparison_data['label_actual_revenue'] = label_actual_revenue
#         comparison_data['label_accuracy_percentage'] = (estimated_revenue / label_actual_revenue * 100) if label_actual_revenue > 0 else 0
#         comparison_data['label_variance'] = estimated_revenue - label_actual_revenue
#         comparison_data['label_variance_percentage'] = ((estimated_revenue - label_actual_revenue) / label_actual_revenue * 100) if label_actual_revenue > 0 else 0
    
#     return comparison_data

# def generate_comprehensive_analysis_prompt(user_query, label_performance, revenue_predictions, comparison_data, youtube_df, revenue_df):
#     """Generate comprehensive prompt for AI analysis"""
    
#     # Get top performing labels
#     top_labels = label_performance.head(5)[['Record Label', 'performance_score', 'total_views', 'total_videos', 'engagement_rate']].to_dict('records')
    
#     # Get revenue predictions
#     top_revenue_labels = revenue_predictions.head(5)[['Record Label', 'estimated_revenue_inr', 'view_count', 'video_count']].to_dict('records')
    
#     # Get conversation context
#     conversation_context = get_conversation_context()
    
#     return f"""
# You are a senior Business Intelligence analyst specializing in music industry analytics and YouTube performance prediction.

# {conversation_context}

# 🏆 **TOP PERFORMING LABELS BY PERFORMANCE SCORE:**
# {json.dumps(top_labels, indent=2)}

# 💰 **TOP REVENUE GENERATING LABELS (PREDICTED):**
# {json.dumps(top_revenue_labels, indent=2)}

# 📊 **REVENUE COMPARISON DATA:**
# {json.dumps(comparison_data, indent=2)}

# 📈 **OVERALL YOUTUBE DATASET SUMMARY:**
# - Total Labels: {youtube_df['Record Label'].nunique()}
# - Total Videos: {len(youtube_df)}
# - Total Views: {youtube_df['view_count'].sum():,}
# - Average Views per Video: {youtube_df['view_count'].mean():,.0f}

# 💼 **REVENUE DATASET INFO:**
# - Revenue Sources: {revenue_df.columns.tolist() if not revenue_df.empty else 'No revenue data available'}
# - Total Revenue Records: {len(revenue_df) if not revenue_df.empty else 0}

# ---

# ### Current User Question:
# "{user_query}"

# ---

# ANALYSIS INSTRUCTIONS:
# 1. **Label Performance Analysis**: Identify top-performing labels based on comprehensive metrics
# 2. **Revenue Prediction**: Analyze predicted vs actual revenue patterns
# 3. **Comparative Analysis**: Compare performance across different labels and revenue streams
# 4. **Market Insights**: Provide insights about label market positioning
# 5. **Strategic Recommendations**: Suggest data-driven strategies for label growth
# 6. **Accuracy Assessment**: Evaluate prediction accuracy and model reliability

# Provide detailed, data-driven insights with specific numbers and actionable recommendations.
# Use markdown formatting with clear sections and bullet points.
# """

# def add_to_conversation_history(query, response, insights=None, charts_generated=None):
#     """Add query and response to conversation history with metadata"""
#     entry = {
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'query': query,
#         'response': response,
#         'insights': insights or [],
#         'charts_generated': charts_generated or [],
#         'session_id': st.session_state.session_id
#     }
#     st.session_state.conversation_history.append(entry)
    
#     # Keep only last 10 conversations to manage memory
#     if len(st.session_state.conversation_history) > 10:
#         st.session_state.conversation_history = st.session_state.conversation_history[-10:]

# def get_conversation_context():
#     """Generate conversation context for AI prompt"""
#     if not st.session_state.conversation_history:
#         return ""
    
#     context = "\n📝 **Previous Conversation Context:**\n"
#     for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
#         context += f"\n**Q{i}:** {entry['query']}\n"
#         context += f"**A{i}:** {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}\n"
    
#     return context

# def get_mistral_analysis(prompt, api_key):
#     """Get analysis from Mistral AI"""
#     try:
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 2000
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}: {response.text}"
#     except Exception as e:
#         return "", str(e)

# @st.cache_data
# def load_youtube_metadata():
#     """Load YouTube metadata with enhanced label detection"""
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)
#         df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
        
#         # Enhanced label detection
#         known_labels = [
#             "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#             "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official",
#             "Eros Music", "Shemaroo", "Ultra Music", "Panorama Music", "Rajshri Music"
#         ]
        
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')} {row.get('channel_title', '')}".lower()
#             for label in known_labels:
#                 if label.lower() in text:
#                     return label
#             return "Independent/Other"
        
#         df["Record Label"] = df.apply(detect_label, axis=1)
        
#         # Ensure numeric columns
#         numeric_columns = ['view_count', 'like_count', 'comment_count']
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
#         # Parse dates
#         if 'published_at' in df.columns:
#             df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
#         return df
#     except Exception as e:
#         st.error(f"Error loading YouTube metadata: {e}")
#         return pd.DataFrame()

# # === Sidebar for Navigation and Context Management ===
# with st.sidebar:
#     st.header("🎯 Analysis Dashboard")
    
#     analysis_mode = st.selectbox(
#         "Choose Analysis Mode:",
#         ["Label Performance Prediction", "Revenue Comparison", "Comprehensive Analysis"]
#     )
    
#     st.header("🧠 Context Memory")
    
#     if st.button("🔄 Clear Session Data"):
#         st.session_state.conversation_history = []
#         st.session_state.analysis_context = {}
#         st.session_state.label_performance_data = None
#         st.session_state.revenue_comparison_data = None
#         st.success("Session data cleared!")
    
#     st.subheader("📊 Session Stats")
#     st.metric("Queries Asked", len(st.session_state.conversation_history))
#     st.metric("Session ID", st.session_state.session_id)

# # === Main App ===
# st.subheader("📁 Data Upload Section")

# col1, col2 = st.columns(2)

# with col1:
#     st.write("**YouTube Metadata** (JSON file)")
#     youtube_file = st.file_uploader("Upload YouTube metadata JSON", type=["json"], key="youtube")
    
# with col2:
#     st.write("**Revenue Data** (CSV file)")
#     revenue_file = st.file_uploader("Upload Revenue CSV", type=["csv"], key="revenue")

# # Load default data if no files uploaded
# youtube_df = pd.DataFrame()
# revenue_df = pd.DataFrame()

# if youtube_file:
#     try:
#         data = json.load(youtube_file)
#         youtube_df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
#     except Exception as e:
#         st.error(f"Error loading YouTube file: {e}")
# else:
#     # Try to load default data
#     youtube_df = load_youtube_metadata()

# if revenue_file:
#     try:
#         revenue_df = pd.read_csv(revenue_file)
#     except Exception as e:
#         st.error(f"Error loading revenue file: {e}")

# # Main analysis section
# if not youtube_df.empty:
#     st.success(f"✅ YouTube data loaded: {len(youtube_df)} videos from {youtube_df.get('Record Label', pd.Series()).nunique()} labels")
    
#     # === LABEL PERFORMANCE PREDICTION SECTION ===
#     if analysis_mode in ["Label Performance Prediction", "Comprehensive Analysis"]:
#         st.header("🏆 Label Performance Prediction & Ranking")
        
#         # Calculate performance metrics
#         with st.spinner("Calculating label performance metrics..."):
#             label_performance = calculate_label_performance_metrics(youtube_df)
#             st.session_state.label_performance_data = label_performance
        
#         # Display top performers
#         st.subheader("🥇 Top Performing Labels")
        
#         # Performance metrics display
#         col1, col2, col3, col4 = st.columns(4)
        
#         top_performer = label_performance.iloc[0]
        
#         with col1:
#             st.metric("🏆 Top Performer", top_performer['Record Label'])
#         with col2:
#             st.metric("📊 Performance Score", f"{top_performer['performance_score']:.1f}/100")
#         with col3:
#             st.metric("👀 Total Views", f"{top_performer['total_views']:,.0f}")
#         with col4:
#             st.metric("🎥 Total Videos", f"{top_performer['total_videos']:.0f}")
        
#         # Performance visualization
#         fig_performance = px.bar(
#             label_performance.head(10), 
#             x='performance_score', 
#             y='Record Label',
#             orientation='h',
#             title="Top 10 Labels by Performance Score",
#             color='performance_score',
#             color_continuous_scale='Viridis'
#         )
#         fig_performance.update_layout(height=500)
#         st.plotly_chart(fig_performance, use_container_width=True)
        
#         # Detailed performance table
#         st.subheader("📊 Detailed Performance Metrics")
#         display_cols = ['Record Label', 'performance_score', 'performance_tier', 'total_views', 
#                        'total_videos', 'avg_views_per_video', 'engagement_rate', 'consistency_score']
#         st.dataframe(label_performance[display_cols], use_container_width=True)
    
#     # === REVENUE PREDICTION AND COMPARISON SECTION ===
#     if analysis_mode in ["Revenue Comparison", "Comprehensive Analysis"]:
#         st.header("💰 Revenue Prediction & Comparison")
        
#         # RPM input
#         rpm = st.number_input("💸 RPM (Revenue per Million Views)", min_value=1000, max_value=500000, value=125000)
        
#         # Calculate revenue predictions
#         with st.spinner("Calculating revenue predictions..."):
#             revenue_predictions = predict_revenue_for_labels(youtube_df, rpm)
        
#         # Display revenue predictions
#         st.subheader("💹 Predicted Revenue by Label")
        
#         col1, col2, col3 = st.columns(3)
        
#         top_revenue_label = revenue_predictions.iloc[0]
        
#         with col1:
#             st.metric("💰 Top Revenue Label", top_revenue_label['Record Label'])
#         with col2:
#             st.metric("💵 Predicted Revenue", f"₹{top_revenue_label['estimated_revenue_inr']:,.0f}")
#         with col3:
#             st.metric("📊 Avg Revenue/Video", f"₹{top_revenue_label['avg_revenue_per_video']:,.0f}")
        
#         # Revenue visualization
#         fig_revenue = px.bar(
#             revenue_predictions.head(10),
#             x='estimated_revenue_inr',
#             y='Record Label',
#             orientation='h',
#             title="Top 10 Labels by Predicted Revenue",
#             color='estimated_revenue_inr',
#             color_continuous_scale='Blues'
#         )
#         fig_revenue.update_layout(height=500)
#         st.plotly_chart(fig_revenue, use_container_width=True)
        
#         # Label selection for detailed comparison
#         if not revenue_df.empty:
#             st.subheader("🔍 Detailed Revenue Comparison")
            
#             selected_label = st.selectbox(
#                 "Select a label for detailed revenue comparison:",
#                 options=revenue_predictions['Record Label'].tolist()
#             )
            
#             if selected_label:
#                 # Get estimated revenue for selected label
#                 estimated_revenue = revenue_predictions[
#                     revenue_predictions['Record Label'] == selected_label
#                 ]['estimated_revenue_inr'].iloc[0]
                
#                 # Compare with actual revenue data
#                 comparison_data = compare_actual_vs_estimated_revenue(
#                     revenue_df, estimated_revenue, selected_label
#                 )
                
#                 st.session_state.revenue_comparison_data = comparison_data
                
#                 # Display comparison results
#                 if comparison_data:
#                     st.subheader(f"📊 Revenue Analysis: {selected_label}")
                    
#                     comparison_cols = st.columns(4)
                    
#                     with comparison_cols[0]:
#                         st.metric("🎯 Estimated Revenue", f"₹{estimated_revenue:,.0f}")
                    
#                     if 'actual_revenue' in comparison_data:
#                         with comparison_cols[1]:
#                             st.metric("💼 Actual Revenue", f"₹{comparison_data['actual_revenue']:,.0f}")
#                         with comparison_cols[2]:
#                             st.metric("📈 Accuracy", f"{comparison_data['accuracy_percentage']:.1f}%")
#                         with comparison_cols[3]:
#                             variance_color = "normal" if abs(comparison_data['variance_percentage']) < 20 else "inverse"
#                             st.metric("📊 Variance", f"{comparison_data['variance_percentage']:+.1f}%")
                    
#                     # Comparison visualization
#                     if 'actual_revenue' in comparison_data:
#                         fig_comparison = go.Figure(data=[
#                             go.Bar(name='Estimated', x=[selected_label], y=[estimated_revenue]),
#                             go.Bar(name='Actual', x=[selected_label], y=[comparison_data['actual_revenue']])
#                         ])
#                         fig_comparison.update_layout(
#                             title=f"Revenue Comparison: {selected_label}",
#                             barmode='group',
#                             yaxis_title="Revenue (INR)"
#                         )
#                         st.plotly_chart(fig_comparison, use_container_width=True)
#         else:
#             st.info("📄 Upload revenue CSV file to enable detailed revenue comparison")
    
#     # === AI ANALYSIS SECTION ===
#     st.header("🧠 AI-Powered Analysis")
    
#     user_query = st.text_area(
#         "Ask me anything about label performance, revenue predictions, or comparative analysis:",
#         placeholder="E.g., Which labels are underperforming? How accurate are our revenue predictions? What strategies should top labels adopt?"
#     )
    
#     if user_query and API_KEY:
#         with st.spinner("🤖 Analyzing with AI..."):
#             # Prepare data for analysis
#             label_performance = st.session_state.label_performance_data
#             if label_performance is None:
#                 label_performance = calculate_label_performance_metrics(youtube_df)
            
#             revenue_predictions = predict_revenue_for_labels(youtube_df, rpm if 'rpm' in locals() else 125000)
#             comparison_data = st.session_state.revenue_comparison_data or {}
            
#             # Generate comprehensive prompt
#             full_prompt = generate_comprehensive_analysis_prompt(
#                 user_query, label_performance, revenue_predictions, 
#                 comparison_data, youtube_df, revenue_df
#             )
            
#             # Get AI response
#             response, error = get_mistral_analysis(full_prompt, API_KEY)
            
#             if error:
#                 st.error(f"AI Analysis Error: {error}")
#             else:
#                 st.markdown("### 🎯 AI Analysis Results")
#                 st.markdown(response)
                
#                 # Add to conversation history
#                 add_to_conversation_history(user_query, response)
    
#     elif user_query and not API_KEY:
#         st.warning("⚠️ Please set your OpenRouter API key in the .env file to use AI analysis")
    
#     # === DATA EXPORT SECTION ===
#     st.header("📥 Data Export")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.session_state.label_performance_data is not None:
#             st.download_button(
#                 "📊 Export Performance Data",
#                 st.session_state.label_performance_data.to_csv(index=False),
#                 "label_performance_analysis.csv",
#                 "text/csv"
#             )
    
#     with col2:
#         revenue_predictions = predict_revenue_for_labels(youtube_df, rpm if 'rpm' in locals() else 125000)
#         st.download_button(
#             "💰 Export Revenue Predictions",
#             revenue_predictions.to_csv(index=False),
#             "revenue_predictions.csv",
#             "text/csv"
#         )
    
#     with col3:
#         if st.session_state.conversation_history:
#             conversation_export = pd.DataFrame(st.session_state.conversation_history)
#             st.download_button(
#                 "💬 Export Analysis History",
#                 conversation_export.to_csv(index=False),
#                 f"analysis_history_{st.session_state.session_id}.csv",
#                 "text/csv"
#             )

# else:
#     st.info("📁 Please upload YouTube metadata JSON file or ensure youtube_metadata.json exists in the current directory")
    
#     # Show example data structure
#     st.subheader("📋 Expected Data Structure")
    
#     st.markdown("""
#     **YouTube Metadata JSON Structure:**
#     ```json
#     {
#         "videos": [
#             {
#                 "title": "Song Title",
#                 "channel_title": "Channel Name",
#                 "view_count": 1000000,
#                 "like_count": 50000,
#                 "comment_count": 5000,
#                 "published_at": "2024-01-01T00:00:00Z",
#                 "description": "Song description..."
#             }
#         ]
#     }
#     ```
    
#     **Revenue CSV Structure:**
#     ```
#     Store Name,Annual Revenue in INR,Platform,Label
#     YouTube,50000000,Digital,T-Series
#     Spotify,25000000,Streaming,Sony Music
#     ```
#     """)

# # === FOOTER ===
# st.markdown("---")
# st.markdown("🎵 **Smart Data Analyzer Pro** - Advanced YouTube Label Performance & Revenue Analytics")


########################################################
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import warnings
# warnings.filterwarnings('ignore')

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# # Load NLP model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     st.error("Please install spacy English model: python -m spacy download en_core_web_sm")
#     nlp = None

# st.set_page_config(page_title="Smart Data Analyzer Pro - Label Performance Predictor", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Label Performance & Revenue Validation")

# # Initialize session state
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
# if 'label_performance_data' not in st.session_state:
#     st.session_state.label_performance_data = None
# if 'revenue_comparison_data' not in st.session_state:
#     st.session_state.revenue_comparison_data = None

# def calculate_label_performance_metrics(youtube_df):
#     """Calculate comprehensive performance metrics for each label"""
    
#     # Group by record label and calculate metrics
#     label_metrics = youtube_df.groupby('Record Label').agg({
#         'view_count': ['sum', 'mean', 'count', 'std'],
#         'like_count': ['sum', 'mean'],
#         'comment_count': ['sum', 'mean'],
#         'published_at': ['min', 'max']
#     }).round(2)
    
#     # Flatten column names
#     label_metrics.columns = ['_'.join(col).strip() for col in label_metrics.columns]
#     label_metrics = label_metrics.reset_index()
    
#     # Calculate additional performance metrics
#     label_metrics['total_videos'] = label_metrics['view_count_count']
#     label_metrics['avg_views_per_video'] = label_metrics['view_count_mean']
#     label_metrics['total_views'] = label_metrics['view_count_sum']
#     label_metrics['engagement_rate'] = (label_metrics['like_count_sum'] + label_metrics['comment_count_sum']) / label_metrics['view_count_sum'] * 100
    
#     # Calculate consistency score (lower std deviation relative to mean = more consistent)
#     label_metrics['consistency_score'] = 100 - (label_metrics['view_count_std'] / label_metrics['view_count_mean'] * 100).fillna(0)
#     label_metrics['consistency_score'] = label_metrics['consistency_score'].clip(0, 100)
    
#     # Calculate performance score (weighted combination of metrics)
#     scaler = StandardScaler()
#     metrics_for_scoring = ['total_views', 'avg_views_per_video', 'engagement_rate', 'consistency_score', 'total_videos']
    
#     # Handle missing values
#     for col in metrics_for_scoring:
#         label_metrics[col] = label_metrics[col].fillna(label_metrics[col].median())
    
#     # Normalize metrics for scoring
#     normalized_metrics = scaler.fit_transform(label_metrics[metrics_for_scoring])
    
#     # Calculate weighted performance score
#     weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Adjust weights as needed
#     label_metrics['performance_score'] = np.dot(normalized_metrics, weights)
    
#     # Normalize performance score to 0-100 scale
#     min_score = label_metrics['performance_score'].min()
#     max_score = label_metrics['performance_score'].max()
#     label_metrics['performance_score'] = ((label_metrics['performance_score'] - min_score) / (max_score - min_score) * 100).round(2)
    
#     # Add performance tier
#     label_metrics['performance_tier'] = pd.cut(label_metrics['performance_score'], 
#                                              bins=[0, 33, 66, 100], 
#                                              labels=['Bronze', 'Silver', 'Gold'])
    
#     # Sort by performance score
#     label_metrics = label_metrics.sort_values('performance_score', ascending=False)
    
#     return label_metrics

# def predict_revenue_for_labels(youtube_df, rpm_value):
#     """Predict revenue for all labels based on views and RPM"""
#     label_revenue = youtube_df.groupby('Record Label').agg({
#         'view_count': 'sum',
#         'published_at': 'count'
#     }).rename(columns={'published_at': 'video_count'}).reset_index()
    
#     # Calculate estimated revenue
#     label_revenue['estimated_revenue_inr'] = (label_revenue['view_count'] / 1_000_000) * rpm_value
#     label_revenue['avg_revenue_per_video'] = label_revenue['estimated_revenue_inr'] / label_revenue['video_count']
    
#     # Sort by estimated revenue
#     label_revenue = label_revenue.sort_values('estimated_revenue_inr', ascending=False)
    
#     return label_revenue

# def compare_actual_vs_estimated_revenue(revenue_df, estimated_revenue, selected_label):
#     """Compare actual vs estimated revenue for selected label"""
#     comparison_data = {}
    
#     # Try to find matching revenue data
#     if 'Store Name' in revenue_df.columns:
#         # Look for YouTube revenue data
#         youtube_rows = revenue_df[revenue_df['Store Name'].str.lower().str.contains('youtube', na=False)]
        
#         if not youtube_rows.empty:
#             actual_revenue = youtube_rows['Annual Revenue in INR'].sum()
#             comparison_data['actual_revenue'] = actual_revenue
#             comparison_data['estimated_revenue'] = estimated_revenue
#             comparison_data['accuracy_percentage'] = (estimated_revenue / actual_revenue * 100) if actual_revenue > 0 else 0
#             comparison_data['variance'] = estimated_revenue - actual_revenue
#             comparison_data['variance_percentage'] = ((estimated_revenue - actual_revenue) / actual_revenue * 100) if actual_revenue > 0 else 0
    
#     # Try to find label-specific revenue data
#     label_specific_rows = revenue_df[revenue_df.apply(lambda x: selected_label.lower() in str(x).lower(), axis=1)]
    
#     if not label_specific_rows.empty and 'Annual Revenue in INR' in revenue_df.columns:
#         label_actual_revenue = label_specific_rows['Annual Revenue in INR'].sum()
#         comparison_data['label_actual_revenue'] = label_actual_revenue
#         comparison_data['label_accuracy_percentage'] = (estimated_revenue / label_actual_revenue * 100) if label_actual_revenue > 0 else 0
#         comparison_data['label_variance'] = estimated_revenue - label_actual_revenue
#         comparison_data['label_variance_percentage'] = ((estimated_revenue - label_actual_revenue) / label_actual_revenue * 100) if label_actual_revenue > 0 else 0
    
#     return comparison_data

# def generate_comprehensive_analysis_prompt(user_query, label_performance, revenue_predictions, comparison_data, youtube_df, revenue_df):
#     """Generate comprehensive prompt for AI analysis"""
    
#     # Get top performing labels
#     top_labels = label_performance.head(5)[['Record Label', 'performance_score', 'total_views', 'total_videos', 'engagement_rate']].to_dict('records')
    
#     # Get revenue predictions
#     top_revenue_labels = revenue_predictions.head(5)[['Record Label', 'estimated_revenue_inr', 'view_count', 'video_count']].to_dict('records')
    
#     # Get conversation context
#     conversation_context = get_conversation_context()
    
#     return f"""
# You are a senior Business Intelligence analyst specializing in music industry analytics and YouTube performance prediction.

# {conversation_context}

# 🏆 **TOP PERFORMING LABELS BY PERFORMANCE SCORE:**
# {json.dumps(top_labels, indent=2)}

# 💰 **TOP REVENUE GENERATING LABELS (PREDICTED):**
# {json.dumps(top_revenue_labels, indent=2)}

# 📊 **REVENUE COMPARISON DATA:**
# {json.dumps(comparison_data, indent=2)}

# 📈 **OVERALL YOUTUBE DATASET SUMMARY:**
# - Total Labels: {youtube_df['Record Label'].nunique()}
# - Total Videos: {len(youtube_df)}
# - Total Views: {youtube_df['view_count'].sum():,}
# - Average Views per Video: {youtube_df['view_count'].mean():,.0f}

# 💼 **REVENUE DATASET INFO:**
# - Revenue Sources: {revenue_df.columns.tolist() if not revenue_df.empty else 'No revenue data available'}
# - Total Revenue Records: {len(revenue_df) if not revenue_df.empty else 0}

# ---

# ### Current User Question:
# "{user_query}"

# ---

# ANALYSIS INSTRUCTIONS:
# 1. **Label Performance Analysis**: Identify top-performing labels based on comprehensive metrics
# 2. **Revenue Prediction**: Analyze predicted vs actual revenue patterns
# 3. **Comparative Analysis**: Compare performance across different labels and revenue streams
# 4. **Market Insights**: Provide insights about label market positioning
# 5. **Strategic Recommendations**: Suggest data-driven strategies for label growth
# 6. **Accuracy Assessment**: Evaluate prediction accuracy and model reliability

# Provide detailed, data-driven insights with specific numbers and actionable recommendations.
# Use markdown formatting with clear sections and bullet points.
# """

# def add_to_conversation_history(query, response, insights=None, charts_generated=None):
#     """Add query and response to conversation history with metadata"""
#     entry = {
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'query': query,
#         'response': response,
#         'insights': insights or [],
#         'charts_generated': charts_generated or [],
#         'session_id': st.session_state.session_id
#     }
#     st.session_state.conversation_history.append(entry)
    
#     # Keep only last 10 conversations to manage memory
#     if len(st.session_state.conversation_history) > 10:
#         st.session_state.conversation_history = st.session_state.conversation_history[-10:]

# def get_conversation_context():
#     """Generate conversation context for AI prompt"""
#     if not st.session_state.conversation_history:
#         return ""
    
#     context = "\n📝 **Previous Conversation Context:**\n"
#     for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
#         context += f"\n**Q{i}:** {entry['query']}\n"
#         context += f"**A{i}:** {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}\n"
    
#     return context

# def get_mistral_analysis(prompt, api_key):
#     """Get analysis from Mistral AI"""
#     try:
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 2000
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}: {response.text}"
#     except Exception as e:
#         return "", str(e)

# @st.cache_data
# def load_youtube_metadata():
#     """Load YouTube metadata with enhanced label detection"""
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)
#         df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
        
#         # Enhanced label detection
#         known_labels = [
#             "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#             "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official",
#             "Eros Music", "Shemaroo", "Ultra Music", "Panorama Music", "Rajshri Music"
#         ]
        
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')} {row.get('channel_title', '')}".lower()
#             for label in known_labels:
#                 if label.lower() in text:
#                     return label
#             return "Independent/Other"
        
#         df["Record Label"] = df.apply(detect_label, axis=1)
        
#         # Ensure numeric columns
#         numeric_columns = ['view_count', 'like_count', 'comment_count']
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
#         # Parse dates
#         if 'published_at' in df.columns:
#             df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
#         return df
#     except Exception as e:
#         st.error(f"Error loading YouTube metadata: {e}")
#         return pd.DataFrame()

# # === Sidebar for Navigation and Context Management ===
# with st.sidebar:
#     st.header("🎯 Analysis Dashboard")
    
#     analysis_mode = st.selectbox(
#         "Choose Analysis Mode:",
#         ["Label Performance Prediction", "Revenue Comparison", "Comprehensive Analysis"]
#     )
    
#     st.header("📁 Data Upload")
#     st.write("**Revenue Data** (CSV file)")
#     revenue_file = st.file_uploader("Upload Revenue CSV", type=["csv"], key="revenue")
    
#     st.header("🧠 Context Memory")
    
#     if st.button("🔄 Clear Session Data"):
#         st.session_state.conversation_history = []
#         st.session_state.analysis_context = {}
#         st.session_state.label_performance_data = None
#         st.session_state.revenue_comparison_data = None
#         st.success("Session data cleared!")
    
#     st.subheader("📊 Session Stats")
#     st.metric("Queries Asked", len(st.session_state.conversation_history))
#     st.metric("Session ID", st.session_state.session_id)

# # === Main App ===

# # Load data - YouTube data from backend, revenue from uploaded file
# youtube_df = load_youtube_metadata()
# revenue_df = pd.DataFrame()

# if revenue_file:
#     try:
#         revenue_df = pd.read_csv(revenue_file)
#         st.sidebar.success(f"✅ Revenue data loaded: {len(revenue_df)} records")
#     except Exception as e:
#         st.sidebar.error(f"Error loading revenue file: {e}")

# # Main analysis section
# if not youtube_df.empty:
#     st.success(f"✅ YouTube data loaded: {len(youtube_df)} videos from {youtube_df.get('Record Label', pd.Series()).nunique()} labels")
    
#     # === LABEL PERFORMANCE PREDICTION SECTION ===
#     if analysis_mode in ["Label Performance Prediction", "Comprehensive Analysis"]:
#         st.header("🏆 Label Performance Prediction & Ranking")
        
#         # Calculate performance metrics
#         with st.spinner("Calculating label performance metrics..."):
#             label_performance = calculate_label_performance_metrics(youtube_df)
#             st.session_state.label_performance_data = label_performance
        
#         # Display top performers
#         st.subheader("🥇 Top Performing Labels")
        
#         # Performance metrics display
#         col1, col2, col3, col4 = st.columns(4)
        
#         top_performer = label_performance.iloc[0]
        
#         with col1:
#             st.metric("🏆 Top Performer", top_performer['Record Label'])
#         with col2:
#             st.metric("📊 Performance Score", f"{top_performer['performance_score']:.1f}/100")
#         with col3:
#             st.metric("👀 Total Views", f"{top_performer['total_views']:,.0f}")
#         with col4:
#             st.metric("🎥 Total Videos", f"{top_performer['total_videos']:.0f}")
        
#         # Performance visualization
#         fig_performance = px.bar(
#             label_performance.head(10), 
#             x='performance_score', 
#             y='Record Label',
#             orientation='h',
#             title="Top 10 Labels by Performance Score",
#             color='performance_score',
#             color_continuous_scale='Viridis'
#         )
#         fig_performance.update_layout(height=500)
#         st.plotly_chart(fig_performance, use_container_width=True)
        
#         # Detailed performance table
#         st.subheader("📊 Detailed Performance Metrics")
#         display_cols = ['Record Label', 'performance_score', 'performance_tier', 'total_views', 
#                        'total_videos', 'avg_views_per_video', 'engagement_rate', 'consistency_score']
#         st.dataframe(label_performance[display_cols], use_container_width=True)
    
#     # === REVENUE PREDICTION AND COMPARISON SECTION ===
#     if analysis_mode in ["Revenue Comparison", "Comprehensive Analysis"]:
#         st.header("💰 Revenue Prediction & Comparison")
        
#         # RPM input
#         rpm = st.number_input("💸 RPM (Revenue per Million Views)", min_value=1000, max_value=500000, value=225000)
        
#         # Calculate revenue predictions
#         with st.spinner("Calculating revenue predictions..."):
#             revenue_predictions = predict_revenue_for_labels(youtube_df, rpm)
        
#         # Display revenue predictions
#         st.subheader("💹 Predicted Revenue by Label")
        
#         col1, col2, col3 = st.columns(3)
        
#         top_revenue_label = revenue_predictions.iloc[0]
        
#         with col1:
#             st.metric("💰 Top Revenue Label", top_revenue_label['Record Label'])
#         with col2:
#             st.metric("💵 Predicted Revenue", f"₹{top_revenue_label['estimated_revenue_inr']:,.0f}")
#         with col3:
#             st.metric("📊 Avg Revenue/Video", f"₹{top_revenue_label['avg_revenue_per_video']:,.0f}")
        
#         # Revenue visualization
#         fig_revenue = px.bar(
#             revenue_predictions.head(10),
#             x='estimated_revenue_inr',
#             y='Record Label',
#             orientation='h',
#             title="Top 10 Labels by Predicted Revenue",
#             color='estimated_revenue_inr',
#             color_continuous_scale='Blues'
#         )
#         fig_revenue.update_layout(height=500)
#         st.plotly_chart(fig_revenue, use_container_width=True)
        
#         # Label selection for detailed comparison
#         if not revenue_df.empty:
#             st.subheader("🔍 Detailed Revenue Comparison")
            
#             selected_label = st.selectbox(
#                 "Select a label for detailed revenue comparison:",
#                 options=revenue_predictions['Record Label'].tolist()
#             )
            
#             if selected_label:
#                 # Get estimated revenue for selected label
#                 estimated_revenue = revenue_predictions[
#                     revenue_predictions['Record Label'] == selected_label
#                 ]['estimated_revenue_inr'].iloc[0]
                
#                 # Compare with actual revenue data
#                 comparison_data = compare_actual_vs_estimated_revenue(
#                     revenue_df, estimated_revenue, selected_label
#                 )
                
#                 st.session_state.revenue_comparison_data = comparison_data
                
#                 # Display comparison results
#                 if comparison_data:
#                     st.subheader(f"📊 Revenue Analysis: {selected_label}")
                    
#                     comparison_cols = st.columns(4)
                    
#                     with comparison_cols[0]:
#                         st.metric("🎯 Estimated Revenue", f"₹{estimated_revenue:,.0f}")
                    
#                     if 'actual_revenue' in comparison_data:
#                         with comparison_cols[1]:
#                             st.metric("💼 Actual Revenue", f"₹{comparison_data['actual_revenue']:,.0f}")
#                         with comparison_cols[2]:
#                             st.metric("📈 Accuracy", f"{comparison_data['accuracy_percentage']:.1f}%")
#                         with comparison_cols[3]:
#                             variance_color = "normal" if abs(comparison_data['variance_percentage']) < 20 else "inverse"
#                             st.metric("📊 Variance", f"{comparison_data['variance_percentage']:+.1f}%")
                    
#                     # Comparison visualization
#                     if 'actual_revenue' in comparison_data:
#                         fig_comparison = go.Figure(data=[
#                             go.Bar(name='Estimated', x=[selected_label], y=[estimated_revenue]),
#                             go.Bar(name='Actual', x=[selected_label], y=[comparison_data['actual_revenue']])
#                         ])
#                         fig_comparison.update_layout(
#                             title=f"Revenue Comparison: {selected_label}",
#                             barmode='group',
#                             yaxis_title="Revenue (INR)"
#                         )
#                         st.plotly_chart(fig_comparison, use_container_width=True)
#         else:
#             st.info("📄 Upload revenue CSV file in the sidebar to enable detailed revenue comparison")
    
#     # === AI ANALYSIS SECTION ===
#     st.header("🧠 AI-Powered Analysis")
    
#     user_query = st.text_area(
#         "Ask me anything about label performance, revenue predictions, or comparative analysis:",
#         placeholder="E.g., Which labels are underperforming? How accurate are our revenue predictions? What strategies should top labels adopt?"
#     )
    
#     if user_query and API_KEY:
#         with st.spinner("🤖 Analyzing with AI..."):
#             # Prepare data for analysis
#             label_performance = st.session_state.label_performance_data
#             if label_performance is None:
#                 label_performance = calculate_label_performance_metrics(youtube_df)
            
#             revenue_predictions = predict_revenue_for_labels(youtube_df, rpm if 'rpm' in locals() else 125000)
#             comparison_data = st.session_state.revenue_comparison_data or {}
            
#             # Generate comprehensive prompt
#             full_prompt = generate_comprehensive_analysis_prompt(
#                 user_query, label_performance, revenue_predictions, 
#                 comparison_data, youtube_df, revenue_df
#             )
            
#             # Get AI response
#             response, error = get_mistral_analysis(full_prompt, API_KEY)
            
#             if error:
#                 st.error(f"AI Analysis Error: {error}")
#             else:
#                 st.markdown("### 🎯 AI Analysis Results")
#                 st.markdown(response)
                
#                 # Add to conversation history
#                 add_to_conversation_history(user_query, response)
    
#     elif user_query and not API_KEY:
#         st.warning("⚠️ Please set your OpenRouter API key in the .env file to use AI analysis")
    
#     # === DATA EXPORT SECTION ===
#     st.header("📥 Data Export")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.session_state.label_performance_data is not None:
#             st.download_button(
#                 "📊 Export Performance Data",
#                 st.session_state.label_performance_data.to_csv(index=False),
#                 "label_performance_analysis.csv",
#                 "text/csv"
#             )
    
#     with col2:
#         revenue_predictions = predict_revenue_for_labels(youtube_df, rpm if 'rpm' in locals() else 125000)
#         st.download_button(
#             "💰 Export Revenue Predictions",
#             revenue_predictions.to_csv(index=False),
#             "revenue_predictions.csv",
#             "text/csv"
#         )
    
#     with col3:
#         if st.session_state.conversation_history:
#             conversation_export = pd.DataFrame(st.session_state.conversation_history)
#             st.download_button(
#                 "💬 Export Analysis History",
#                 conversation_export.to_csv(index=False),
#                 f"analysis_history_{st.session_state.session_id}.csv",
#                 "text/csv"
#             )

# else:
#     st.warning("📁 YouTube metadata not found. Please ensure the backend connection is working and youtube_metadata.json is available.")
    
#     # Show example data structure for revenue CSV
#     st.subheader("📋 Expected Revenue CSV Structure")
    
#     st.markdown("""
#     **Revenue CSV Structure:**
#     ```
#     Store Name,Annual Revenue in INR,Platform,Label
#     YouTube,50000000,Digital,T-Series
#     Spotify,25000000,Streaming,Sony Music
#     Apple Music,15000000,Streaming,Zee Music
#     ```
    
#     Upload your revenue CSV file using the sidebar to enable detailed revenue comparison and analysis.
#     """)

# # === FOOTER ===
# st.markdown("---")
# st.markdown("🎵 **Smart Data Analyzer Pro** - Advanced YouTube Label Performance & Revenue Analytics")

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import warnings
# from pymongo import MongoClient
# from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
# import urllib.parse
# warnings.filterwarnings('ignore')

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# # MongoDB configuration
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
# MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "youtube_analytics")
# MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "youtube_metadata")

# # Load NLP model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     st.error("Please install spacy English model: python -m spacy download en_core_web_sm")
#     nlp = None

# st.set_page_config(page_title="Smart Data Analyzer Pro - Label Performance Predictor", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Label Performance & Revenue Validation")

# # Initialize session state
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
# if 'label_performance_data' not in st.session_state:
#     st.session_state.label_performance_data = None
# if 'revenue_comparison_data' not in st.session_state:
#     st.session_state.revenue_comparison_data = None

# def calculate_label_performance_metrics(youtube_df):
#     """Calculate comprehensive performance metrics for each label"""
    
#     # Group by record label and calculate metrics
#     label_metrics = youtube_df.groupby('Record Label').agg({
#         'view_count': ['sum', 'mean', 'count', 'std'],
#         'like_count': ['sum', 'mean'],
#         'comment_count': ['sum', 'mean'],
#         'published_at': ['min', 'max']
#     }).round(2)
    
#     # Flatten column names
#     label_metrics.columns = ['_'.join(col).strip() for col in label_metrics.columns]
#     label_metrics = label_metrics.reset_index()
    
#     # Calculate additional performance metrics
#     label_metrics['total_videos'] = label_metrics['view_count_count']
#     label_metrics['avg_views_per_video'] = label_metrics['view_count_mean']
#     label_metrics['total_views'] = label_metrics['view_count_sum']
#     label_metrics['engagement_rate'] = (label_metrics['like_count_sum'] + label_metrics['comment_count_sum']) / label_metrics['view_count_sum'] * 100
    
#     # Calculate consistency score (lower std deviation relative to mean = more consistent)
#     label_metrics['consistency_score'] = 100 - (label_metrics['view_count_std'] / label_metrics['view_count_mean'] * 100).fillna(0)
#     label_metrics['consistency_score'] = label_metrics['consistency_score'].clip(0, 100)
    
#     # Calculate performance score (weighted combination of metrics)
#     scaler = StandardScaler()
#     metrics_for_scoring = ['total_views', 'avg_views_per_video', 'engagement_rate', 'consistency_score', 'total_videos']
    
#     # Handle missing values
#     for col in metrics_for_scoring:
#         label_metrics[col] = label_metrics[col].fillna(label_metrics[col].median())
    
#     # Normalize metrics for scoring
#     normalized_metrics = scaler.fit_transform(label_metrics[metrics_for_scoring])
    
#     # Calculate weighted performance score
#     weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Adjust weights as needed
#     label_metrics['performance_score'] = np.dot(normalized_metrics, weights)
    
#     # Normalize performance score to 0-100 scale
#     min_score = label_metrics['performance_score'].min()
#     max_score = label_metrics['performance_score'].max()
#     label_metrics['performance_score'] = ((label_metrics['performance_score'] - min_score) / (max_score - min_score) * 100).round(2)
    
#     # Add performance tier
#     label_metrics['performance_tier'] = pd.cut(label_metrics['performance_score'], 
#                                              bins=[0, 33, 66, 100], 
#                                              labels=['Bronze', 'Silver', 'Gold'])
    
#     # Sort by performance score
#     label_metrics = label_metrics.sort_values('performance_score', ascending=False)
    
#     return label_metrics

# def predict_revenue_for_labels(youtube_df, rpm_value):
#     """Predict revenue for all labels based on views and RPM"""
#     label_revenue = youtube_df.groupby('Record Label').agg({
#         'view_count': 'sum',
#         'published_at': 'count'
#     }).rename(columns={'published_at': 'video_count'}).reset_index()
    
#     # Calculate estimated revenue
#     label_revenue['estimated_revenue_inr'] = (label_revenue['view_count'] / 1_000_000) * rpm_value
#     label_revenue['avg_revenue_per_video'] = label_revenue['estimated_revenue_inr'] / label_revenue['video_count']
    
#     # Sort by estimated revenue
#     label_revenue = label_revenue.sort_values('estimated_revenue_inr', ascending=False)
    
#     return label_revenue

# def compare_actual_vs_estimated_revenue(revenue_df, estimated_revenue, selected_label):
#     """Compare actual vs estimated revenue for selected label"""
#     comparison_data = {}
    
#     # Try to find matching revenue data
#     if 'Store Name' in revenue_df.columns:
#         # Look for YouTube revenue data
#         youtube_rows = revenue_df[revenue_df['Store Name'].str.lower().str.contains('youtube', na=False)]
        
#         if not youtube_rows.empty:
#             actual_revenue = youtube_rows['Annual Revenue in INR'].sum()
#             comparison_data['actual_revenue'] = actual_revenue
#             comparison_data['estimated_revenue'] = estimated_revenue
#             comparison_data['accuracy_percentage'] = (estimated_revenue / actual_revenue * 100) if actual_revenue > 0 else 0
#             comparison_data['variance'] = estimated_revenue - actual_revenue
#             comparison_data['variance_percentage'] = ((estimated_revenue - actual_revenue) / actual_revenue * 100) if actual_revenue > 0 else 0
    
#     # Try to find label-specific revenue data
#     label_specific_rows = revenue_df[revenue_df.apply(lambda x: selected_label.lower() in str(x).lower(), axis=1)]
    
#     if not label_specific_rows.empty and 'Annual Revenue in INR' in revenue_df.columns:
#         label_actual_revenue = label_specific_rows['Annual Revenue in INR'].sum()
#         comparison_data['label_actual_revenue'] = label_actual_revenue
#         comparison_data['label_accuracy_percentage'] = (estimated_revenue / label_actual_revenue * 100) if label_actual_revenue > 0 else 0
#         comparison_data['label_variance'] = estimated_revenue - label_actual_revenue
#         comparison_data['label_variance_percentage'] = ((estimated_revenue - label_actual_revenue) / label_actual_revenue * 100) if label_actual_revenue > 0 else 0
    
#     return comparison_data

# def generate_comprehensive_analysis_prompt(user_query, label_performance, revenue_predictions, comparison_data, youtube_df, revenue_df):
#     """Generate comprehensive prompt for AI analysis"""
    
#     # Get top performing labels
#     top_labels = label_performance.head(5)[['Record Label', 'performance_score', 'total_views', 'total_videos', 'engagement_rate']].to_dict('records')
    
#     # Get revenue predictions
#     top_revenue_labels = revenue_predictions.head(5)[['Record Label', 'estimated_revenue_inr', 'view_count', 'video_count']].to_dict('records')
    
#     # Get conversation context
#     conversation_context = get_conversation_context()
    
#     return f"""
# You are a senior Business Intelligence analyst specializing in music industry analytics and YouTube performance prediction.

# {conversation_context}

# 🏆 **TOP PERFORMING LABELS BY PERFORMANCE SCORE:**
# {json.dumps(top_labels, indent=2)}

# 💰 **TOP REVENUE GENERATING LABELS (PREDICTED):**
# {json.dumps(top_revenue_labels, indent=2)}

# 📊 **REVENUE COMPARISON DATA:**
# {json.dumps(comparison_data, indent=2)}

# 📈 **OVERALL YOUTUBE DATASET SUMMARY:**
# - Total Labels: {youtube_df['Record Label'].nunique()}
# - Total Videos: {len(youtube_df)}
# - Total Views: {youtube_df['view_count'].sum():,}
# - Average Views per Video: {youtube_df['view_count'].mean():,.0f}

# 💼 **REVENUE DATASET INFO:**
# - Revenue Sources: {revenue_df.columns.tolist() if not revenue_df.empty else 'No revenue data available'}
# - Total Revenue Records: {len(revenue_df) if not revenue_df.empty else 0}

# ---

# ### Current User Question:
# "{user_query}"

# ---

# ANALYSIS INSTRUCTIONS:
# 1. **Label Performance Analysis**: Identify top-performing labels based on comprehensive metrics
# 2. **Revenue Prediction**: Analyze predicted vs actual revenue patterns
# 3. **Comparative Analysis**: Compare performance across different labels and revenue streams
# 4. **Market Insights**: Provide insights about label market positioning
# 5. **Strategic Recommendations**: Suggest data-driven strategies for label growth
# 6. **Accuracy Assessment**: Evaluate prediction accuracy and model reliability

# Provide detailed, data-driven insights with specific numbers and actionable recommendations.
# Use markdown formatting with clear sections and bullet points.
# """

# def add_to_conversation_history(query, response, insights=None, charts_generated=None):
#     """Add query and response to conversation history with metadata"""
#     entry = {
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'query': query,
#         'response': response,
#         'insights': insights or [],
#         'charts_generated': charts_generated or [],
#         'session_id': st.session_state.session_id
#     }
#     st.session_state.conversation_history.append(entry)
    
#     # Keep only last 10 conversations to manage memory
#     if len(st.session_state.conversation_history) > 10:
#         st.session_state.conversation_history = st.session_state.conversation_history[-10:]

# def get_conversation_context():
#     """Generate conversation context for AI prompt"""
#     if not st.session_state.conversation_history:
#         return ""
    
#     context = "\n📝 **Previous Conversation Context:**\n"
#     for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
#         context += f"\n**Q{i}:** {entry['query']}\n"
#         context += f"**A{i}:** {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}\n"
    
#     return context

# def get_mistral_analysis(prompt, api_key):
#     """Get analysis from Mistral AI"""
#     try:
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 2000
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}: {response.text}"
#     except Exception as e:
#         return "", str(e)

# def connect_to_mongodb():
#     """Connect to MongoDB and return client and database"""
#     try:
#         # Parse URI to handle authentication
#         if "@" in MONGO_URI:
#             # URI contains authentication
#             client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
#         else:
#             # Simple URI
#             client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
#         # Test connection
#         client.admin.command('ping')
#         db = client[MONGO_DB_NAME]
        
#         st.sidebar.success("✅ MongoDB Connected")
#         return client, db
    
#     except ConnectionFailure:
#         st.sidebar.error("❌ MongoDB Connection Failed")
#         return None, None
#     except ServerSelectionTimeoutError:
#         st.sidebar.error("❌ MongoDB Server Timeout")
#         return None, None
#     except Exception as e:
#         st.sidebar.error(f"❌ MongoDB Error: {str(e)}")
#         return None, None

# @st.cache_data(ttl=300)  # Cache for 5 minutes
# def load_youtube_metadata_from_mongodb():
#     """Load YouTube metadata from MongoDB with enhanced label detection"""
#     try:
#         client, db = connect_to_mongodb()
#         if not client or not db:
#             return pd.DataFrame()
        
#         collection = db[MONGO_COLLECTION_NAME]
        
#         # Count documents
#         doc_count = collection.count_documents({})
#         if doc_count == 0:
#             st.sidebar.warning("⚠️ No documents found in MongoDB collection")
#             return pd.DataFrame()
        
#         st.sidebar.info(f"📊 Loading {doc_count} documents from MongoDB...")
        
#         # Fetch data with progress
#         with st.sidebar:
#             progress_bar = st.progress(0)
#             status_text = st.empty()
        
#         # Get all documents
#         cursor = collection.find({})
#         documents = []
        
#         for i, doc in enumerate(cursor):
#             # Remove MongoDB ObjectId for JSON serialization
#             if '_id' in doc:
#                 del doc['_id']
#             documents.append(doc)
            
#             # Update progress
#             progress = (i + 1) / doc_count
#             progress_bar.progress(progress)
#             status_text.text(f"Loading... {i+1}/{doc_count}")
        
#         progress_bar.empty()
#         status_text.empty()
        
#         # Convert to DataFrame
#         if documents:
#             # Check if documents are in 'videos' wrapper format
#             if len(documents) == 1 and 'videos' in documents[0]:
#                 df = pd.DataFrame(documents[0]['videos'])
#             else:
#                 df = pd.DataFrame(documents)
#         else:
#             df = pd.DataFrame()
        
#         if df.empty:
#             st.sidebar.warning("⚠️ No video data found in MongoDB")
#             return df
        
#         # Enhanced label detection
#         known_labels = [
#             "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#             "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official",
#             "Eros Music", "Shemaroo", "Ultra Music", "Panorama Music", "Rajshri Music",
#             "Desi Music Factory", "Goldmines", "Pen Movies", "Bollywood Classics"
#         ]
        
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')} {row.get('channel_title', '')}".lower()
#             for label in known_labels:
#                 if label.lower() in text:
#                     return label
#             return "Independent/Other"
        
#         df["Record Label"] = df.apply(detect_label, axis=1)
        
#         # Ensure numeric columns
#         numeric_columns = ['view_count', 'like_count', 'comment_count']
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
#         # Parse dates
#         if 'published_at' in df.columns:
#             df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
#         # Close MongoDB connection
#         client.close()
        
#         st.sidebar.success(f"✅ Loaded {len(df)} videos from MongoDB")
#         return df
        
#     except Exception as e:
#         st.sidebar.error(f"❌ Error loading from MongoDB: {str(e)}")
#         return pd.DataFrame()

# def get_mongodb_stats():
#     """Get MongoDB collection statistics"""
#     try:
#         client, db = connect_to_mongodb()
#         if not client or not db:
#             return {}
        
#         collection = db[MONGO_COLLECTION_NAME]
        
#         stats = {
#             'total_documents': collection.count_documents({}),
#             'collection_name': MONGO_COLLECTION_NAME,
#             'database_name': MONGO_DB_NAME,
#         }
        
#         # Get sample document structure
#         sample_doc = collection.find_one({})
#         if sample_doc:
#             if '_id' in sample_doc:
#                 del sample_doc['_id']
#             stats['sample_fields'] = list(sample_doc.keys())
        
#         client.close()
#         return stats
        
#     except Exception as e:
#         return {'error': str(e)}

# # === Sidebar for Navigation and Context Management ===
# with st.sidebar:
#     st.header("🎯 Analysis Dashboard")
    
#     analysis_mode = st.selectbox(
#         "Choose Analysis Mode:",
#         ["Label Performance Prediction", "Revenue Comparison", "Comprehensive Analysis"]
#     )
    
#     st.header("🗄️ MongoDB Connection")
    
#     # MongoDB connection status and controls
#     if st.button("🔄 Refresh MongoDB Data"):
#         st.cache_data.clear()
    
#     # Show MongoDB stats
#     mongo_stats = get_mongodb_stats()
#     if mongo_stats and 'error' not in mongo_stats:
#         st.success(f"📊 DB: {mongo_stats.get('database_name', 'N/A')}")
#         st.success(f"📁 Collection: {mongo_stats.get('collection_name', 'N/A')}")
#         st.success(f"📄 Documents: {mongo_stats.get('total_documents', 0):,}")
        
#         if 'sample_fields' in mongo_stats:
#             with st.expander("🔍 View Data Fields"):
#                 st.write("Available fields in MongoDB:")
#                 for field in mongo_stats['sample_fields']:
#                     st.write(f"• {field}")
#     elif mongo_stats and 'error' in mongo_stats:
#         st.error(f"MongoDB Error: {mongo_stats['error']}")
    
#     st.header("📁 Data Upload")
#     st.write("**Revenue Data** (CSV file)")
#     revenue_file = st.file_uploader("Upload Revenue CSV", type=["csv"], key="revenue")
    
#     st.header("🧠 Context Memory")
    
#     if st.button("🔄 Clear Session Data"):
#         st.session_state.conversation_history = []
#         st.session_state.analysis_context = {}
#         st.session_state.label_performance_data = None
#         st.session_state.revenue_comparison_data = None
#         st.success("Session data cleared!")
    
#     st.subheader("📊 Session Stats")
#     st.metric("Queries Asked", len(st.session_state.conversation_history))
#     st.metric("Session ID", st.session_state.session_id)

# # === Main App ===

# # Load data - YouTube data from MongoDB, revenue from uploaded file
# with st.spinner("🔄 Loading YouTube data from MongoDB..."):
#     youtube_df = load_youtube_metadata_from_mongodb()
# revenue_df = pd.DataFrame()

# if revenue_file:
#     try:
#         revenue_df = pd.read_csv(revenue_file)
#         st.sidebar.success(f"✅ Revenue data loaded: {len(revenue_df)} records")
#     except Exception as e:
#         st.sidebar.error(f"Error loading revenue file: {e}")

# # Main analysis section
# if not youtube_df.empty:
#     st.success(f"✅ YouTube data loaded: {len(youtube_df)} videos from {youtube_df.get('Record Label', pd.Series()).nunique()} labels")
    
#     # === LABEL PERFORMANCE PREDICTION SECTION ===
#     if analysis_mode in ["Label Performance Prediction", "Comprehensive Analysis"]:
#         st.header("🏆 Label Performance Prediction & Ranking")
        
#         # Calculate performance metrics
#         with st.spinner("Calculating label performance metrics..."):
#             label_performance = calculate_label_performance_metrics(youtube_df)
#             st.session_state.label_performance_data = label_performance
        
#         # Display top performers
#         st.subheader("🥇 Top Performing Labels")
        
#         # Performance metrics display
#         col1, col2, col3, col4 = st.columns(4)
        
#         top_performer = label_performance.iloc[0]
        
#         with col1:
#             st.metric("🏆 Top Performer", top_performer['Record Label'])
#         with col2:
#             st.metric("📊 Performance Score", f"{top_performer['performance_score']:.1f}/100")
#         with col3:
#             st.metric("👀 Total Views", f"{top_performer['total_views']:,.0f}")
#         with col4:
#             st.metric("🎥 Total Videos", f"{top_performer['total_videos']:.0f}")
        
#         # Performance visualization
#         fig_performance = px.bar(
#             label_performance.head(10), 
#             x='performance_score', 
#             y='Record Label',
#             orientation='h',
#             title="Top 10 Labels by Performance Score",
#             color='performance_score',
#             color_continuous_scale='Viridis'
#         )
#         fig_performance.update_layout(height=500)
#         st.plotly_chart(fig_performance, use_container_width=True)
        
#         # Detailed performance table
#         st.subheader("📊 Detailed Performance Metrics")
#         display_cols = ['Record Label', 'performance_score', 'performance_tier', 'total_views', 
#                        'total_videos', 'avg_views_per_video', 'engagement_rate', 'consistency_score']
#         st.dataframe(label_performance[display_cols], use_container_width=True)
    
#     # === REVENUE PREDICTION AND COMPARISON SECTION ===
#     if analysis_mode in ["Revenue Comparison", "Comprehensive Analysis"]:
#         st.header("💰 Revenue Prediction & Comparison")
        
#         # RPM input
#         rpm = st.number_input("💸 RPM (Revenue per Million Views)", min_value=1000, max_value=500000, value=125000)
        
#         # Calculate revenue predictions
#         with st.spinner("Calculating revenue predictions..."):
#             revenue_predictions = predict_revenue_for_labels(youtube_df, rpm)
        
#         # Display revenue predictions
#         st.subheader("💹 Predicted Revenue by Label")
        
#         col1, col2, col3 = st.columns(3)
        
#         top_revenue_label = revenue_predictions.iloc[0]
        
#         with col1:
#             st.metric("💰 Top Revenue Label", top_revenue_label['Record Label'])
#         with col2:
#             st.metric("💵 Predicted Revenue", f"₹{top_revenue_label['estimated_revenue_inr']:,.0f}")
#         with col3:
#             st.metric("📊 Avg Revenue/Video", f"₹{top_revenue_label['avg_revenue_per_video']:,.0f}")
        
#         # Revenue visualization
#         fig_revenue = px.bar(
#             revenue_predictions.head(10),
#             x='estimated_revenue_inr',
#             y='Record Label',
#             orientation='h',
#             title="Top 10 Labels by Predicted Revenue",
#             color='estimated_revenue_inr',
#             color_continuous_scale='Blues'
#         )
#         fig_revenue.update_layout(height=500)
#         st.plotly_chart(fig_revenue, use_container_width=True)
        
#         # Label selection for detailed comparison
#         if not revenue_df.empty:
#             st.subheader("🔍 Detailed Revenue Comparison")
            
#             selected_label = st.selectbox(
#                 "Select a label for detailed revenue comparison:",
#                 options=revenue_predictions['Record Label'].tolist()
#             )
            
#             if selected_label:
#                 # Get estimated revenue for selected label
#                 estimated_revenue = revenue_predictions[
#                     revenue_predictions['Record Label'] == selected_label
#                 ]['estimated_revenue_inr'].iloc[0]
                
#                 # Compare with actual revenue data
#                 comparison_data = compare_actual_vs_estimated_revenue(
#                     revenue_df, estimated_revenue, selected_label
#                 )
                
#                 st.session_state.revenue_comparison_data = comparison_data
                
#                 # Display comparison results
#                 if comparison_data:
#                     st.subheader(f"📊 Revenue Analysis: {selected_label}")
                    
#                     comparison_cols = st.columns(4)
                    
#                     with comparison_cols[0]:
#                         st.metric("🎯 Estimated Revenue", f"₹{estimated_revenue:,.0f}")
                    
#                     if 'actual_revenue' in comparison_data:
#                         with comparison_cols[1]:
#                             st.metric("💼 Actual Revenue", f"₹{comparison_data['actual_revenue']:,.0f}")
#                         with comparison_cols[2]:
#                             st.metric("📈 Accuracy", f"{comparison_data['accuracy_percentage']:.1f}%")
#                         with comparison_cols[3]:
#                             variance_color = "normal" if abs(comparison_data['variance_percentage']) < 20 else "inverse"
#                             st.metric("📊 Variance", f"{comparison_data['variance_percentage']:+.1f}%")
                    
#                     # Comparison visualization
#                     if 'actual_revenue' in comparison_data:
#                         fig_comparison = go.Figure(data=[
#                             go.Bar(name='Estimated', x=[selected_label], y=[estimated_revenue]),
#                             go.Bar(name='Actual', x=[selected_label], y=[comparison_data['actual_revenue']])
#                         ])
#                         fig_comparison.update_layout(
#                             title=f"Revenue Comparison: {selected_label}",
#                             barmode='group',
#                             yaxis_title="Revenue (INR)"
#                         )
#                         st.plotly_chart(fig_comparison, use_container_width=True)
#         else:
#             st.info("📄 Upload revenue CSV file in the sidebar to enable detailed revenue comparison")
    
#     # === AI ANALYSIS SECTION ===
#     st.header("🧠 AI-Powered Analysis")
    
#     user_query = st.text_area(
#         "Ask me anything about label performance, revenue predictions, or comparative analysis:",
#         placeholder="E.g., Which labels are underperforming? How accurate are our revenue predictions? What strategies should top labels adopt?"
#     )
    
#     if user_query and API_KEY:
#         with st.spinner("🤖 Analyzing with AI..."):
#             # Prepare data for analysis
#             label_performance = st.session_state.label_performance_data
#             if label_performance is None:
#                 label_performance = calculate_label_performance_metrics(youtube_df)
            
#             revenue_predictions = predict_revenue_for_labels(youtube_df, rpm if 'rpm' in locals() else 125000)
#             comparison_data = st.session_state.revenue_comparison_data or {}
            
#             # Generate comprehensive prompt
#             full_prompt = generate_comprehensive_analysis_prompt(
#                 user_query, label_performance, revenue_predictions, 
#                 comparison_data, youtube_df, revenue_df
#             )
            
#             # Get AI response
#             response, error = get_mistral_analysis(full_prompt, API_KEY)
            
#             if error:
#                 st.error(f"AI Analysis Error: {error}")
#             else:
#                 st.markdown("### 🎯 AI Analysis Results")
#                 st.markdown(response)
                
#                 # Add to conversation history
#                 add_to_conversation_history(user_query, response)
    
#     elif user_query and not API_KEY:
#         st.warning("⚠️ Please set your OpenRouter API key in the .env file to use AI analysis")
    
#     # === DATA EXPORT SECTION ===
#     st.header("📥 Data Export")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.session_state.label_performance_data is not None:
#             st.download_button(
#                 "📊 Export Performance Data",
#                 st.session_state.label_performance_data.to_csv(index=False),
#                 "label_performance_analysis.csv",
#                 "text/csv"
#             )
    
#     with col2:
#         revenue_predictions = predict_revenue_for_labels(youtube_df, rpm if 'rpm' in locals() else 125000)
#         st.download_button(
#             "💰 Export Revenue Predictions",
#             revenue_predictions.to_csv(index=False),
#             "revenue_predictions.csv",
#             "text/csv"
#         )
    
#     with col3:
#         if st.session_state.conversation_history:
#             conversation_export = pd.DataFrame(st.session_state.conversation_history)
#             st.download_button(
#                 "💬 Export Analysis History",
#                 conversation_export.to_csv(index=False),
#                 f"analysis_history_{st.session_state.session_id}.csv",
#                 "text/csv"
#             )

# else:
#     st.warning("📁 Failed to load YouTube data from MongoDB. Please check your MongoDB connection and configuration.")
    
#     # Show MongoDB configuration help
#     st.subheader("🔧 MongoDB Configuration")
    
#     st.markdown("""
#     **Required Environment Variables:**
    
#     Create a `.env` file in your project directory with:
    
#     ```env
#     # MongoDB Configuration
#     MONGO_URI=mongodb://localhost:27017/
#     # For MongoDB Atlas: mongodb+srv://username:password@cluster.mongodb.net/
#     # For authenticated local: mongodb://username:password@localhost:27017/
    
#     MONGO_DB_NAME=youtube_analytics
#     MONGO_COLLECTION_NAME=youtube_metadata
    
#     # OpenRouter API Key (optional, for AI analysis)
#     OPENROUTER_API_KEY=your_api_key_here
#     ```
    
#     **Expected MongoDB Document Structure:**
    
#     **Option 1: Wrapper format**
#     ```json
#     {
#         "videos": [
#             {
#                 "title": "Song Title",
#                 "channel_title": "Channel Name",
#                 "view_count": 1000000,
#                 "like_count": 50000,
#                 "comment_count": 5000,
#                 "published_at": "2024-01-01T00:00:00Z",
#                 "description": "Song description..."
#             }
#         ]
#     }
#     ```
    
#     **Option 2: Direct video documents**
#     ```json
#     {
#         "title": "Song Title",
#         "channel_title": "Channel Name",
#         "view_count": 1000000,
#         "like_count": 50000,
#         "comment_count": 5000,
#         "published_at": "2024-01-01T00:00:00Z",
#         "description": "Song description..."
#     }
#     ```
#     """)
    
#     # Show current configuration
#     st.subheader("📋 Current Configuration")
#     st.code(f"""
#     MongoDB URI: {MONGO_URI}
#     Database Name: {MONGO_DB_NAME}
#     Collection Name: {MONGO_COLLECTION_NAME}
#     """)
    
#     # Show example data structure for revenue CSV
#     st.subheader("📋 Expected Revenue CSV Structure")
    
#     st.markdown("""
#     **Revenue CSV Structure:**
#     ```
#     Store Name,Annual Revenue in INR,Platform,Label
#     YouTube,50000000,Digital,T-Series
#     Spotify,25000000,Streaming,Sony Music
#     Apple Music,15000000,Streaming,Zee Music
#     ```
    
#     Upload your revenue CSV file using the sidebar to enable detailed revenue comparison and analysis.
#     """)

# # === FOOTER ===
# st.markdown("---")
# st.markdown("🎵 **Smart Data Analyzer Pro** - Advanced YouTube Label Performance & Revenue Analytics")




# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import warnings
# warnings.filterwarnings('ignore')

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# # Load NLP model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     st.error("Please install spacy English model: python -m spacy download en_core_web_sm")
#     nlp = None

# st.set_page_config(page_title="Smart Data Analyzer Pro - Label Performance Predictor", layout="wide")
# st.title("🧠 Smart Data Analyzer Pro — YouTube Label Performance & Revenue Validation")

# # Initialize session state
# if 'conversation_history' not in st.session_state:
#     st.session_state.conversation_history = []
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
# if 'label_performance_data' not in st.session_state:
#     st.session_state.label_performance_data = None
# if 'revenue_comparison_data' not in st.session_state:
#     st.session_state.revenue_comparison_data = None

# def calculate_label_performance_metrics(youtube_df):
#     """Calculate comprehensive performance metrics for each label"""
    
#     # Group by record label and calculate metrics
#     label_metrics = youtube_df.groupby('Record Label').agg({
#         'view_count': ['sum', 'mean', 'count', 'std'],
#         'like_count': ['sum', 'mean'],
#         'comment_count': ['sum', 'mean'],
#         'published_at': ['min', 'max']
#     }).round(2)
    
#     # Flatten column names
#     label_metrics.columns = ['_'.join(col).strip() for col in label_metrics.columns]
#     label_metrics = label_metrics.reset_index()
    
#     # Calculate additional performance metrics
#     label_metrics['total_videos'] = label_metrics['view_count_count']
#     label_metrics['avg_views_per_video'] = label_metrics['view_count_mean']
#     label_metrics['total_views'] = label_metrics['view_count_sum']
#     label_metrics['engagement_rate'] = (label_metrics['like_count_sum'] + label_metrics['comment_count_sum']) / label_metrics['view_count_sum'] * 100
    
#     # Calculate consistency score (lower std deviation relative to mean = more consistent)
#     label_metrics['consistency_score'] = 100 - (label_metrics['view_count_std'] / label_metrics['view_count_mean'] * 100).fillna(0)
#     label_metrics['consistency_score'] = label_metrics['consistency_score'].clip(0, 100)
    
#     # Calculate performance score (weighted combination of metrics)
#     scaler = StandardScaler()
#     metrics_for_scoring = ['total_views', 'avg_views_per_video', 'engagement_rate', 'consistency_score', 'total_videos']
    
#     # Handle missing values
#     for col in metrics_for_scoring:
#         label_metrics[col] = label_metrics[col].fillna(label_metrics[col].median())
    
#     # Normalize metrics for scoring
#     normalized_metrics = scaler.fit_transform(label_metrics[metrics_for_scoring])
    
#     # Calculate weighted performance score
#     weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Adjust weights as needed
#     label_metrics['performance_score'] = np.dot(normalized_metrics, weights)
    
#     # Normalize performance score to 0-100 scale
#     min_score = label_metrics['performance_score'].min()
#     max_score = label_metrics['performance_score'].max()
#     label_metrics['performance_score'] = ((label_metrics['performance_score'] - min_score) / (max_score - min_score) * 100).round(2)
    
#     # Add performance tier
#     label_metrics['performance_tier'] = pd.cut(label_metrics['performance_score'], 
#                                              bins=[0, 33, 66, 100], 
#                                              labels=['Bronze', 'Silver', 'Gold'])
    
#     # Sort by performance score
#     label_metrics = label_metrics.sort_values('performance_score', ascending=False)
    
#     return label_metrics

# def predict_revenue_for_labels(youtube_df, rpm_value):
#     """Predict revenue for all labels based on views and RPM"""
#     label_revenue = youtube_df.groupby('Record Label').agg({
#         'view_count': 'sum',
#         'published_at': 'count'
#     }).rename(columns={'published_at': 'video_count'}).reset_index()
    
#     # Calculate estimated revenue
#     label_revenue['estimated_revenue_inr'] = (label_revenue['view_count'] / 1_000_000) * rpm_value
#     label_revenue['avg_revenue_per_video'] = label_revenue['estimated_revenue_inr'] / label_revenue['video_count']
    
#     # Sort by estimated revenue
#     label_revenue = label_revenue.sort_values('estimated_revenue_inr', ascending=False)
    
#     return label_revenue

# def compare_actual_vs_estimated_revenue(revenue_df, estimated_revenue, selected_label):
#     """Compare actual vs estimated revenue for selected label"""
#     comparison_data = {}
    
#     # Try to find matching revenue data
#     if 'Store Name' in revenue_df.columns:
#         # Look for YouTube revenue data
#         youtube_rows = revenue_df[revenue_df['Store Name'].str.lower().str.contains('youtube', na=False)]
        
#         if not youtube_rows.empty:
#             actual_revenue = youtube_rows['Annual Revenue in INR'].sum()
#             comparison_data['actual_revenue'] = actual_revenue
#             comparison_data['estimated_revenue'] = estimated_revenue
#             comparison_data['accuracy_percentage'] = (estimated_revenue / actual_revenue * 100) if actual_revenue > 0 else 0
#             comparison_data['variance'] = estimated_revenue - actual_revenue
#             comparison_data['variance_percentage'] = ((estimated_revenue - actual_revenue) / actual_revenue * 100) if actual_revenue > 0 else 0
    
#     # Try to find label-specific revenue data
#     label_specific_rows = revenue_df[revenue_df.apply(lambda x: selected_label.lower() in str(x).lower(), axis=1)]
    
#     if not label_specific_rows.empty and 'Annual Revenue in INR' in revenue_df.columns:
#         label_actual_revenue = label_specific_rows['Annual Revenue in INR'].sum()
#         comparison_data['label_actual_revenue'] = label_actual_revenue
#         comparison_data['label_accuracy_percentage'] = (estimated_revenue / label_actual_revenue * 100) if label_actual_revenue > 0 else 0
#         comparison_data['label_variance'] = estimated_revenue - label_actual_revenue
#         comparison_data['label_variance_percentage'] = ((estimated_revenue - label_actual_revenue) / label_actual_revenue * 100) if label_actual_revenue > 0 else 0
    
#     return comparison_data

# def generate_comprehensive_analysis_prompt(user_query, label_performance, revenue_predictions, comparison_data, youtube_df, revenue_df):
#     """Generate comprehensive prompt for AI analysis"""
    
#     # Get top performing labels
#     top_labels = label_performance.head(5)[['Record Label', 'performance_score', 'total_views', 'total_videos', 'engagement_rate']].to_dict('records')
    
#     # Get revenue predictions
#     top_revenue_labels = revenue_predictions.head(5)[['Record Label', 'estimated_revenue_inr', 'view_count', 'video_count']].to_dict('records')
    
#     # Get conversation context
#     conversation_context = get_conversation_context()
    
#     return f"""
# You are a senior Business Intelligence analyst specializing in music industry analytics and YouTube performance prediction.

# {conversation_context}

# 🏆 **TOP PERFORMING LABELS BY PERFORMANCE SCORE:**
# {json.dumps(top_labels, indent=2)}

# 💰 **TOP REVENUE GENERATING LABELS (PREDICTED):**
# {json.dumps(top_revenue_labels, indent=2)}

# 📊 **REVENUE COMPARISON DATA:**
# {json.dumps(comparison_data, indent=2)}

# 📈 **OVERALL YOUTUBE DATASET SUMMARY:**
# - Total Labels: {youtube_df['Record Label'].nunique()}
# - Total Videos: {len(youtube_df)}
# - Total Views: {youtube_df['view_count'].sum():,}
# - Average Views per Video: {youtube_df['view_count'].mean():,.0f}

# 💼 **REVENUE DATASET INFO:**
# - Revenue Sources: {revenue_df.columns.tolist() if not revenue_df.empty else 'No revenue data available'}
# - Total Revenue Records: {len(revenue_df) if not revenue_df.empty else 0}

# ---

# ### Current User Question:
# "{user_query}"

# ---

# ANALYSIS INSTRUCTIONS:
# 1. **Label Performance Analysis**: Identify top-performing labels based on comprehensive metrics
# 2. **Revenue Prediction**: Analyze predicted vs actual revenue patterns
# 3. **Comparative Analysis**: Compare performance across different labels and revenue streams
# 4. **Market Insights**: Provide insights about label market positioning
# 5. **Strategic Recommendations**: Suggest data-driven strategies for label growth
# 6. **Accuracy Assessment**: Evaluate prediction accuracy and model reliability

# Provide detailed, data-driven insights with specific numbers and actionable recommendations.
# Use markdown formatting with clear sections and bullet points.
# """

# def add_to_conversation_history(query, response, insights=None, charts_generated=None):
#     """Add query and response to conversation history with metadata"""
#     entry = {
#         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'query': query,
#         'response': response,
#         'insights': insights or [],
#         'charts_generated': charts_generated or [],
#         'session_id': st.session_state.session_id
#     }
#     st.session_state.conversation_history.append(entry)
    
#     # Keep only last 10 conversations to manage memory
#     if len(st.session_state.conversation_history) > 10:
#         st.session_state.conversation_history = st.session_state.conversation_history[-10:]

# def get_conversation_context():
#     """Generate conversation context for AI prompt"""
#     if not st.session_state.conversation_history:
#         return ""
    
#     context = "\n📝 **Previous Conversation Context:**\n"
#     for i, entry in enumerate(st.session_state.conversation_history[-3:], 1):
#         context += f"\n**Q{i}:** {entry['query']}\n"
#         context += f"**A{i}:** {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}\n"
    
#     return context

# def get_mistral_analysis(prompt, api_key):
#     """Get analysis from Mistral AI"""
#     try:
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 2000
#         }
#         response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], None
#         return "", f"Error {response.status_code}: {response.text}"
#     except Exception as e:
#         return "", str(e)

# @st.cache_data
# def load_youtube_metadata():
#     """Load YouTube metadata with enhanced label detection"""
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)
#         df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
        
#         # Enhanced label detection
#         known_labels = [
#             "T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#             "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official",
#             "Eros Music", "Shemaroo", "Ultra Music", "Panorama Music", "Rajshri Music"
#         ]
        
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')} {row.get('channel_title', '')}".lower()
#             for label in known_labels:
#                 if label.lower() in text:
#                     return label
#             return "Independent/Other"
        
#         df["Record Label"] = df.apply(detect_label, axis=1)
        
#         # Ensure numeric columns
#         numeric_columns = ['view_count', 'like_count', 'comment_count']
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
#         # Parse dates
#         if 'published_at' in df.columns:
#             df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
#         return df
#     except Exception as e:
#         st.error(f"Error loading YouTube metadata: {e}")
#         return pd.DataFrame()

# # === Sidebar for Navigation and Context Management ===
# with st.sidebar:
#     st.header("🎯 Analysis Dashboard")
    
#     analysis_mode = st.selectbox(
#         "Choose Analysis Mode:",
#         ["Label Performance Prediction", "Revenue Comparison", "Comprehensive Analysis"]
#     )
    
#     st.header("🧠 Context Memory")
    
#     if st.button("🔄 Clear Session Data"):
#         st.session_state.conversation_history = []
#         st.session_state.analysis_context = {}
#         st.session_state.label_performance_data = None
#         st.session_state.revenue_comparison_data = None
#         st.success("Session data cleared!")
    
#     st.subheader("📊 Session Stats")
#     st.metric("Queries Asked", len(st.session_state.conversation_history))
#     st.metric("Session ID", st.session_state.session_id)

# # === Main App ===
# st.subheader("📁 Data Upload Section")

# # Load YouTube metadata from default file (youtube_metadata.json)
# youtube_df = load_youtube_metadata()

# if youtube_df.empty:
#     st.error("❌ YouTube metadata file (youtube_metadata.json) not found. Please ensure it exists in the current directory.")
#     st.stop()

# st.success(f"✅ YouTube database loaded: {len(youtube_df)} videos from {youtube_df.get('Record Label', pd.Series()).nunique()} labels")

# # User uploads only CSV file
# st.write("**📊 Upload Your Revenue Data (CSV file)**")
# revenue_file = st.file_uploader("Upload Revenue CSV", type=["csv"], key="revenue")

# revenue_df = pd.DataFrame()

# if revenue_file:
#     try:
#         revenue_df = pd.read_csv(revenue_file)
#         st.success(f"✅ Revenue data loaded: {len(revenue_df)} records")
#     except Exception as e:
#         st.error(f"Error loading revenue file: {e}")

# # Main analysis section - YouTube data is always available from default file
#     st.success(f"✅ YouTube data loaded: {len(youtube_df)} videos from {youtube_df.get('Record Label', pd.Series()).nunique()} labels")
    
# # === LABEL PERFORMANCE PREDICTION SECTION ===
# if analysis_mode in ["Label Performance Prediction", "Comprehensive Analysis"]:
#     st.header("🏆 Label Performance Prediction & Ranking")
    
#     # Calculate performance metrics
#     with st.spinner("Calculating label performance metrics..."):
#         label_performance = calculate_label_performance_metrics(youtube_df)
#         st.session_state.label_performance_data = label_performance
    
#     # Display top performers
#     st.subheader("🥇 Top Performing Labels")
    
#     # Performance metrics display
#     col1, col2, col3, col4 = st.columns(4)
    
#     top_performer = label_performance.iloc[0]
    
#     with col1:
#         st.metric("🏆 Top Performer", top_performer['Record Label'])
#     with col2:
#         st.metric("📊 Performance Score", f"{top_performer['performance_score']:.1f}/100")
#     with col3:
#         st.metric("👀 Total Views", f"{top_performer['total_views']:,.0f}")
#     with col4:
#         st.metric("🎥 Total Videos", f"{top_performer['total_videos']:.0f}")
    
#     # Performance visualization
#     fig_performance = px.bar(
#         label_performance.head(10), 
#         x='performance_score', 
#         y='Record Label',
#         orientation='h',
#         title="Top 10 Labels by Performance Score",
#         color='performance_score',
#         color_continuous_scale='Viridis'
#     )
#     fig_performance.update_layout(height=500)
#     st.plotly_chart(fig_performance, use_container_width=True)
    
#     # Detailed performance table
#     st.subheader("📊 Detailed Performance Metrics")
#     display_cols = ['Record Label', 'performance_score', 'performance_tier', 'total_views', 
#                    'total_videos', 'avg_views_per_video', 'engagement_rate', 'consistency_score']
#     st.dataframe(label_performance[display_cols], use_container_width=True)
    
# # === REVENUE PREDICTION AND COMPARISON SECTION ===
# if analysis_mode in ["Revenue Comparison", "Comprehensive Analysis"]:
#     st.header("💰 Revenue Prediction & Comparison")
    
#     # RPM input
#     rpm = st.number_input("💸 RPM (Revenue per Million Views)", min_value=1000, max_value=500000, value=125000)
    
#     # Calculate revenue predictions
#     with st.spinner("Calculating revenue predictions..."):
#         revenue_predictions = predict_revenue_for_labels(youtube_df, rpm)
    
#     # Display revenue predictions
#     st.subheader("💹 Predicted Revenue by Label")
    
#     col1, col2, col3 = st.columns(3)
    
#     top_revenue_label = revenue_predictions.iloc[0]
    
#     with col1:
#         st.metric("💰 Top Revenue Label", top_revenue_label['Record Label'])
#     with col2:
#         st.metric("💵 Predicted Revenue", f"₹{top_revenue_label['estimated_revenue_inr']:,.0f}")
#     with col3:
#         st.metric("📊 Avg Revenue/Video", f"₹{top_revenue_label['avg_revenue_per_video']:,.0f}")
    
#     # Revenue visualization
#     fig_revenue = px.bar(
#         revenue_predictions.head(10),
#         x='estimated_revenue_inr',
#         y='Record Label',
#         orientation='h',
#         title="Top 10 Labels by Predicted Revenue",
#         color='estimated_revenue_inr',
#         color_continuous_scale='Blues'
#     )
#     fig_revenue.update_layout(height=500)
#     st.plotly_chart(fig_revenue, use_container_width=True)
    
#     # Label selection for detailed comparison
#     if not revenue_df.empty:
#         st.subheader("🔍 Detailed Revenue Comparison")
        
#         selected_label = st.selectbox(
#             "Select a label for detailed revenue comparison:",
#             options=revenue_predictions['Record Label'].tolist()
#         )
        
#         if selected_label:
#             # Get estimated revenue for selected label
#             estimated_revenue = revenue_predictions[
#                 revenue_predictions['Record Label'] == selected_label
#             ]['estimated_revenue_inr'].iloc[0]
            
#             # Compare with actual revenue data
#             comparison_data = compare_actual_vs_estimated_revenue(
#                 revenue_df, estimated_revenue, selected_label
#             )
            
#             st.session_state.revenue_comparison_data = comparison_data
            
#             # Display comparison results
#             if comparison_data:
#                 st.subheader(f"📊 Revenue Analysis: {selected_label}")
                
#                 comparison_cols = st.columns(4)
                
#                 with comparison_cols[0]:
#                     st.metric("🎯 Estimated Revenue", f"₹{estimated_revenue:,.0f}")
                
#                 if 'actual_revenue' in comparison_data:
#                     with comparison_cols[1]:
#                         st.metric("💼 Actual Revenue", f"₹{comparison_data['actual_revenue']:,.0f}")
#                     with comparison_cols[2]:
#                         st.metric("📈 Accuracy", f"{comparison_data['accuracy_percentage']:.1f}%")
#                     with comparison_cols[3]:
#                         variance_color = "normal" if abs(comparison_data['variance_percentage']) < 20 else "inverse"
#                         st.metric("📊 Variance", f"{comparison_data['variance_percentage']:+.1f}%")
                
#                 # Comparison visualization
#                 if 'actual_revenue' in comparison_data:
#                     fig_comparison = go.Figure(data=[
#                         go.Bar(name='Estimated', x=[selected_label], y=[estimated_revenue]),
#                         go.Bar(name='Actual', x=[selected_label], y=[comparison_data['actual_revenue']])
#                     ])
#                     fig_comparison.update_layout(
#                         title=f"Revenue Comparison: {selected_label}",
#                         barmode='group',
#                         yaxis_title="Revenue (INR)"
#                     )
#                     st.plotly_chart(fig_comparison, use_container_width=True)
#     else:
#         st.info("📄 Upload revenue CSV file to enable detailed revenue comparison")
    
#     # === AI ANALYSIS SECTION ===
#     st.header("🧠 AI-Powered Analysis")
    
#     user_query = st.text_area(
#         "Ask me anything about label performance, revenue predictions, or comparative analysis:",
#         placeholder="E.g., Which labels are underperforming? How accurate are our revenue predictions? What strategies should top labels adopt?"
#     )
    
#     if user_query and API_KEY:
#         with st.spinner("🤖 Analyzing with AI..."):
#             # Prepare data for analysis
#             label_performance = st.session_state.label_performance_data
#             if label_performance is None:
#                 label_performance = calculate_label_performance_metrics(youtube_df)
            
#             revenue_predictions = predict_revenue_for_labels(youtube_df, rpm if 'rpm' in locals() else 125000)
#             comparison_data = st.session_state.revenue_comparison_data or {}
            
#             # Generate comprehensive prompt
#             full_prompt = generate_comprehensive_analysis_prompt(
#                 user_query, label_performance, revenue_predictions, 
#                 comparison_data, youtube_df, revenue_df
#             )
            
#             # Get AI response
#             response, error = get_mistral_analysis(full_prompt, API_KEY)
            
#             if error:
#                 st.error(f"AI Analysis Error: {error}")
#             else:
#                 st.markdown("### 🎯 AI Analysis Results")
#                 st.markdown(response)
                
#                 # Add to conversation history
#                 add_to_conversation_history(user_query, response)
    
#     elif user_query and not API_KEY:
#         st.warning("⚠️ Please set your OpenRouter API key in the .env file to use AI analysis")
    
#     # === DATA EXPORT SECTION ===
#     st.header("📥 Data Export")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.session_state.label_performance_data is not None:
#             st.download_button(
#                 "📊 Export Performance Data",
#                 st.session_state.label_performance_data.to_csv(index=False),
#                 "label_performance_analysis.csv",
#                 "text/csv"
#             )
    
#     with col2:
#         revenue_predictions = predict_revenue_for_labels(youtube_df, rpm if 'rpm' in locals() else 125000)
#         st.download_button(
#             "💰 Export Revenue Predictions",
#             revenue_predictions.to_csv(index=False),
#             "revenue_predictions.csv",
#             "text/csv"
#         )
    
#     with col3:
#         if st.session_state.conversation_history:
#             conversation_export = pd.DataFrame(st.session_state.conversation_history)
#             st.download_button(
#                 "💬 Export Analysis History",
#                 conversation_export.to_csv(index=False),
#                 f"analysis_history_{st.session_state.session_id}.csv",
#                 "text/csv"
#             )

# else:
#     st.info("📁 Please upload YouTube metadata JSON file or ensure youtube_metadata.json exists in the current directory")
    
#     # Show example data structure
#     st.subheader("📋 Expected Data Structure")
    
#     st.markdown("""
#     **YouTube Metadata JSON Structure:**
#     ```json
#     {
#         "videos": [
#             {
#                 "title": "Song Title",
#                 "channel_title": "Channel Name",
#                 "view_count": 1000000,
#                 "like_count": 50000,
#                 "comment_count": 5000,
#                 "published_at": "2024-01-01T00:00:00Z",
#                 "description": "Song description..."
#             }
#         ]
#     }
#     ```
    
#     **Revenue CSV Structure:**
#     ```
#     Store Name,Annual Revenue in INR,Platform,Label
#     YouTube,50000000,Digital,T-Series
#     Spotify,25000000,Streaming,Sony Music
#     ```
#     """)

# # === FOOTER ===
# st.markdown("---")
# st.markdown("🎵 **Smart Data Analyzer Pro** - Advanced YouTube Label Performance & Revenue Analytics")



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import re

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# # Load NLP model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except:
#     st.error("Please install spacy English model: python -m spacy download en_core_web_sm")
#     nlp = None

# st.set_page_config(page_title="Interactive Data Analyzer", layout="wide")
# st.title("💬 Interactive YouTube Revenue Analyzer")

# # Initialize session state
# if 'chat_messages' not in st.session_state:
#     st.session_state.chat_messages = []
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'label_videos' not in st.session_state:
#     st.session_state.label_videos = pd.DataFrame()
# if 'monthly_revenue' not in st.session_state:
#     st.session_state.monthly_revenue = pd.DataFrame()
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'pending_clarification' not in st.session_state:
#     st.session_state.pending_clarification = False
# if 'last_analysis_data' not in st.session_state:
#     st.session_state.last_analysis_data = {}

# def add_message(role, content, charts=None, data=None):
#     """Add message to chat history"""
#     message = {
#         'role': role,
#         'content': content,
#         'timestamp': datetime.now().strftime("%H:%M:%S"),
#         'charts': charts or [],
#         'data': data
#     }
#     st.session_state.chat_messages.append(message)

# def create_smart_visualizations(analysis_response, data_context):
#     """Automatically create relevant visualizations based on AI analysis"""
#     charts = []
    
#     # Extract chart instructions from AI response
#     chart_patterns = {
#         'monthly_trend': r'(?i)(monthly|month|trend|time|temporal|seasonal)',
#         'top_videos': r'(?i)(top|best|highest|ranking|performance)',
#         'revenue_distribution': r'(?i)(distribution|spread|variance|compare)',
#         'growth_analysis': r'(?i)(growth|increase|decrease|change)',
#         'correlation': r'(?i)(correlation|relationship|vs|versus)',
#         'breakdown': r'(?i)(breakdown|category|segment|group)'
#     }
    
#     response_lower = analysis_response.lower()
    
#     # Monthly Revenue Trend
#     if re.search(chart_patterns['monthly_trend'], response_lower) and not st.session_state.monthly_revenue.empty:
#         fig = px.line(
#             st.session_state.monthly_revenue, 
#             x="Month", 
#             y="Estimated Revenue INR",
#             title="📈 Monthly Revenue Trend",
#             markers=True
#         )
#         fig.update_layout(height=400)
#         charts.append(('Monthly Revenue Trend', fig, st.session_state.monthly_revenue))
    
#     # Top Videos Performance
#     if re.search(chart_patterns['top_videos'], response_lower) and not st.session_state.label_videos.empty:
#         top_videos = st.session_state.label_videos.nlargest(10, 'Estimated Revenue INR')
#         fig = px.bar(
#             top_videos,
#             x='Estimated Revenue INR',
#             y='title',
#             orientation='h',
#             title="🏆 Top 10 Videos by Revenue"
#         )
#         fig.update_layout(height=500)
#         charts.append(('Top Videos Revenue', fig, top_videos[['title', 'view_count', 'Estimated Revenue INR']]))
    
#     # RPV Analysis
#     if 'rpv' in response_lower or 'revenue per view' in response_lower:
#         if not st.session_state.label_videos.empty:
#             st.session_state.label_videos['RPV'] = st.session_state.label_videos['Estimated Revenue INR'] / st.session_state.label_videos['view_count']
#             top_rpv = st.session_state.label_videos.nlargest(10, 'RPV')
#             fig = px.scatter(
#                 top_rpv,
#                 x='view_count',
#                 y='RPV',
#                 size='Estimated Revenue INR',
#                 hover_data=['title'],
#                 title="💰 Revenue Per View vs Views"
#             )
#             charts.append(('RPV Analysis', fig, top_rpv[['title', 'view_count', 'RPV', 'Estimated Revenue INR']]))
    
#     # Revenue Distribution
#     if re.search(chart_patterns['revenue_distribution'], response_lower) and not st.session_state.label_videos.empty:
#         fig = px.histogram(
#             st.session_state.label_videos,
#             x='Estimated Revenue INR',
#             nbins=20,
#             title="📊 Revenue Distribution"
#         )
#         charts.append(('Revenue Distribution', fig, None))
    
#     # Views vs Revenue Correlation
#     if re.search(chart_patterns['correlation'], response_lower) and not st.session_state.label_videos.empty:
#         fig = px.scatter(
#             st.session_state.label_videos,
#             x='view_count',
#             y='Estimated Revenue INR',
#             title="🔗 Views vs Revenue Correlation",
#             trendline="ols"
#         )
#         charts.append(('Views vs Revenue', fig, None))
    
#     return charts

# def extract_data_requirements(user_query):
#     """Extract what data the user is asking about"""
#     query_lower = user_query.lower()
#     requirements = {
#         'needs_monthly_data': any(word in query_lower for word in ['monthly', 'month', 'trend', 'seasonal', 'time']),
#         'needs_top_videos': any(word in query_lower for word in ['top', 'best', 'highest', 'ranking']),
#         'needs_comparison': any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']),
#         'needs_specific_metric': any(word in query_lower for word in ['rpm', 'rpv', 'cpm', 'revenue per']),
#         'needs_growth_data': any(word in query_lower for word in ['growth', 'increase', 'decrease', 'change']),
#         'unclear_intent': len([word for word in ['what', 'how', 'why', 'which', 'when'] if word in query_lower]) > 1
#     }
#     return requirements

# @st.cache_data
# def load_youtube_metadata():
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)
#         df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
#         known_labels = ["T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#                         "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"]
        
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#             for label in known_labels:
#                 if label.lower() in text:
#                     return label
#             return "Other"
        
#         df["Record Label"] = df.apply(detect_label, axis=1)
#         return df
#     except FileNotFoundError:
#         st.error("youtube_metadata.json file not found!")
#         return pd.DataFrame()

# def generate_intelligent_prompt(user_query, data_context, conversation_history):
#     """Generate intelligent prompt that can ask for clarification"""
    
#     # Analyze user query
#     requirements = extract_data_requirements(user_query)
    
#     # Build context
#     context = f"""
# You are an expert Business Intelligence analyst with access to YouTube revenue data. You can:
# 1. Analyze data and provide insights
# 2. Ask for clarification when queries are ambiguous
# 3. Suggest specific visualizations
# 4. Provide actionable recommendations

# **Available Data:**
# - Total Videos: {len(st.session_state.label_videos)} videos
# - Revenue Period: {data_context.get('date_range', 'Multiple months')}
# - RPM: ₹{data_context.get('rpm', 'Not specified')}
# - Total Estimated Revenue: ₹{data_context.get('est_total', 0):,.2f}
# - Actual Revenue: ₹{data_context.get('actual_total', 0):,.2f}

# **Sample Data (Top 5 videos):**
# {st.session_state.label_videos.head()[['title', 'view_count', 'Estimated Revenue INR']].to_string() if not st.session_state.label_videos.empty else 'No data available'}

# **Recent Conversation Context:**
# {conversation_history}

# **User Query:** "{user_query}"

# **Analysis Requirements Detected:**
# {requirements}

# **INSTRUCTIONS:**
# 1. If the query is clear and you have sufficient data, provide detailed analysis with specific numbers
# 2. If the query is ambiguous or you need more information, ask ONE specific clarifying question
# 3. Always suggest relevant visualizations using phrases like "I should create a [chart_type] showing [data]"
# 4. Use markdown formatting for clear structure
# 5. Be conversational and helpful
# 6. If you detect data limitations, mention them and suggest alternatives

# **CLARIFICATION TRIGGERS:**
# - Ask for clarification if user says "analyze this" without specifying what
# - Ask which time period if multiple periods are available
# - Ask which metrics to focus on if query is too broad
# - Ask for comparison criteria if user wants comparisons

# Respond as a helpful analyst who can either provide insights OR ask for clarification when needed.
# """
    
#     return context

# def get_chatgpt_analysis(prompt, api_key):
#     """Get analysis from ChatGPT with clarification capabilities"""
#     try:
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
        
#         payload = {
#             "model": "openai/chatgpt-4o-latest",
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": """You are an expert Business Intelligence analyst. You can:
# 1. Provide detailed data analysis with specific numbers and insights
# 2. Ask clarifying questions when user queries are ambiguous
# 3. Suggest specific visualizations by mentioning chart types
# 4. Give actionable business recommendations

# When you need clarification, ask ONE specific question. When you have enough info, provide comprehensive analysis."""
#                 },
#                 {
#                     "role": "user", 
#                     "content": prompt
#                 }
#             ],
#             "temperature": 0.7,
#             "max_tokens": 2500
#         }
        
#         response = requests.post(
#             "https://openrouter.ai/api/v1/chat/completions",
#             headers=headers,
#             json=payload,
#             timeout=120
#         )
        
#         if response.status_code == 200:
#             result = response.json()
#             if "choices" in result and len(result["choices"]) > 0:
#                 return result["choices"][0]["message"]["content"], None
#             else:
#                 return "", "No response generated"
#         else:
#             return "", f"API Error {response.status_code}: {response.text}"
            
#     except Exception as e:
#         return "", f"Error: {str(e)}"

# def process_user_input(user_input):
#     """Process user input and generate response with potential clarification"""
#     if not st.session_state.data_loaded:
#         add_message("assistant", "⚠️ Please upload a revenue CSV file first to begin analysis.")
#         return
    
#     # Add user message
#     add_message("user", user_input)
    
#     # Build conversation history
#     conversation_history = ""
#     for msg in st.session_state.chat_messages[-5:]:  # Last 5 messages
#         conversation_history += f"{msg['role']}: {msg['content'][:100]}...\n"
    
#     # Prepare data context
#     data_context = {
#         'rpm': st.session_state.last_analysis_data.get('rpm', 0),
#         'est_total': st.session_state.last_analysis_data.get('est_total', 0),
#         'actual_total': st.session_state.last_analysis_data.get('actual_total', 0),
#         'date_range': 'Multiple months' if not st.session_state.monthly_revenue.empty else 'Single period'
#     }
    
#     # Generate prompt
#     prompt = generate_intelligent_prompt(user_input, data_context, conversation_history)
    
#     # Get AI response
#     with st.spinner("🤖 ChatGPT is analyzing..."):
#         response, error = get_chatgpt_analysis(prompt, API_KEY)
        
#         if error:
#             add_message("assistant", f"❌ Error: {error}")
#             return
        
#         # Check if AI is asking for clarification
#         clarification_indicators = ['?', 'clarify', 'specify', 'which', 'what do you mean', 'could you']
#         is_asking_clarification = any(indicator in response.lower() for indicator in clarification_indicators)
        
#         if is_asking_clarification:
#             st.session_state.pending_clarification = True
#             add_message("assistant", response)
#         else:
#             st.session_state.pending_clarification = False
            
#             # Generate visualizations based on AI response
#             charts = create_smart_visualizations(response, data_context)
            
#             # Add response with charts
#             add_message("assistant", response, charts=charts)

# def display_chat_interface():
#     """Display the chat interface"""
#     st.subheader("💬 Chat with AI Analyst")
    
#     # Display chat messages
#     chat_container = st.container()
#     with chat_container:
#         for message in st.session_state.chat_messages:
#             if message['role'] == 'user':
#                 with st.chat_message("user"):
#                     st.write(f"**You** ({message['timestamp']})")
#                     st.write(message['content'])
#             else:
#                 with st.chat_message("assistant"):
#                     st.write(f"**AI Analyst** ({message['timestamp']})")
#                     st.markdown(message['content'])
                    
#                     # Display charts if any
#                     if message.get('charts'):
#                         st.write("**📊 Generated Visualizations:**")
#                         for chart_name, fig, data in message['charts']:
#                             st.plotly_chart(fig, use_container_width=True)
#                             if data is not None and not data.empty:
#                                 with st.expander(f"📋 Data for {chart_name}"):
#                                     st.dataframe(data)
    
#     # Chat input
#     user_input = st.chat_input("Ask me anything about your YouTube revenue data...")
    
#     if user_input and API_KEY:
#         process_user_input(user_input)
#         st.rerun()
#     elif user_input and not API_KEY:
#         add_message("assistant", "❌ Please set your OpenRouter API key in the .env file")
#         st.rerun()

# # Sidebar for data loading
# with st.sidebar:
#     st.header("📊 Data Setup")
    
#     if API_KEY:
#         st.success("✅ ChatGPT 4.1 Ready")
#     else:
#         st.error("❌ Set OPENROUTER_API_KEY")
    
#     # File upload
#     uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])
    
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
        
#         # Load YouTube metadata
#         youtube_df = load_youtube_metadata()
        
#         if not youtube_df.empty:
#             selected_label = st.selectbox("🎙️ Record Label", sorted(youtube_df["Record Label"].unique()))
#             rpm = st.number_input("💸 RPM (₹)", min_value=500, value=125000)
            
#             if st.button("🚀 Load Data & Start Chat"):
#                 # Process data
#                 label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#                 label_videos = label_videos.dropna(subset=["view_count"])
#                 label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
                
#                 # Get actual revenue
#                 if "Store Name" in df.columns:
#                     yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#                     actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
#                 else:
#                     actual_total = 0
                
#                 # Process dates
#                 label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#                 label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#                 monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
                
#                 # Store in session state
#                 st.session_state.label_videos = label_videos
#                 st.session_state.monthly_revenue = monthly_revenue
#                 st.session_state.data_loaded = True
#                 st.session_state.last_analysis_data = {
#                     'rpm': rpm,
#                     'est_total': label_videos["Estimated Revenue INR"].sum(),
#                     'actual_total': actual_total,
#                     'selected_label': selected_label
#                 }
                
#                 # Clear previous chat and add welcome message
#                 st.session_state.chat_messages = []
#                 welcome_msg = f"""
# ## 🎉 Data Loaded Successfully!

# **Analysis Ready for:** {selected_label}
# - **Total Videos:** {len(label_videos):,}
# - **Estimated Revenue:** ₹{st.session_state.last_analysis_data['est_total']:,.2f}
# - **Actual Revenue:** ₹{actual_total:,.2f}
# - **RPM Used:** ₹{rpm:,}

# **💡 Try asking me:**
# - "What's the monthly revenue trend?"
# - "Which videos perform best?"
# - "How accurate are our estimates?"
# - "Show me revenue distribution"
# - "What recommendations do you have?"

# **I can also ask for clarification if your question needs more specifics!**
#                 """
#                 add_message("assistant", welcome_msg)
#                 st.success("✅ Ready to chat!")
#                 st.rerun()
    
#     # Clear chat option
#     if st.session_state.chat_messages:
#         if st.button("🔄 Clear Chat"):
#             st.session_state.chat_messages = []
#             st.rerun()
        
#         st.metric("Messages", len(st.session_state.chat_messages))

# # Main interface
# if st.session_state.data_loaded:
#     # Display current data summary
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Videos", len(st.session_state.label_videos))
#     with col2:
#         st.metric("Est. Revenue", f"₹{st.session_state.last_analysis_data['est_total']:,.0f}")
#     with col3:
#         accuracy = st.session_state.last_analysis_data['est_total'] / st.session_state.last_analysis_data['actual_total'] if st.session_state.last_analysis_data['actual_total'] else 0
#         st.metric("Accuracy", f"{accuracy:.1%}")
    
#     # Chat interface
#     display_chat_interface()
    
# else:
#     st.info("👆 Please upload your revenue CSV file in the sidebar to start chatting with the AI analyst!")
    
#     st.markdown("""
#     ## 🚀 How This Works:
    
#     1. **Upload Data**: Upload your revenue CSV file
#     2. **Select Label**: Choose which record label to analyze  
#     3. **Set RPM**: Define your revenue per million views
#     4. **Start Chatting**: Ask questions and get intelligent responses!
    
#     ## 🤖 AI Features:
    
#     - **Smart Analysis**: Detailed insights with real numbers
#     - **Clarification**: Asks questions when your query is unclear
#     - **Auto Visualizations**: Creates relevant charts automatically
#     - **Context Memory**: Remembers your conversation
#     - **Actionable Advice**: Provides specific recommendations
    
#     ## 💬 Example Questions:
    
#     - "What's driving our revenue growth?"
#     - "Compare this month vs last month"
#     - "Which videos should I promote more?"
#     - "How can we improve our RPM?"
#     """)








# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import re

# # Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# # Load NLP model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except:
#     st.error("Please install spacy English model: python -m spacy download en_core_web_sm")
#     nlp = None

# st.set_page_config(page_title="Interactive Data Analyzer", layout="wide")
# st.title("💬 Interactive YouTube Revenue Analyzer with Gemini AI")

# # Initialize session state
# if 'chat_messages' not in st.session_state:
#     st.session_state.chat_messages = []
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'label_videos' not in st.session_state:
#     st.session_state.label_videos = pd.DataFrame()
# if 'monthly_revenue' not in st.session_state:
#     st.session_state.monthly_revenue = pd.DataFrame()
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'pending_clarification' not in st.session_state:
#     st.session_state.pending_clarification = False
# if 'last_analysis_data' not in st.session_state:
#     st.session_state.last_analysis_data = {}

# def add_message(role, content, charts=None, data=None):
#     """Add message to chat history"""
#     message = {
#         'role': role,
#         'content': content,
#         'timestamp': datetime.now().strftime("%H:%M:%S"),
#         'charts': charts or [],
#         'data': data
#     }
#     st.session_state.chat_messages.append(message)

# def create_smart_visualizations(analysis_response, data_context):
#     """Automatically create relevant visualizations based on AI analysis"""
#     charts = []
    
#     # Extract chart instructions from AI response
#     chart_patterns = {
#         'monthly_trend': r'(?i)(monthly|month|trend|time|temporal|seasonal)',
#         'top_videos': r'(?i)(top|best|highest|ranking|performance)',
#         'revenue_distribution': r'(?i)(distribution|spread|variance|compare)',
#         'growth_analysis': r'(?i)(growth|increase|decrease|change)',
#         'correlation': r'(?i)(correlation|relationship|vs|versus)',
#         'breakdown': r'(?i)(breakdown|category|segment|group)',
#         'line_chart': r'(?i)(line chart|trend chart|time series)',
#         'bar_chart': r'(?i)(bar chart|bar graph|comparison)',
#         'scatter_plot': r'(?i)(scatter|correlation|relationship)',
#         'histogram': r'(?i)(histogram|distribution)',
#         'pie_chart': r'(?i)(pie chart|proportion|percentage)'
#     }
    
#     response_lower = analysis_response.lower()
    
#     # Monthly Revenue Trend
#     if re.search(chart_patterns['monthly_trend'], response_lower) and not st.session_state.monthly_revenue.empty:
#         fig = px.line(
#             st.session_state.monthly_revenue, 
#             x="Month", 
#             y="Estimated Revenue INR",
#             title="📈 Monthly Revenue Trend",
#             markers=True
#         )
#         fig.update_layout(height=400)
#         charts.append(('Monthly Revenue Trend', fig, st.session_state.monthly_revenue))
    
#     # Top Videos Performance
#     if re.search(chart_patterns['top_videos'], response_lower) and not st.session_state.label_videos.empty:
#         top_videos = st.session_state.label_videos.nlargest(10, 'Estimated Revenue INR')
#         fig = px.bar(
#             top_videos,
#             x='Estimated Revenue INR',
#             y='title',
#             orientation='h',
#             title="🏆 Top 10 Videos by Revenue"
#         )
#         fig.update_layout(height=500)
#         charts.append(('Top Videos Revenue', fig, top_videos[['title', 'view_count', 'Estimated Revenue INR']]))
    
#     # RPV Analysis
#     if 'rpv' in response_lower or 'revenue per view' in response_lower:
#         if not st.session_state.label_videos.empty:
#             st.session_state.label_videos['RPV'] = st.session_state.label_videos['Estimated Revenue INR'] / st.session_state.label_videos['view_count']
#             top_rpv = st.session_state.label_videos.nlargest(10, 'RPV')
#             fig = px.scatter(
#                 top_rpv,
#                 x='view_count',
#                 y='RPV',
#                 size='Estimated Revenue INR',
#                 hover_data=['title'],
#                 title="💰 Revenue Per View vs Views"
#             )
#             charts.append(('RPV Analysis', fig, top_rpv[['title', 'view_count', 'RPV', 'Estimated Revenue INR']]))
    
#     # Revenue Distribution
#     if re.search(chart_patterns['revenue_distribution'], response_lower) and not st.session_state.label_videos.empty:
#         fig = px.histogram(
#             st.session_state.label_videos,
#             x='Estimated Revenue INR',
#             nbins=20,
#             title="📊 Revenue Distribution"
#         )
#         charts.append(('Revenue Distribution', fig, None))
    
#     # Views vs Revenue Correlation
#     if re.search(chart_patterns['correlation'], response_lower) and not st.session_state.label_videos.empty:
#         fig = px.scatter(
#             st.session_state.label_videos,
#             x='view_count',
#             y='Estimated Revenue INR',
#             title="🔗 Views vs Revenue Correlation",
#             trendline="ols"
#         )
#         charts.append(('Views vs Revenue', fig, None))
    
#     return charts

# def extract_data_requirements(user_query):
#     """Extract what data the user is asking about"""
#     query_lower = user_query.lower()
#     requirements = {
#         'needs_monthly_data': any(word in query_lower for word in ['monthly', 'month', 'trend', 'seasonal', 'time']),
#         'needs_top_videos': any(word in query_lower for word in ['top', 'best', 'highest', 'ranking']),
#         'needs_comparison': any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']),
#         'needs_specific_metric': any(word in query_lower for word in ['rpm', 'rpv', 'cpm', 'revenue per']),
#         'needs_growth_data': any(word in query_lower for word in ['growth', 'increase', 'decrease', 'change']),
#         'unclear_intent': len([word for word in ['what', 'how', 'why', 'which', 'when'] if word in query_lower]) > 1
#     }
#     return requirements

# @st.cache_data
# def load_youtube_metadata():
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)
#         df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
#         known_labels = ["T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#                         "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"]
        
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#             for label in known_labels:
#                 if label.lower() in text:
#                     return label
#             return "Other"
        
#         df["Record Label"] = df.apply(detect_label, axis=1)
#         return df
#     except FileNotFoundError:
#         st.error("youtube_metadata.json file not found!")
#         return pd.DataFrame()

# def generate_intelligent_prompt(user_query, data_context, conversation_history):
#     """Generate intelligent prompt for Gemini that can ask for clarification"""
    
#     # Analyze user query
#     requirements = extract_data_requirements(user_query)
    
#     # Build context with data summary
#     data_summary = ""
#     if not st.session_state.label_videos.empty:
#         top_5_videos = st.session_state.label_videos.nlargest(5, 'Estimated Revenue INR')[['title', 'view_count', 'Estimated Revenue INR']]
#         data_summary = f"\nTop 5 Videos by Revenue:\n{top_5_videos.to_string(index=False)}"
    
#     context = f"""
# You are an expert Business Intelligence analyst with access to YouTube revenue data. You have advanced analytical capabilities and can:

# 1. **Analyze data and provide detailed insights with specific numbers**
# 2. **Ask for clarification when queries are ambiguous** 
# 3. **Suggest specific visualizations** (mention chart types like "line chart", "bar chart", "scatter plot", etc.)
# 4. **Provide actionable business recommendations**
# 5. **Generate data-driven insights and trends**

# **AVAILABLE DATA OVERVIEW:**
# - Total Videos: {len(st.session_state.label_videos)} videos
# - Revenue Period: {data_context.get('date_range', 'Multiple months')}
# - RPM (Revenue Per Million): ₹{data_context.get('rpm', 'Not specified'):,}
# - Total Estimated Revenue: ₹{data_context.get('est_total', 0):,.2f}
# - Actual Revenue: ₹{data_context.get('actual_total', 0):,.2f}
# - Record Label: {data_context.get('selected_label', 'Not specified')}
# {data_summary}

# **RECENT CONVERSATION:**
# {conversation_history}

# **USER QUERY:** "{user_query}"

# **DETECTED REQUIREMENTS:**
# {requirements}

# **YOUR RESPONSE GUIDELINES:**

# 1. **For Clear Queries**: Provide comprehensive analysis with:
#    - Specific numbers and percentages
#    - Key insights and trends
#    - Business implications
#    - Actionable recommendations
#    - Suggest relevant visualizations using phrases like "I recommend creating a [chart_type] showing [data]"

# 2. **For Ambiguous Queries**: Ask ONE specific clarifying question to better understand what the user wants

# 3. **Visualization Suggestions**: When recommending charts, use specific terms:
#    - "line chart" for trends over time
#    - "bar chart" for comparisons
#    - "scatter plot" for correlations
#    - "histogram" for distributions
#    - "pie chart" for proportions

# 4. **Be Conversational**: Use a friendly, helpful tone while being professional

# 5. **Focus on Value**: Always provide insights that can drive business decisions

# **CLARIFICATION SCENARIOS:**
# - If user says "analyze this" without specifying what aspect
# - If multiple time periods are available and user doesn't specify which
# - If query is too broad (ask which specific metrics to focus on)
# - If comparison is requested but criteria are unclear

# **RESPONSE FORMAT:**
# Use markdown formatting with clear sections:
# - ## 📊 Analysis Results
# - ## 💡 Key Insights  
# - ## 🎯 Recommendations
# - ## 📈 Suggested Visualizations

# Respond as a helpful analyst who provides value through both insights and clarifying questions when needed.
# """
    
#     return context

# def get_gemini_analysis(prompt, api_key):
#     """Get analysis from Gemini AI with clarification capabilities"""
#     try:
#         headers = {
#             "Content-Type": "application/json"
#         }
        
#         # Gemini API endpoint
#         url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
#         payload = {
#             "contents": [
#                 {
#                     "parts": [
#                         {
#                             "text": prompt
#                         }
#                     ]
#                 }
#             ],
#             "generationConfig": {
#                 "temperature": 0.7,
#                 "topK": 40,
#                 "topP": 0.95,
#                 "maxOutputTokens": 8192,
#             },
#             "safetySettings": [
#                 {
#                     "category": "HARM_CATEGORY_HARASSMENT",
#                     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_HATE_SPEECH",
#                     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#                 }
#             ]
#         }
        
#         response = requests.post(url, headers=headers, json=payload, timeout=120)
        
#         if response.status_code == 200:
#             result = response.json()
#             if "candidates" in result and len(result["candidates"]) > 0:
#                 content = result["candidates"][0]["content"]["parts"][0]["text"]
#                 return content, None
#             else:
#                 return "", "No response generated from Gemini"
#         else:
#             error_msg = f"Gemini API Error {response.status_code}"
#             if response.status_code == 429:
#                 error_msg += ": Rate limit exceeded. Please wait a moment and try again."
#             elif response.status_code == 400:
#                 error_msg += ": Invalid request. Please check your API key."
#             elif response.status_code == 403:
#                 error_msg += ": API key invalid or quota exceeded."
#             else:
#                 error_msg += f": {response.text}"
#             return "", error_msg
            
#     except requests.exceptions.Timeout:
#         return "", "Request timed out. Please try again."
#     except Exception as e:
#         return "", f"Error connecting to Gemini: {str(e)}"

# def process_user_input(user_input):
#     """Process user input and generate response with potential clarification"""
#     if not st.session_state.data_loaded:
#         add_message("assistant", "⚠️ Please upload a revenue CSV file first to begin analysis.")
#         return
    
#     # Add user message
#     add_message("user", user_input)
    
#     # Build conversation history
#     conversation_history = ""
#     for msg in st.session_state.chat_messages[-6:]:  # Last 6 messages for better context
#         role = "User" if msg['role'] == 'user' else "AI"
#         conversation_history += f"{role}: {msg['content'][:150]}...\n"
    
#     # Prepare data context
#     data_context = {
#         'rpm': st.session_state.last_analysis_data.get('rpm', 0),
#         'est_total': st.session_state.last_analysis_data.get('est_total', 0),
#         'actual_total': st.session_state.last_analysis_data.get('actual_total', 0),
#         'selected_label': st.session_state.last_analysis_data.get('selected_label', ''),
#         'date_range': 'Multiple months' if not st.session_state.monthly_revenue.empty else 'Single period'
#     }
    
#     # Generate prompt
#     prompt = generate_intelligent_prompt(user_input, data_context, conversation_history)
    
#     # Get AI response
#     with st.spinner("🤖 Gemini AI is analyzing your data..."):
#         response, error = get_gemini_analysis(prompt, GEMINI_API_KEY)
        
#         if error:
#             add_message("assistant", f"❌ Error: {error}")
#             return
        
#         # Check if AI is asking for clarification
#         clarification_indicators = ['?', 'clarify', 'specify', 'which', 'what do you mean', 'could you', 'can you specify', 'please provide']
#         is_asking_clarification = any(indicator in response.lower() for indicator in clarification_indicators)
        
#         if is_asking_clarification:
#             st.session_state.pending_clarification = True
#             add_message("assistant", response)
#         else:
#             st.session_state.pending_clarification = False
            
#             # Generate visualizations based on AI response
#             charts = create_smart_visualizations(response, data_context)
            
#             # Add response with charts
#             add_message("assistant", response, charts=charts)

# def display_chat_interface():
#     """Display the chat interface"""
#     st.subheader("💬 Chat with Gemini AI Analyst")
    
#     # Display chat messages
#     chat_container = st.container()
#     with chat_container:
#         for message in st.session_state.chat_messages:
#             if message['role'] == 'user':
#                 with st.chat_message("user"):
#                     st.write(f"**You** ({message['timestamp']})")
#                     st.write(message['content'])
#             else:
#                 with st.chat_message("assistant"):
#                     st.write(f"**Gemini AI** ({message['timestamp']})")
#                     st.markdown(message['content'])
                    
#                     # Display charts if any
#                     if message.get('charts'):
#                         st.write("**📊 Generated Visualizations:**")
#                         for chart_name, fig, data in message['charts']:
#                             st.plotly_chart(fig, use_container_width=True)
#                             if data is not None and not data.empty:
#                                 with st.expander(f"📋 Data for {chart_name}"):
#                                     st.dataframe(data)
    
#     # Chat input
#     user_input = st.chat_input("Ask me anything about your YouTube revenue data...")
    
#     if user_input and GEMINI_API_KEY:
#         process_user_input(user_input)
#         st.rerun()
#     elif user_input and not GEMINI_API_KEY:
#         add_message("assistant", "❌ Please set your GEMINI_API_KEY in the .env file")
#         st.rerun()

# # Sidebar for data loading
# with st.sidebar:
#     st.header("📊 Data Setup")
    
#     if GEMINI_API_KEY:
#         st.success("✅ Gemini AI Ready")
#         st.info("🆓 Free: 15 req/min, 1500 req/day")
#     else:
#         st.error("❌ Set GEMINI_API_KEY in .env")
#         st.markdown("""
#         **Get Gemini API Key:**
#         1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
#         2. Sign in with Google account
#         3. Click "Create API Key"
#         4. Copy key to .env file as:
#         ```
#         GEMINI_API_KEY=your_key_here
#         ```
#         """)
    
#     # File upload
#     uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])
    
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
        
#         # Load YouTube metadata
#         youtube_df = load_youtube_metadata()
        
#         if not youtube_df.empty:
#             selected_label = st.selectbox("🎙️ Record Label", sorted(youtube_df["Record Label"].unique()))
#             rpm = st.number_input("💸 RPM (₹)", min_value=500, value=125000)
            
#             if st.button("🚀 Load Data & Start Chat"):
#                 # Process data
#                 label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#                 label_videos = label_videos.dropna(subset=["view_count"])
#                 label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
                
#                 # Get actual revenue
#                 if "Store Name" in df.columns:
#                     yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#                     actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
#                 else:
#                     actual_total = 0
                
#                 # Process dates
#                 label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#                 label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#                 monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
                
#                 # Store in session state
#                 st.session_state.label_videos = label_videos
#                 st.session_state.monthly_revenue = monthly_revenue
#                 st.session_state.data_loaded = True
#                 st.session_state.last_analysis_data = {
#                     'rpm': rpm,
#                     'est_total': label_videos["Estimated Revenue INR"].sum(),
#                     'actual_total': actual_total,
#                     'selected_label': selected_label
#                 }
                
#                 # Clear previous chat and add welcome message
#                 st.session_state.chat_messages = []
#                 welcome_msg = f"""
# ## 🎉 Data Loaded Successfully!

# **Analysis Ready for:** {selected_label}
# - **Total Videos:** {len(label_videos):,}
# - **Estimated Revenue:** ₹{st.session_state.last_analysis_data['est_total']:,.2f}
# - **Actual Revenue:** ₹{actual_total:,.2f}
# - **RPM Used:** ₹{rpm:,}

# **💡 Try asking me:**
# - "What's the monthly revenue trend?"
# - "Which videos perform best and why?"
# - "How accurate are our revenue estimates?"
# - "Show me revenue distribution patterns"
# - "What actionable recommendations do you have?"
# - "Create visualizations for top performing content"

# **🤖 Powered by Gemini AI - I can provide detailed analysis and create relevant charts automatically!**
#                 """
#                 add_message("assistant", welcome_msg)
#                 st.success("✅ Ready to analyze with Gemini!")
#                 st.rerun()
    
#     # Clear chat option
#     if st.session_state.chat_messages:
#         if st.button("🔄 Clear Chat"):
#             st.session_state.chat_messages = []
#             st.rerun()
        
#         st.metric("Messages", len(st.session_state.chat_messages))

# # Main interface
# if st.session_state.data_loaded:
#     # Display current data summary
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Videos", len(st.session_state.label_videos))
#     with col2:
#         st.metric("Est. Revenue", f"₹{st.session_state.last_analysis_data['est_total']:,.0f}")
#     with col3:
#         accuracy = (st.session_state.last_analysis_data['est_total'] / st.session_state.last_analysis_data['actual_total'] 
#                    if st.session_state.last_analysis_data['actual_total'] else 0)
#         st.metric("Accuracy", f"{accuracy:.1%}")
#     with col4:
#         avg_revenue = st.session_state.last_analysis_data['est_total'] / len(st.session_state.label_videos) if len(st.session_state.label_videos) > 0 else 0
#         st.metric("Avg/Video", f"₹{avg_revenue:,.0f}")
    
#     # Chat interface
#     display_chat_interface()
    
# else:
#     st.info("👆 Please upload your revenue CSV file in the sidebar to start chatting with Gemini AI!")
    
#     st.markdown("""
#     ## 🚀 Enhanced with Gemini AI:
    
#     1. **Upload Data**: Upload your revenue CSV file
#     2. **Select Label**: Choose which record label to analyze  
#     3. **Set RPM**: Define your revenue per million views
#     4. **Start Chatting**: Ask questions and get intelligent responses from Gemini!
    
#     ## 🤖 Gemini AI Features:
    
#     - **📊 Advanced Analysis**: Deep insights with specific numbers and trends
#     - **❓ Smart Clarification**: Asks targeted questions when queries are unclear
#     - **📈 Auto Visualizations**: Creates relevant charts based on your questions
#     - **🧠 Context Memory**: Remembers conversation flow for better responses
#     - **🎯 Actionable Advice**: Provides specific, data-driven recommendations
#     - **⚡ Fast & Free**: Google's powerful AI with generous free tier
    
#     ## 💬 Sample Questions for Gemini:
    
#     - "What trends do you see in our monthly revenue data?"
#     - "Which video characteristics correlate with higher revenue?"
#     - "Create a comparative analysis of our top vs bottom performing videos"
#     - "What recommendations do you have to optimize our YouTube strategy?"
#     - "Show me revenue patterns and suggest improvement areas"
    
#     ## 🔑 Setup Instructions:
    
#     1. Get your **free** Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
#     2. Add it to your `.env` file as: `GEMINI_API_KEY=your_key_here`
#     3. Upload your data and start analyzing!
#     """)





# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# import re

# # Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# # Load NLP model
# try:
#     nlp = spacy.load("en_core_web_sm")
# except:
#     st.error("Please install spacy English model: python -m spacy download en_core_web_sm")
#     nlp = None

# st.set_page_config(page_title="Interactive Data Analyzer", layout="wide")
# st.title("💬 Interactive YouTube Revenue Analyzer with Gemini AI")

# # Initialize session state
# if 'chat_messages' not in st.session_state:
#     st.session_state.chat_messages = []
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'label_videos' not in st.session_state:
#     st.session_state.label_videos = pd.DataFrame()
# if 'monthly_revenue' not in st.session_state:
#     st.session_state.monthly_revenue = pd.DataFrame()
# if 'analysis_context' not in st.session_state:
#     st.session_state.analysis_context = {}
# if 'pending_clarification' not in st.session_state:
#     st.session_state.pending_clarification = False
# if 'last_analysis_data' not in st.session_state:
#     st.session_state.last_analysis_data = {}

# def add_message(role, content, charts=None, data=None, tables=None):
#     """Add message to chat history"""
#     message = {
#         'role': role,
#         'content': content,
#         'timestamp': datetime.now().strftime("%H:%M:%S"),
#         'charts': charts or [],
#         'tables': tables or [],
#         'data': data
#     }
#     st.session_state.chat_messages.append(message)

# def create_smart_visualizations(analysis_response, data_context):
#     """Automatically create relevant visualizations based on AI analysis"""
#     charts = []
#     tables = []
    
#     # Enhanced chart detection patterns
#     chart_patterns = {
#         'monthly_trend': r'(?i)(monthly|month|trend|time|temporal|seasonal|over time|timeline)',
#         'top_videos': r'(?i)(top|best|highest|ranking|performance|leading|most.*revenue)',
#         'revenue_distribution': r'(?i)(distribution|spread|variance|histogram|frequency)',
#         'growth_analysis': r'(?i)(growth|increase|decrease|change|compare.*month)',
#         'correlation': r'(?i)(correlation|relationship|vs|versus|scatter|association)',
#         'breakdown': r'(?i)(breakdown|category|segment|group|proportion)',
#         'line_chart': r'(?i)(line chart|trend chart|time series|plot.*trend)',
#         'bar_chart': r'(?i)(bar chart|bar graph|comparison|compare.*performance)',
#         'scatter_plot': r'(?i)(scatter|plot.*correlation|relationship.*plot)',
#         'histogram': r'(?i)(histogram|distribution|frequency)',
#         'pie_chart': r'(?i)(pie chart|proportion|percentage|share)',
#         'table_request': r'(?i)(table|tabular|list|show.*data|details|breakdown.*data)',
#         'summary_stats': r'(?i)(summary|statistics|stats|describe|overview)',
#         'create_chart': r'(?i)(create|generate|show|plot|draw|make.*chart|visualiz)',
#         'rpm_analysis': r'(?i)(rpm|revenue per.*million|efficiency)',
#         'performance_metrics': r'(?i)(performance|metrics|kpi|key.*indicator)',
#         'comparative': r'(?i)(compare|comparison|versus|vs|against)',
#         'descriptive': r'(?i)(describe|explain|detail|insight|analysis)'
#     }
    
#     response_lower = analysis_response.lower()
    
#     # 1. Monthly Revenue Trend (Always show if mentioned)
#     if (re.search(chart_patterns['monthly_trend'], response_lower) or 
#         re.search(chart_patterns['line_chart'], response_lower) or
#         re.search(chart_patterns['create_chart'], response_lower)) and not st.session_state.monthly_revenue.empty:
        
#         fig = px.line(
#             st.session_state.monthly_revenue, 
#             x="Month", 
#             y="Estimated Revenue INR",
#             title="📈 Monthly Revenue Trend Analysis",
#             markers=True,
#             text="Estimated Revenue INR"
#         )
#         fig.update_traces(texttemplate='₹%{text:,.0f}', textposition="top center")
#         fig.update_layout(height=450, showlegend=False)
#         fig.update_xaxis(title="Month")
#         fig.update_yaxis(title="Revenue (₹)")
#         charts.append(('Monthly Revenue Trend', fig, st.session_state.monthly_revenue))
    
#     # 2. Top Videos Performance (Enhanced)
#     if (re.search(chart_patterns['top_videos'], response_lower) or 
#         re.search(chart_patterns['bar_chart'], response_lower) or
#         re.search(chart_patterns['ranking'], response_lower) or
#         re.search(chart_patterns['create_chart'], response_lower)) and not st.session_state.label_videos.empty:
        
#         top_videos = st.session_state.label_videos.nlargest(15, 'Estimated Revenue INR')
#         # Truncate long titles
#         top_videos['short_title'] = top_videos['title'].str[:50] + '...'
        
#         fig = px.bar(
#             top_videos,
#             x='Estimated Revenue INR',
#             y='short_title',
#             orientation='h',
#             title="🏆 Top 15 Videos by Revenue Performance",
#             text='Estimated Revenue INR'
#         )
#         fig.update_traces(texttemplate='₹%{text:,.0f}', textposition="outside")
#         fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
#         fig.update_xaxis(title="Revenue (₹)")
#         fig.update_yaxis(title="Video Title")
#         charts.append(('Top Videos Revenue', fig, top_videos[['title', 'view_count', 'Estimated Revenue INR']]))
    
#     # 3. RPV Analysis (Revenue Per View)
#     if ('rpv' in response_lower or 'revenue per view' in response_lower or 
#         re.search(chart_patterns['rpm_analysis'], response_lower) or
#         re.search(chart_patterns['efficiency'], response_lower)):
        
#         if not st.session_state.label_videos.empty:
#             df_rpv = st.session_state.label_videos.copy()
#             df_rpv['RPV'] = df_rpv['Estimated Revenue INR'] / df_rpv['view_count']
#             top_rpv = df_rpv.nlargest(20, 'RPV')
            
#             fig = px.scatter(
#                 top_rpv,
#                 x='view_count',
#                 y='RPV',
#                 size='Estimated Revenue INR',
#                 hover_data=['title'],
#                 title="💰 Revenue Efficiency: RPV vs Views Analysis",
#                 color='Estimated Revenue INR',
#                 size_max=60
#             )
#             fig.update_layout(height=500)
#             fig.update_xaxis(title="View Count")
#             fig.update_yaxis(title="Revenue Per View (₹)")
#             charts.append(('RPV Analysis', fig, top_rpv[['title', 'view_count', 'RPV', 'Estimated Revenue INR']]))
    
#     # 4. Revenue Distribution
#     if (re.search(chart_patterns['revenue_distribution'], response_lower) or 
#         re.search(chart_patterns['histogram'], response_lower)):
        
#         if not st.session_state.label_videos.empty:
#             fig = px.histogram(
#                 st.session_state.label_videos,
#                 x='Estimated Revenue INR',
#                 nbins=25,
#                 title="📊 Revenue Distribution Across All Videos",
#                 marginal="box"
#             )
#             fig.update_layout(height=450)
#             fig.update_xaxis(title="Revenue (₹)")
#             fig.update_yaxis(title="Number of Videos")
#             charts.append(('Revenue Distribution', fig, None))
    
#     # 5. Views vs Revenue Correlation
#     if (re.search(chart_patterns['correlation'], response_lower) or 
#         re.search(chart_patterns['scatter_plot'], response_lower) or
#         'views.*revenue' in response_lower):
        
#         if not st.session_state.label_videos.empty:
#             fig = px.scatter(
#                 st.session_state.label_videos,
#                 x='view_count',
#                 y='Estimated Revenue INR',
#                 title="🔗 Views vs Revenue Correlation Analysis",
#                 trendline="ols",
#                 hover_data=['title']
#             )
#             fig.update_layout(height=450)
#             fig.update_xaxis(title="View Count")
#             fig.update_yaxis(title="Revenue (₹)")
#             charts.append(('Views vs Revenue Correlation', fig, None))
    
#     # 6. Performance Comparison Chart
#     if re.search(chart_patterns['comparative'], response_lower):
#         if not st.session_state.label_videos.empty:
#             # Create performance tiers
#             df_perf = st.session_state.label_videos.copy()
#             df_perf['Performance_Tier'] = pd.cut(df_perf['Estimated Revenue INR'], 
#                                                bins=4, 
#                                                labels=['Low', 'Medium', 'High', 'Top'])
#             tier_summary = df_perf.groupby('Performance_Tier').agg({
#                 'Estimated Revenue INR': ['count', 'sum', 'mean']
#             }).round(2)
            
#             fig = px.pie(
#                 df_perf.groupby('Performance_Tier').size().reset_index(name='count'),
#                 values='count',
#                 names='Performance_Tier',
#                 title="🎯 Video Performance Distribution"
#             )
#             charts.append(('Performance Comparison', fig, tier_summary))
    
#     # 7. Create Summary Tables when requested
#     if (re.search(chart_patterns['table_request'], response_lower) or 
#         re.search(chart_patterns['summary_stats'], response_lower) or
#         re.search(chart_patterns['descriptive'], response_lower)):
        
#         if not st.session_state.label_videos.empty:
#             # Summary Statistics Table
#             summary_stats = {
#                 'Metric': ['Total Videos', 'Total Revenue', 'Average Revenue/Video', 
#                           'Highest Revenue Video', 'Total Views', 'Average Views/Video'],
#                 'Value': [
#                     f"{len(st.session_state.label_videos):,}",
#                     f"₹{st.session_state.label_videos['Estimated Revenue INR'].sum():,.2f}",
#                     f"₹{st.session_state.label_videos['Estimated Revenue INR'].mean():,.2f}",
#                     f"₹{st.session_state.label_videos['Estimated Revenue INR'].max():,.2f}",
#                     f"{st.session_state.label_videos['view_count'].sum():,}",
#                     f"{st.session_state.label_videos['view_count'].mean():,.0f}"
#                 ]
#             }
#             tables.append(('Summary Statistics', pd.DataFrame(summary_stats)))
            
#             # Top 10 Videos Table
#             top_10_table = st.session_state.label_videos.nlargest(10, 'Estimated Revenue INR')[
#                 ['title', 'view_count', 'Estimated Revenue INR']
#             ].copy()
#             top_10_table['Estimated Revenue INR'] = top_10_table['Estimated Revenue INR'].apply(lambda x: f"₹{x:,.2f}")
#             top_10_table['view_count'] = top_10_table['view_count'].apply(lambda x: f"{x:,}")
#             top_10_table.columns = ['Video Title', 'Views', 'Revenue']
#             tables.append(('Top 10 Videos', top_10_table))
    
#     return charts, tables

# def extract_data_requirements(user_query):
#     """Extract what data the user is asking about"""
#     query_lower = user_query.lower()
#     requirements = {
#         'needs_monthly_data': any(word in query_lower for word in ['monthly', 'month', 'trend', 'seasonal', 'time']),
#         'needs_top_videos': any(word in query_lower for word in ['top', 'best', 'highest', 'ranking']),
#         'needs_comparison': any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']),
#         'needs_specific_metric': any(word in query_lower for word in ['rpm', 'rpv', 'cpm', 'revenue per']),
#         'needs_growth_data': any(word in query_lower for word in ['growth', 'increase', 'decrease', 'change']),
#         'unclear_intent': len([word for word in ['what', 'how', 'why', 'which', 'when'] if word in query_lower]) > 1
#     }
#     return requirements

# @st.cache_data
# def load_youtube_metadata():
#     try:
#         with open("youtube_metadata.json", "r", encoding="utf-8") as f:
#             data = json.load(f)
#         df = pd.DataFrame(data["videos"]) if "videos" in data else pd.DataFrame(data)
#         known_labels = ["T-Series", "Sony Music", "Zee Music", "Tips", "Saregama", "Aditya Music",
#                         "Lahari", "Speed Records", "YRF", "Venus", "SVF", "White Hill", "TIPS Official"]
        
#         def detect_label(row):
#             text = f"{row.get('title', '')} {row.get('description', '')}".lower()
#             for label in known_labels:
#                 if label.lower() in text:
#                     return label
#             return "Other"
        
#         df["Record Label"] = df.apply(detect_label, axis=1)
#         return df
#     except FileNotFoundError:
#         st.error("youtube_metadata.json file not found!")
#         return pd.DataFrame()

# def generate_intelligent_prompt(user_query, data_context, conversation_history):
#     """Generate intelligent prompt for Gemini that can ask for clarification"""
    
#     # Analyze user query
#     requirements = extract_data_requirements(user_query)
    
#     # Build context with data summary
#     data_summary = ""
#     if not st.session_state.label_videos.empty:
#         top_5_videos = st.session_state.label_videos.nlargest(5, 'Estimated Revenue INR')[['title', 'view_count', 'Estimated Revenue INR']]
#         data_summary = f"\nTop 5 Videos by Revenue:\n{top_5_videos.to_string(index=False)}"
    
#     context = f"""
# You are an expert Business Intelligence analyst with access to YouTube revenue data. You have advanced analytical capabilities and can:

# 1. **Analyze data and provide detailed insights with specific numbers**
# 2. **Ask for clarification when queries are ambiguous** 
# 3. **Suggest specific visualizations** (mention chart types like "line chart", "bar chart", "scatter plot", etc.)
# 4. **Provide actionable business recommendations**
# 5. **Generate data-driven insights and trends**

# **AVAILABLE DATA OVERVIEW:**
# - Total Videos: {len(st.session_state.label_videos)} videos
# - Revenue Period: {data_context.get('date_range', 'Multiple months')}
# - RPM (Revenue Per Million): ₹{data_context.get('rpm', 'Not specified'):,}
# - Total Estimated Revenue: ₹{data_context.get('est_total', 0):,.2f}
# - Actual Revenue: ₹{data_context.get('actual_total', 0):,.2f}
# - Record Label: {data_context.get('selected_label', 'Not specified')}
# {data_summary}

# **RECENT CONVERSATION:**
# {conversation_history}

# **USER QUERY:** "{user_query}"

# **DETECTED REQUIREMENTS:**
# {requirements}

# **YOUR RESPONSE GUIDELINES:**

# 1. **For Clear Queries**: Provide comprehensive analysis with:
#    - Specific numbers and percentages
#    - Key insights and trends
#    - Business implications
#    - Actionable recommendations
#    - Suggest relevant visualizations using phrases like "I recommend creating a [chart_type] showing [data]"

# 2. **For Ambiguous Queries**: Ask ONE specific clarifying question to better understand what the user wants

# 3. **Visualization Suggestions**: When recommending charts, use specific terms:
#    - "line chart" for trends over time
#    - "bar chart" for comparisons
#    - "scatter plot" for correlations
#    - "histogram" for distributions
#    - "pie chart" for proportions

# 4. **Be Conversational**: Use a friendly, helpful tone while being professional

# 5. **Focus on Value**: Always provide insights that can drive business decisions

# **CLARIFICATION SCENARIOS:**
# - If user says "analyze this" without specifying what aspect
# - If multiple time periods are available and user doesn't specify which
# - If query is too broad (ask which specific metrics to focus on)
# - If comparison is requested but criteria are unclear

# **RESPONSE FORMAT:**
# Use markdown formatting with clear sections:
# - ## 📊 Analysis Results
# - ## 💡 Key Insights  
# - ## 🎯 Recommendations
# - ## 📈 Suggested Visualizations

# Respond as a helpful analyst who provides value through both insights and clarifying questions when needed.
# """
    
#     return context

# def get_gemini_analysis(prompt, api_key):
#     """Get analysis from Gemini AI with clarification capabilities"""
#     try:
#         headers = {
#             "Content-Type": "application/json"
#         }
        
#         # Gemini API endpoint
#         url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
#         payload = {
#             "contents": [
#                 {
#                     "parts": [
#                         {
#                             "text": prompt
#                         }
#                     ]
#                 }
#             ],
#             "generationConfig": {
#                 "temperature": 0.7,
#                 "topK": 40,
#                 "topP": 0.95,
#                 "maxOutputTokens": 8192,
#             },
#             "safetySettings": [
#                 {
#                     "category": "HARM_CATEGORY_HARASSMENT",
#                     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_HATE_SPEECH",
#                     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#                 }
#             ]
#         }
        
#         response = requests.post(url, headers=headers, json=payload, timeout=120)
        
#         if response.status_code == 200:
#             result = response.json()
#             if "candidates" in result and len(result["candidates"]) > 0:
#                 content = result["candidates"][0]["content"]["parts"][0]["text"]
#                 return content, None
#             else:
#                 return "", "No response generated from Gemini"
#         else:
#             error_msg = f"Gemini API Error {response.status_code}"
#             if response.status_code == 429:
#                 error_msg += ": Rate limit exceeded. Please wait a moment and try again."
#             elif response.status_code == 400:
#                 error_msg += ": Invalid request. Please check your API key."
#             elif response.status_code == 403:
#                 error_msg += ": API key invalid or quota exceeded."
#             else:
#                 error_msg += f": {response.text}"
#             return "", error_msg
            
#     except requests.exceptions.Timeout:
#         return "", "Request timed out. Please try again."
#     except Exception as e:
#         return "", f"Error connecting to Gemini: {str(e)}"

# def process_user_input(user_input):
#     """Process user input and generate response with potential clarification"""
#     if not st.session_state.data_loaded:
#         add_message("assistant", "⚠️ Please upload a revenue CSV file first to begin analysis.")
#         return
    
#     # Add user message
#     add_message("user", user_input)
    
#     # Build conversation history
#     conversation_history = ""
#     for msg in st.session_state.chat_messages[-6:]:  # Last 6 messages for better context
#         role = "User" if msg['role'] == 'user' else "AI"
#         conversation_history += f"{role}: {msg['content'][:150]}...\n"
    
#     # Prepare data context
#     data_context = {
#         'rpm': st.session_state.last_analysis_data.get('rpm', 0),
#         'est_total': st.session_state.last_analysis_data.get('est_total', 0),
#         'actual_total': st.session_state.last_analysis_data.get('actual_total', 0),
#         'selected_label': st.session_state.last_analysis_data.get('selected_label', ''),
#         'date_range': 'Multiple months' if not st.session_state.monthly_revenue.empty else 'Single period'
#     }
    
#     # Generate prompt
#     prompt = generate_intelligent_prompt(user_input, data_context, conversation_history)
    
#     # Get AI response
#     with st.spinner("🤖 Gemini AI is analyzing your data..."):
#         response, error = get_gemini_analysis(prompt, GEMINI_API_KEY)
        
#         if error:
#             add_message("assistant", f"❌ Error: {error}")
#             return
        
#         # Check if AI is asking for clarification
#         clarification_indicators = ['?', 'clarify', 'specify', 'which', 'what do you mean', 'could you', 'can you specify', 'please provide']
#         is_asking_clarification = any(indicator in response.lower() for indicator in clarification_indicators)
        
#         if is_asking_clarification:
#             st.session_state.pending_clarification = True
#             add_message("assistant", response)
#         else:
#             st.session_state.pending_clarification = False
            
#             # Generate visualizations and tables based on AI response
#             charts, tables = create_smart_visualizations(response, data_context)
            
#             # Add response with charts and tables
#             add_message("assistant", response, charts=charts, tables=tables)

# def display_chat_interface():
#     """Display the chat interface"""
#     st.subheader("💬 Chat with Gemini AI Analyst")
    
#     # Display chat messages
#     chat_container = st.container()
#     with chat_container:
#         for message in st.session_state.chat_messages:
#             if message['role'] == 'user':
#                 with st.chat_message("user"):
#                     st.write(f"**You** ({message['timestamp']})")
#                     st.write(message['content'])
#             else:
#                 with st.chat_message("assistant"):
#                     st.write(f"**Gemini AI** ({message['timestamp']})")
#                     st.markdown(message['content'])
                    
#                     # Display tables if any
#                     if message.get('tables'):
#                         st.write("**📋 Data Tables:**")
#                         for table_name, table_data in message['tables']:
#                             st.write(f"**{table_name}:**")
#                             st.dataframe(table_data, use_container_width=True)
#                             st.write("---")
                    
#                     # Display charts if any
#                     if message.get('charts'):
#                         st.write("**📊 Generated Visualizations:**")
#                         for chart_name, fig, data in message['charts']:
#                             st.plotly_chart(fig, use_container_width=True)
#                             if data is not None and not data.empty:
#                                 with st.expander(f"📋 Raw Data for {chart_name}"):
#                                     st.dataframe(data, use_container_width=True)
    
#     # Chat input
#     user_input = st.chat_input("Ask me anything about your YouTube revenue data...")
    
#     if user_input and GEMINI_API_KEY:
#         process_user_input(user_input)
#         st.rerun()
#     elif user_input and not GEMINI_API_KEY:
#         add_message("assistant", "❌ Please set your GEMINI_API_KEY in the .env file")
#         st.rerun()

# # Sidebar for data loading
# with st.sidebar:
#     st.header("📊 Data Setup")
    
#     if GEMINI_API_KEY:
#         st.success("✅ Gemini AI Ready")
#         st.info("🆓 Free: 15 req/min, 1500 req/day")
#     else:
#         st.error("❌ Set GEMINI_API_KEY in .env")
#         st.markdown("""
#         **Get Gemini API Key:**
#         1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
#         2. Sign in with Google account
#         3. Click "Create API Key"
#         4. Copy key to .env file as:
#         ```
#         GEMINI_API_KEY=your_key_here
#         ```
#         """)
    
#     # File upload
#     uploaded_file = st.file_uploader("📁 Upload Revenue CSV", type=["csv"])
    
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
        
#         # Load YouTube metadata
#         youtube_df = load_youtube_metadata()
        
#         if not youtube_df.empty:
#             selected_label = st.selectbox("🎙️ Record Label", sorted(youtube_df["Record Label"].unique()))
#             rpm = st.number_input("💸 RPM (₹)", min_value=500, value=125000)
            
#             if st.button("🚀 Load Data & Start Chat"):
#                 # Process data
#                 label_videos = youtube_df[youtube_df["Record Label"] == selected_label].copy()
#                 label_videos = label_videos.dropna(subset=["view_count"])
#                 label_videos["Estimated Revenue INR"] = label_videos["view_count"] / 1_000_000 * rpm
                
#                 # Get actual revenue
#                 if "Store Name" in df.columns:
#                     yt_row = df[df["Store Name"].str.lower().str.contains("youtube", na=False)]
#                     actual_total = yt_row["Annual Revenue in INR"].values[0] if not yt_row.empty else 0
#                 else:
#                     actual_total = 0
                
#                 # Process dates
#                 label_videos["published_at"] = pd.to_datetime(label_videos["published_at"], errors="coerce")
#                 label_videos["Month"] = label_videos["published_at"].dt.to_period("M").astype(str)
#                 monthly_revenue = label_videos.groupby("Month")["Estimated Revenue INR"].sum().reset_index()
                
#                 # Store in session state
#                 st.session_state.label_videos = label_videos
#                 st.session_state.monthly_revenue = monthly_revenue
#                 st.session_state.data_loaded = True
#                 st.session_state.last_analysis_data = {
#                     'rpm': rpm,
#                     'est_total': label_videos["Estimated Revenue INR"].sum(),
#                     'actual_total': actual_total,
#                     'selected_label': selected_label
#                 }
                
#                 # Clear previous chat and add welcome message
#                 st.session_state.chat_messages = []
#                 welcome_msg = f"""
# ## 🎉 Data Loaded Successfully!

# **Analysis Ready for:** {selected_label}
# - **Total Videos:** {len(label_videos):,}
# - **Estimated Revenue:** ₹{st.session_state.last_analysis_data['est_total']:,.2f}
# - **Actual Revenue:** ₹{actual_total:,.2f}
# - **RPM Used:** ₹{rpm:,}

# **💡 Try asking me:**
# - "What's the monthly revenue trend?"
# - "Which videos perform best and why?"
# - "How accurate are our revenue estimates?"
# - "Show me revenue distribution patterns"
# - "What actionable recommendations do you have?"
# - "Create visualizations for top performing content"

# **🤖 Powered by Gemini AI - I can provide detailed analysis and create relevant charts automatically!**
#                 """
#                 add_message("assistant", welcome_msg)
#                 st.success("✅ Ready to analyze with Gemini!")
#                 st.rerun()
    
#     # Clear chat option
#     if st.session_state.chat_messages:
#         if st.button("🔄 Clear Chat"):
#             st.session_state.chat_messages = []
#             st.rerun()
        
#         st.metric("Messages", len(st.session_state.chat_messages))

# # Main interface
# if st.session_state.data_loaded:
#     # Display current data summary
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Videos", len(st.session_state.label_videos))
#     with col2:
#         st.metric("Est. Revenue", f"₹{st.session_state.last_analysis_data['est_total']:,.0f}")
#     with col3:
#         accuracy = (st.session_state.last_analysis_data['est_total'] / st.session_state.last_analysis_data['actual_total'] 
#                    if st.session_state.last_analysis_data['actual_total'] else 0)
#         st.metric("Accuracy", f"{accuracy:.1%}")
#     with col4:
#         avg_revenue = st.session_state.last_analysis_data['est_total'] / len(st.session_state.label_videos) if len(st.session_state.label_videos) > 0 else 0
#         st.metric("Avg/Video", f"₹{avg_revenue:,.0f}")
    
#     # Chat interface
#     display_chat_interface()
    
# else:
#     st.info("👆 Please upload your revenue CSV file in the sidebar to start chatting with Gemini AI!")
    
#     st.markdown("""
#     ## 🚀 Enhanced with Gemini AI:
    
#     1. **Upload Data**: Upload your revenue CSV file
#     2. **Select Label**: Choose which record label to analyze  
#     3. **Set RPM**: Define your revenue per million views
#     4. **Start Chatting**: Ask questions and get intelligent responses from Gemini!
    
#     ## 🤖 Gemini AI Features:
    
#     - **📊 Advanced Analysis**: Deep insights with specific numbers and trends
#     - **❓ Smart Clarification**: Asks targeted questions when queries are unclear
#     - **📈 Auto Visualizations**: Creates relevant charts based on your questions
#     - **🧠 Context Memory**: Remembers conversation flow for better responses
#     - **🎯 Actionable Advice**: Provides specific, data-driven recommendations
#     - **⚡ Fast & Free**: Google's powerful AI with generous free tier
    
#     ## 💬 Enhanced Sample Questions:
    
#     - **"Show me the monthly revenue trends"** → Gets line chart + analysis
#     - **"Which are my top performing videos?"** → Gets bar chart + detailed table
#     - **"Analyze the revenue distribution"** → Gets histogram + statistics
#     - **"What's the correlation between views and revenue?"** → Gets scatter plot + insights
#     - **"Create a summary table of all key metrics"** → Gets comprehensive data tables
#     - **"Compare high vs low performing videos"** → Gets comparison charts + breakdown
#     - **"Show me RPV analysis for efficiency"** → Gets RPV scatter plot + recommendations
    
#     ## 🔑 Setup Instructions:
    
#     1. Get your **free** Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
#     2. Add it to your `.env` file as: `GEMINI_API_KEY=your_key_here`
#     3. Upload your data and start analyzing!
#     """)





# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime
# import hashlib
# from lida import Manager, TextGenerationConfig, llm
# from lida.components import TextGenerationConfig
# import base64
# from PIL import Image
# import io
# import openai

# # MUST BE FIRST - Configure Streamlit page
# st.set_page_config(
#     page_title="Smart Data Analyzer Pro with LIDA", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Load environment variables
# load_dotenv()

# # Custom OpenRouter LLM class for LIDA
# class OpenRouterLLM:
#     def __init__(self, api_key, model="mistralai/mistral-7b-instruct:free"):
#         self.api_key = api_key
#         self.model = model
#         self.base_url = "https://openrouter.ai/api/v1"
        
#     def generate(self, messages, **kwargs):
#         """Generate text using OpenRouter API"""
#         try:
#             headers = {
#                 "Authorization": f"Bearer {self.api_key}",
#                 "HTTP-Referer": "http://localhost:8501",
#                 "X-Title": "Smart Data Analyzer Pro",
#                 "Content-Type": "application/json"
#             }
            
#             # Convert messages to OpenRouter format
#             formatted_messages = []
#             for msg in messages:
#                 if isinstance(msg, dict):
#                     formatted_messages.append(msg)
#                 else:
#                     formatted_messages.append({"role": "user", "content": str(msg)})
            
#             payload = {
#                 "model": self.model,
#                 "messages": formatted_messages,
#                 "max_tokens": kwargs.get("max_tokens", 1000),
#                 "temperature": kwargs.get("temperature", 0.7)
#             }
            
#             response = requests.post(
#                 f"{self.base_url}/chat/completions",
#                 headers=headers,
#                 json=payload,
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 return result['choices'][0]['message']['content']
#             else:
#                 st.error(f"OpenRouter API Error: {response.status_code}")
#                 return "Error generating response"
                
#         except Exception as e:
#             st.error(f"Error in OpenRouter LLM: {str(e)}")
#             return "Error generating response"

# # Initialize LIDA with OpenRouter
# @st.cache_resource
# def initialize_lida():
#     """Initialize LIDA with OpenRouter LLM"""
#     try:
#         openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
#         if not openrouter_api_key:
#             st.warning("OpenRouter API key not found. LIDA will use fallback mode.")
#             return None
            
#         # Create custom OpenRouter LLM instance
#         openrouter_llm = OpenRouterLLM(openrouter_api_key)
        
#         # Create LIDA manager with custom LLM
#         # We'll use a wrapper to make OpenRouter compatible with LIDA
#         class LIDAOpenRouterWrapper:
#             def __init__(self, openrouter_llm):
#                 self.llm = openrouter_llm
                
#             def generate(self, messages, config=None):
#                 return self.llm.generate(messages)
                
#         # Initialize LIDA manager
#         lida_manager = Manager()
        
#         # Replace LIDA's default LLM with our OpenRouter wrapper
#         lida_manager.llm = LIDAOpenRouterWrapper(openrouter_llm)
        
#         st.success("✅ LIDA initialized successfully with OpenRouter!")
#         return lida_manager
        
#     except Exception as e:
#         st.warning(f"LIDA initialization failed: {str(e)}. Using fallback visualization.")
#         return None

# # Initialize components
# try:
#     nlp = spacy.load("en_core_web_sm")
# except IOError:
#     st.error("spaCy model not found. Please install it with: python -m spacy download en_core_web_sm")
#     nlp = None

# # Initialize LIDA
# lida_manager = initialize_lida()

# def query_openrouter_llm(prompt, context=""):
#     """Query OpenRouter LLM for insights"""
#     try:
#         api_key = os.getenv("OPENROUTER_API_KEY")
#         if not api_key:
#             return "OpenRouter API key not configured."
        
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "HTTP-Referer": "http://localhost:8501",
#             "X-Title": "Smart Data Analyzer Pro",
#             "Content-Type": "application/json"
#         }
        
#         full_prompt = f"""
#         Context: {context}
        
#         Question: {prompt}
        
#         Please provide a clear, concise analysis based on the data context provided.
#         """
        
#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "messages": [{"role": "user", "content": full_prompt}],
#             "max_tokens": 500,
#             "temperature": 0.7
#         }
        
#         response = requests.post(
#             "https://openrouter.ai/api/v1/chat/completions",
#             headers=headers,
#             json=payload,
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             result = response.json()
#             return result['choices'][0]['message']['content']
#         else:
#             return f"Error: {response.status_code} - {response.text}"
            
#     except Exception as e:
#         return f"Error querying LLM: {str(e)}"

# def generate_lida_visualization(df, query):
#     """Generate visualization using LIDA with OpenRouter"""
#     try:
#         if lida_manager is None:
#             return None, "LIDA not available"
        
#         # Convert DataFrame to LIDA format
#         summary = lida_manager.summarize(df)
        
#         # Generate visualization goals based on query
#         goals = lida_manager.goals(summary, n=3, persona="data analyst")
        
#         # Find the most relevant goal based on query
#         relevant_goal = None
#         for goal in goals:
#             if any(word.lower() in goal.get('question', '').lower() for word in query.lower().split()):
#                 relevant_goal = goal
#                 break
        
#         if not relevant_goal and goals:
#             relevant_goal = goals[0]
        
#         if relevant_goal:
#             # Generate visualization code
#             charts = lida_manager.visualize(
#                 summary=summary,
#                 goal=relevant_goal,
#                 textgen_config=TextGenerationConfig(n=1, temperature=0.2)
#             )
            
#             if charts:
#                 return charts[0], None
        
#         return None, "No suitable visualization found"
        
#     except Exception as e:
#         return None, f"LIDA visualization error: {str(e)}"

# def create_smart_visualization(df, query):
#     """Create intelligent visualizations based on query and data"""
#     query_lower = query.lower()
    
#     # Analyze data types
#     numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
#     # Try LIDA first
#     if lida_manager:
#         lida_chart, lida_error = generate_lida_visualization(df, query)
#         if lida_chart and not lida_error:
#             try:
#                 # Execute LIDA-generated code
#                 exec(lida_chart.code)
#                 return None  # LIDA handles the display
#             except Exception as e:
#                 st.warning(f"LIDA code execution failed: {str(e)}. Using fallback.")
    
#     # Fallback to smart visualization logic
#     if len(numeric_cols) >= 2:
#         if any(word in query_lower for word in ['correlation', 'relationship', 'scatter']):
#             fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
#                            title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
#         elif any(word in query_lower for word in ['trend', 'time', 'over time']) and datetime_cols:
#             fig = px.line(df, x=datetime_cols[0], y=numeric_cols[0], 
#                          title=f"Trend: {numeric_cols[0]} over Time")
#         else:
#             fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
#                            title=f"Relationship: {numeric_cols[0]} vs {numeric_cols[1]}")
    
#     elif len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
#         if any(word in query_lower for word in ['bar', 'count', 'category']):
#             fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], 
#                         title=f"Bar Chart: {numeric_cols[0]} by {categorical_cols[0]}")
#         elif any(word in query_lower for word in ['box', 'distribution']):
#             fig = px.box(df, x=categorical_cols[0], y=numeric_cols[0], 
#                         title=f"Box Plot: {numeric_cols[0]} by {categorical_cols[0]}")
#         else:
#             fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], 
#                         title=f"Analysis: {numeric_cols[0]} by {categorical_cols[0]}")
    
#     elif len(numeric_cols) >= 1:
#         if any(word in query_lower for word in ['histogram', 'distribution']):
#             fig = px.histogram(df, x=numeric_cols[0], 
#                              title=f"Distribution of {numeric_cols[0]}")
#         else:
#             fig = px.histogram(df, x=numeric_cols[0], 
#                              title=f"Distribution: {numeric_cols[0]}")
    
#     elif len(categorical_cols) >= 1:
#         value_counts = df[categorical_cols[0]].value_counts()
#         fig = px.pie(values=value_counts.values, names=value_counts.index, 
#                     title=f"Distribution of {categorical_cols[0]}")
    
#     else:
#         fig = px.scatter(df.reset_index(), y='index', 
#                         title="Data Overview")
    
#     return fig

# def main():
#     # App Header
#     st.title("🚀 Smart Data Analyzer Pro with LIDA")
#     st.markdown("*Powered by OpenRouter LLM and LIDA for Advanced Data Visualization*")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("🔧 Configuration")
        
#         # API Status
#         if os.getenv("OPENROUTER_API_KEY"):
#             st.success("✅ OpenRouter API Connected")
#         else:
#             st.error("❌ OpenRouter API Key Missing")
            
#         if lida_manager:
#             st.success("✅ LIDA Initialized")
#         else:
#             st.warning("⚠️ LIDA Fallback Mode")
        
#         st.markdown("---")
#         st.header("📊 Data Upload")
    
#     # File Upload
#     uploaded_file = st.file_uploader(
#         "Choose a CSV file", 
#         type="csv",
#         help="Upload your CSV file to start analyzing"
#     )
    
#     if uploaded_file is not None:
#         try:
#             # Load data
#             df = pd.read_csv(uploaded_file)
            
#             # Display data info
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Rows", len(df))
#             with col2:
#                 st.metric("Columns", len(df.columns))
#             with col3:
#                 st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
            
#             # Data Preview
#             with st.expander("📋 Data Preview", expanded=True):
#                 st.dataframe(df.head())
            
#             # Query Interface
#             st.header("🤔 Ask Questions About Your Data")
            
#             # Sample questions
#             sample_questions = [
#                 "Show me the correlation between variables",
#                 "What's the distribution of the main categories?",
#                 "Create a trend analysis over time",
#                 "Show me outliers in the data",
#                 "Compare different groups in the dataset"
#             ]
            
#             # Quick question buttons
#             st.subheader("Quick Questions:")
#             cols = st.columns(len(sample_questions))
#             for i, question in enumerate(sample_questions):
#                 if cols[i].button(f"Q{i+1}", help=question):
#                     st.session_state.user_query = question
            
#             # Custom query input
#             user_query = st.text_input(
#                 "Or ask your own question:",
#                 value=st.session_state.get('user_query', ''),
#                 placeholder="e.g., 'Show me the relationship between sales and profit'"
#             )
            
#             if user_query:
#                 st.session_state.user_query = user_query
                
#                 # Create two columns for visualization and insights
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     st.subheader("📊 Visualization")
#                     with st.spinner("Generating visualization..."):
#                         fig = create_smart_visualization(df, user_query)
#                         if fig:
#                             st.plotly_chart(fig, use_container_width=True)
                
#                 with col2:
#                     st.subheader("🧠 AI Insights")
#                     with st.spinner("Analyzing data..."):
#                         # Create context for LLM
#                         context = f"""
#                         Data Summary:
#                         - Shape: {df.shape}
#                         - Columns: {', '.join(df.columns)}
#                         - Numeric columns: {', '.join(df.select_dtypes(include=['number']).columns)}
#                         - Categorical columns: {', '.join(df.select_dtypes(include=['object']).columns)}
#                         - Sample data: {df.head().to_string()}
#                         """
                        
#                         insights = query_openrouter_llm(user_query, context)
#                         st.write(insights)
            
#             # Data Statistics
#             with st.expander("📈 Data Statistics"):
#                 st.subheader("Numeric Columns")
#                 numeric_cols = df.select_dtypes(include=['number']).columns
#                 if len(numeric_cols) > 0:
#                     st.dataframe(df[numeric_cols].describe())
#                 else:
#                     st.info("No numeric columns found")
                
#                 st.subheader("Categorical Columns")
#                 categorical_cols = df.select_dtypes(include=['object', 'category']).columns
#                 if len(categorical_cols) > 0:
#                     for col in categorical_cols:
#                         st.write(f"**{col}:**")
#                         st.write(df[col].value_counts().head())
#                 else:
#                     st.info("No categorical columns found")
        
#         except Exception as e:
#             st.error(f"Error loading file: {str(e)}")
#     else:
#         st.info("👆 Please upload a CSV file to get started!")
        
#         # Show example
#         st.subheader("Example Usage")
#         st.markdown("""
#         1. **Upload your CSV file** using the file uploader
#         2. **Ask questions** about your data in natural language
#         3. **Get AI-powered visualizations** generated by LIDA
#         4. **Receive insights** from the OpenRouter LLM
        
#         **Example Questions:**
#         - "Show me the correlation between sales and profit"
#         - "What's the trend of revenue over time?"
#         - "Create a distribution chart for customer ages"
#         - "Compare performance across different regions"
#         """)

# if __name__ == "__main__":
#     if 'user_query' not in st.session_state:
#         st.session_state.user_query = ''
    
#     main()









# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# import json
# import requests
# import spacy
# from dotenv import load_dotenv
# import os
# from datetime import datetime, timedelta
# import hashlib
# import base64
# from PIL import Image
# import io
# import openai
# from openai import OpenAI
# import seaborn as sns
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# # MUST BE FIRST - Configure Streamlit page
# st.set_page_config(
#     page_title="🚀 Smart Data Analyzer Pro with LIDA & OpenAI", 
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="📊"
# )

# # Load environment variables
# load_dotenv()

# # Initialize LIDA (with proper error handling)
# lida_manager = None
# try:
#     from lida import Manager, TextGenerationConfig, llm
#     from lida.components import TextGenerationConfig
    
#     # Initialize LIDA with OpenAI
#     @st.cache_resource
#     def initialize_lida():
#         """Initialize LIDA with OpenAI"""
#         try:
#             openai_api_key = os.getenv("OPENAI_API_KEY")
            
#             if not openai_api_key:
#                 st.warning("OpenAI API key not found. Some LIDA features will be disabled.")
#                 return None
            
#             # Set OpenAI API key
#             openai.api_key = openai_api_key
            
#             # Initialize LIDA manager
#             lida = Manager(text_gen=llm("openai"))
            
#             st.success("✅ LIDA initialized successfully with OpenAI!")
#             return lida
            
#         except Exception as e:
#             st.warning(f"LIDA initialization failed: {str(e)}. Using fallback visualization.")
#             return None
    
#     lida_manager = initialize_lida()
    
# except ImportError:
#     st.warning("LIDA not installed. Install with: pip install lida")
#     lida_manager = None

# # Initialize OpenAI client
# @st.cache_resource
# def initialize_openai():
#     """Initialize OpenAI client"""
#     try:
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             st.error("OpenAI API key not found in environment variables.")
#             return None
        
#         client = OpenAI(api_key=api_key)
#         st.success("✅ OpenAI client initialized successfully!")
#         return client
        
#     except Exception as e:
#         st.error(f"Failed to initialize OpenAI client: {str(e)}")
#         return None

# openai_client = initialize_openai()

# # Initialize spaCy
# @st.cache_resource
# def load_spacy_model():
#     """Load spaCy model with error handling"""
#     try:
#         nlp = spacy.load("en_core_web_sm")
#         return nlp
#     except IOError:
#         st.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
#         return None

# nlp = load_spacy_model()

# def query_openai_gpt(prompt, context="", model="gpt-3.5-turbo"):
#     """Query OpenAI GPT for insights"""
#     try:
#         if not openai_client:
#             return "OpenAI client not available."
        
#         full_prompt = f"""
#         Context: {context}
        
#         Question: {prompt}
        
#         Please provide a clear, concise analysis based on the data context provided.
#         Focus on actionable insights and key patterns in the data.
#         """
        
#         response = openai_client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a data analyst expert. Provide clear, actionable insights from data."},
#                 {"role": "user", "content": full_prompt}
#             ],
#             max_tokens=500,
#             temperature=0.7
#         )
        
#         return response.choices[0].message.content
        
#     except Exception as e:
#         return f"Error querying OpenAI: {str(e)}"

# def generate_lida_visualization(df, query):
#     """Generate visualization using LIDA"""
#     try:
#         if lida_manager is None:
#             return None, "LIDA not available"
        
#         # Convert DataFrame to LIDA format
#         summary = lida_manager.summarize(df)
        
#         # Generate visualization goals based on query
#         goals = lida_manager.goals(summary, n=3, persona="data analyst")
        
#         # Find the most relevant goal based on query
#         relevant_goal = None
#         query_words = set(query.lower().split())
        
#         for goal in goals:
#             goal_text = goal.get('question', '').lower()
#             if query_words.intersection(set(goal_text.split())):
#                 relevant_goal = goal
#                 break
        
#         if not relevant_goal and goals:
#             relevant_goal = goals[0]
        
#         if relevant_goal:
#             # Generate visualization code
#             textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
#             charts = lida_manager.visualize(
#                 summary=summary,
#                 goal=relevant_goal,
#                 textgen_config=textgen_config
#             )
            
#             if charts:
#                 return charts[0], None
        
#         return None, "No suitable visualization found"
        
#     except Exception as e:
#         return None, f"LIDA visualization error: {str(e)}"

# def execute_lida_code(chart_code, df):
#     """Safely execute LIDA-generated code"""
#     try:
#         # Create a safe execution environment
#         safe_globals = {
#             'pd': pd,
#             'plt': plt,
#             'px': px,
#             'go': go,
#             'np': np,
#             'df': df,
#             'data': df,
#             'sns': sns,
#             '__builtins__': {},
#         }
        
#         # Execute the code
#         exec(chart_code, safe_globals)
        
#         return True, None
        
#     except Exception as e:
#         return False, f"Code execution error: {str(e)}"

# def create_advanced_visualization(df, query):
#     """Create advanced visualizations based on query and data analysis"""
#     query_lower = query.lower()
    
#     # Analyze data types
#     numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#     datetime_cols = pd.to_datetime(df.select_dtypes(include=['object']), errors='ignore').select_dtypes(include=['datetime']).columns.tolist()
    
#     # Try to identify datetime columns more intelligently
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             try:
#                 pd.to_datetime(df[col].head())
#                 datetime_cols.append(col)
#             except:
#                 pass
    
#     # Remove duplicates
#     datetime_cols = list(set(datetime_cols))
    
#     # Advanced visualization logic
#     if any(word in query_lower for word in ['correlation', 'corr', 'relationship', 'relate']):
#         if len(numeric_cols) >= 2:
#             # Correlation heatmap
#             corr_matrix = df[numeric_cols].corr()
#             fig = px.imshow(corr_matrix, 
#                            text_auto=True, 
#                            aspect="auto",
#                            title="Correlation Matrix",
#                            color_continuous_scale='RdBu')
#             return fig
    
#     elif any(word in query_lower for word in ['scatter', 'plot', 'vs']):
#         if len(numeric_cols) >= 2:
#             # Enhanced scatter plot
#             color_col = categorical_cols[0] if categorical_cols else None
#             size_col = numeric_cols[2] if len(numeric_cols) > 2 else None
            
#             fig = px.scatter(df, 
#                            x=numeric_cols[0], 
#                            y=numeric_cols[1],
#                            color=color_col,
#                            size=size_col,
#                            hover_data=numeric_cols[:3],
#                            title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
#             return fig
    
#     elif any(word in query_lower for word in ['trend', 'time', 'over time', 'timeline']):
#         if datetime_cols and numeric_cols:
#             # Time series plot
#             fig = px.line(df, 
#                          x=datetime_cols[0], 
#                          y=numeric_cols[0],
#                          title=f"Trend Analysis: {numeric_cols[0]} over Time")
#             return fig
#         elif len(df) > 1 and numeric_cols:
#             # Use index as time proxy
#             fig = px.line(df.reset_index(), 
#                          x='index', 
#                          y=numeric_cols[0],
#                          title=f"Trend Analysis: {numeric_cols[0]}")
#             return fig
    
#     elif any(word in query_lower for word in ['distribution', 'histogram', 'dist']):
#         if numeric_cols:
#             # Distribution plot with statistics
#             fig = px.histogram(df, 
#                              x=numeric_cols[0], 
#                              marginal="box",
#                              title=f"Distribution of {numeric_cols[0]}")
#             return fig
    
#     elif any(word in query_lower for word in ['box', 'boxplot', 'outlier']):
#         if numeric_cols and categorical_cols:
#             # Box plot for outlier detection
#             fig = px.box(df, 
#                         x=categorical_cols[0], 
#                         y=numeric_cols[0],
#                         title=f"Box Plot: {numeric_cols[0]} by {categorical_cols[0]}")
#             return fig
#         elif numeric_cols:
#             fig = px.box(df, y=numeric_cols[0], title=f"Box Plot: {numeric_cols[0]}")
#             return fig
    
#     elif any(word in query_lower for word in ['bar', 'count', 'category', 'group']):
#         if categorical_cols and numeric_cols:
#             # Grouped bar chart
#             fig = px.bar(df, 
#                         x=categorical_cols[0], 
#                         y=numeric_cols[0],
#                         title=f"Bar Chart: {numeric_cols[0]} by {categorical_cols[0]}")
#             return fig
#         elif categorical_cols:
#             # Count plot
#             value_counts = df[categorical_cols[0]].value_counts()
#             fig = px.bar(x=value_counts.index, 
#                         y=value_counts.values,
#                         title=f"Count of {categorical_cols[0]}")
#             return fig
    
#     elif any(word in query_lower for word in ['pie', 'proportion', 'percentage']):
#         if categorical_cols:
#             value_counts = df[categorical_cols[0]].value_counts()
#             fig = px.pie(values=value_counts.values, 
#                         names=value_counts.index,
#                         title=f"Distribution of {categorical_cols[0]}")
#             return fig
    
#     # Default visualization based on data types
#     if len(numeric_cols) >= 2:
#         fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
#                         title=f"Relationship: {numeric_cols[0]} vs {numeric_cols[1]}")
#     elif len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
#         fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], 
#                     title=f"Analysis: {numeric_cols[0]} by {categorical_cols[0]}")
#     elif len(numeric_cols) >= 1:
#         fig = px.histogram(df, x=numeric_cols[0], 
#                          title=f"Distribution of {numeric_cols[0]}")
#     elif len(categorical_cols) >= 1:
#         value_counts = df[categorical_cols[0]].value_counts()
#         fig = px.pie(values=value_counts.values, names=value_counts.index, 
#                     title=f"Distribution of {categorical_cols[0]}")
#     else:
#         fig = px.scatter(df.reset_index(), y='index', 
#                         title="Data Overview")
    
#     return fig

# def generate_data_insights(df):
#     """Generate comprehensive data insights"""
#     insights = []
    
#     # Basic statistics
#     insights.append(f"📊 **Dataset Overview**: {len(df)} rows and {len(df.columns)} columns")
    
#     # Missing values
#     missing_data = df.isnull().sum()
#     if missing_data.sum() > 0:
#         insights.append(f"⚠️ **Missing Data**: {missing_data.sum()} total missing values")
    
#     # Data types
#     numeric_cols = df.select_dtypes(include=['number']).columns
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
#     insights.append(f"🔢 **Numeric Columns**: {len(numeric_cols)}")
#     insights.append(f"📝 **Categorical Columns**: {len(categorical_cols)}")
    
#     # Key statistics for numeric columns
#     if len(numeric_cols) > 0:
#         for col in numeric_cols[:3]:  # Limit to first 3 columns
#             mean_val = df[col].mean()
#             std_val = df[col].std()
#             insights.append(f"📈 **{col}**: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
    
#     # Unique values for categorical columns
#     if len(categorical_cols) > 0:
#         for col in categorical_cols[:3]:  # Limit to first 3 columns
#             unique_count = df[col].nunique()
#             insights.append(f"🏷️ **{col}**: {unique_count} unique values")
    
#     return insights

# def create_data_summary_cards(df):
#     """Create summary cards for the dataset"""
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric(
#             label="📊 Total Rows",
#             value=f"{len(df):,}",
#             delta=None
#         )
    
#     with col2:
#         st.metric(
#             label="📋 Columns",
#             value=len(df.columns),
#             delta=None
#         )
    
#     with col3:
#         missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
#         st.metric(
#             label="⚠️ Missing Data",
#             value=f"{missing_pct:.1f}%",
#             delta=None
#         )
    
#     with col4:
#         memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
#         st.metric(
#             label="💾 Memory Usage",
#             value=f"{memory_usage:.1f} MB",
#             delta=None
#         )

# def main():
#     # Custom CSS for better styling
#     st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         color: #666;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .insight-box {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 1rem 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # App Header
#     st.markdown('<h1 class="main-header">🚀 Smart Data Analyzer Pro</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Powered by OpenAI GPT & LIDA for Advanced Data Visualization & Analysis</p>', unsafe_allow_html=True)
    
#     # Sidebar Configuration
#     with st.sidebar:
#         st.header("🔧 Configuration Status")
        
#         # API Status Checks
#         if openai_client:
#             st.success("✅ OpenAI API Connected")
#         else:
#             st.error("❌ OpenAI API Key Missing")
            
#         if lida_manager:
#             st.success("✅ LIDA Initialized")
#         else:
#             st.warning("⚠️ LIDA Not Available")
        
#         if nlp:
#             st.success("✅ spaCy Model Loaded")
#         else:
#             st.warning("⚠️ spaCy Model Missing")
        
#         st.markdown("---")
        
#         # Model Selection
#         st.header("🤖 AI Model Settings")
#         selected_model = st.selectbox(
#             "Choose OpenAI Model:",
#             ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
#             index=0,
#             help="Select the OpenAI model for analysis"
#         )
        
#         use_lida = st.checkbox(
#             "Use LIDA for Visualizations",
#             value=bool(lida_manager),
#             disabled=not lida_manager,
#             help="Enable LIDA for AI-powered visualizations"
#         )
        
#         st.markdown("---")
#         st.header("📊 Data Upload")
    
#     # Main Content Area
#     # File Upload Section
#     uploaded_file = st.file_uploader(
#         "📁 Choose a CSV file to analyze",
#         type="csv",
#         help="Upload your CSV file to start the analysis"
#     )
    
#     if uploaded_file is not None:
#         try:
#             # Load and process data
#             with st.spinner("🔄 Loading and processing your data..."):
#                 df = pd.read_csv(uploaded_file)
                
#                 # Basic data cleaning
#                 # Remove completely empty rows and columns
#                 df = df.dropna(how='all').dropna(axis=1, how='all')
                
#                 # Convert obvious datetime columns
#                 for col in df.columns:
#                     if df[col].dtype == 'object' and len(df[col].dropna()) > 0:
#                         try:
#                             pd.to_datetime(df[col].head(), errors='raise')
#                             df[col] = pd.to_datetime(df[col], errors='coerce')
#                         except:
#                             pass
            
#             st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
#             # Data Summary Cards
#             create_data_summary_cards(df)
            
#             # Data Preview Section
#             with st.expander("📋 Data Preview & Information", expanded=True):
#                 tab1, tab2, tab3 = st.tabs(["🔍 Sample Data", "📊 Data Types", "📈 Statistics"])
                
#                 with tab1:
#                     st.subheader("First 10 rows of your data:")
#                     st.dataframe(df.head(10), use_container_width=True)
                
#                 with tab2:
#                     st.subheader("Column Information:")
#                     col_info = pd.DataFrame({
#                         'Column': df.columns,
#                         'Data Type': df.dtypes.astype(str),
#                         'Non-Null Count': df.count(),
#                         'Unique Values': [df[col].nunique() for col in df.columns],
#                         'Missing Values': df.isnull().sum()
#                     })
#                     st.dataframe(col_info, use_container_width=True)
                
#                 with tab3:
#                     st.subheader("Statistical Summary:")
#                     numeric_cols = df.select_dtypes(include=['number']).columns
#                     if len(numeric_cols) > 0:
#                         st.dataframe(df[numeric_cols].describe(), use_container_width=True)
#                     else:
#                         st.info("No numeric columns found for statistical summary.")
            
#             # Query Interface
#             st.header("🤔 Ask Questions About Your Data")
            
#             # Predefined question categories
#             st.subheader("💡 Quick Analysis Options:")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.markdown("**🔍 Exploratory Analysis**")
#                 if st.button("Show Data Overview", key="overview"):
#                     st.session_state.user_query = "Give me a comprehensive overview of this dataset"
#                 if st.button("Find Correlations", key="corr"):
#                     st.session_state.user_query = "Show me correlations between variables"
#                 if st.button("Detect Outliers", key="outliers"):
#                     st.session_state.user_query = "Help me identify outliers in the data"
            
#             with col2:
#                 st.markdown("**📊 Visualizations**")
#                 if st.button("Distribution Analysis", key="dist"):
#                     st.session_state.user_query = "Show me the distribution of key variables"
#                 if st.button("Trend Analysis", key="trend"):
#                     st.session_state.user_query = "Create a trend analysis over time"
#                 if st.button("Comparative Analysis", key="compare"):
#                     st.session_state.user_query = "Compare different groups in the dataset"
            
#             with col3:
#                 st.markdown("**🧠 Advanced Insights**")
#                 if st.button("Key Patterns", key="patterns"):
#                     st.session_state.user_query = "What are the key patterns in this data?"
#                 if st.button("Business Insights", key="business"):
#                     st.session_state.user_query = "What business insights can you derive from this data?"
#                 if st.button("Recommendations", key="recommendations"):
#                     st.session_state.user_query = "What recommendations do you have based on this data?"
            
#             # Custom Query Input
#             st.subheader("✍️ Or Ask Your Own Question:")
#             user_query = st.text_area(
#                 "Enter your question about the data:",
#                 value=st.session_state.get('user_query', ''),
#                 placeholder="e.g., 'What factors influence sales the most?' or 'Show me the relationship between price and demand'",
#                 height=100
#             )
            
#             analyze_button = st.button("🚀 Analyze Data", type="primary")
            
#             if (user_query or st.session_state.get('user_query')) and analyze_button:
#                 query = user_query or st.session_state.get('user_query', '')
#                 st.session_state.user_query = query
                
#                 # Create analysis layout
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     st.subheader("📊 Data Visualization")
                    
#                     with st.spinner("🎨 Creating visualization..."):
#                         # Try LIDA first if enabled and available
#                         if use_lida and lida_manager:
#                             lida_chart, lida_error = generate_lida_visualization(df, query)
                            
#                             if lida_chart and not lida_error:
#                                 st.info("🎯 LIDA Generated Visualization:")
                                
#                                 # Display LIDA chart code
#                                 with st.expander("👨‍💻 View Generated Code"):
#                                     st.code(lida_chart.code, language='python')
                                
#                                 # Execute LIDA code
#                                 success, error = execute_lida_code(lida_chart.code, df)
                                
#                                 if not success:
#                                     st.warning(f"LIDA code execution failed: {error}")
#                                     st.info("🔄 Falling back to standard visualization...")
#                                     fig = create_advanced_visualization(df, query)
#                                     st.plotly_chart(fig, use_container_width=True)
#                             else:
#                                 st.info("🔄 Using standard visualization (LIDA unavailable)")
#                                 fig = create_advanced_visualization(df, query)
#                                 st.plotly_chart(fig, use_container_width=True)
#                         else:
#                             # Use standard advanced visualization
#                             fig = create_advanced_visualization(df, query)
#                             st.plotly_chart(fig, use_container_width=True)
                
#                 with col2:
#                     st.subheader("🧠 AI Analysis")
                    
#                     with st.spinner("🤖 Generating insights..."):
#                         # Create comprehensive context for AI
#                         context = f"""
#                         Dataset Information:
#                         - Shape: {df.shape[0]} rows, {df.shape[1]} columns
#                         - Columns: {', '.join(df.columns.tolist())}
#                         - Numeric columns: {', '.join(df.select_dtypes(include=['number']).columns.tolist())}
#                         - Categorical columns: {', '.join(df.select_dtypes(include=['object', 'category']).columns.tolist())}
#                         - Missing values: {df.isnull().sum().sum()} total
                        
#                         Sample Data:
#                         {df.head().to_string()}
                        
#                         Statistical Summary:
#                         {df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else 'No numeric data available'}
#                         """
                        
#                         insights = query_openai_gpt(query, context, selected_model)
                        
#                         # Display insights in a nice format
#                         st.markdown(f"""
#                         <div class="insight-box">
#                         {insights}
#                         </div>
#                         """, unsafe_allow_html=True)
                
#                 # Additional Analysis Section
#                 st.subheader("📈 Automatic Data Insights")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("**🔍 Key Findings:**")
#                     auto_insights = generate_data_insights(df)
#                     for insight in auto_insights:
#                         st.markdown(f"- {insight}")
                
#                 with col2:
#                     st.markdown("**📊 Data Quality Report:**")
                    
#                     # Data quality metrics
#                     total_cells = len(df) * len(df.columns)
#                     missing_cells = df.isnull().sum().sum()
#                     completeness = ((total_cells - missing_cells) / total_cells) * 100
                    
#                     st.metric("Data Completeness", f"{completeness:.1f}%")
                    
#                     # Duplicate rows
#                     duplicates = df.duplicated().sum()
#                     st.metric("Duplicate Rows", duplicates)
                    
#                     # Data types distribution
#                     numeric_pct = (len(df.select_dtypes(include=['number']).columns) / len(df.columns)) * 100
#                     st.metric("Numeric Columns", f"{numeric_pct:.0f}%")
        
#         except Exception as e:
#             st.error(f"❌ Error processing file: {str(e)}")
#             st.info("Please ensure your CSV file is properly formatted and try again.")
    
#     else:
#         # Welcome section when no file is uploaded
#         st.info("👆 Upload a CSV file to begin your data analysis journey!")
        
#         # Feature showcase
#         st.subheader("✨ What You Can Do:")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.markdown("""
#             **🤖 AI-Powered Analysis**
#             - Natural language queries
#             - GPT-powered insights
#             - Automated pattern detection
#             - Business recommendations
#             """)
        
#         with col2:
#             st.markdown("""
#             **📊 Advanced Visualizations**
#             - LIDA-generated charts
#             - Interactive Plotly graphs
#             - Correlation matrices
#             - Time series analysis
#             """)
        
#         with col3:
#             st.markdown("""
#             **🔍 Data Intelligence**
#             - Automatic data profiling
#             - Quality assessment
#             - Missing data analysis
#             - Statistical summaries
#             """)
        
#         # Example usage
#         st.subheader("💡 Example Questions You Can Ask:")
        
#         example_questions = [
#             "What are the main factors that drive sales in this dataset?",
#             "Show me the correlation between different variables",
#             "What trends can you identify over time?",
#             "Are there any outliers I should be concerned about?",
#             "What insights can help improve business performance?",
#             "How do different categories compare to each other?",
#             "What patterns exist in customer behavior?",
#             "Which variables are most predictive of the target outcome?"
#         ]
        
#         for i, question in enumerate(example_questions, 1):
#             st.markdown(f"{i}. *{question}*")

# # Initialize session state
# if 'user_query' not in st.session_state:
#     st.session_state.user_query = ''

# if __name__ == "__main__":
#     main()