# Hungama BI

**Hungama BI** is a Streamlit-powered Smart Data Analyzer for YouTube Revenue Validation and Business Intelligence.  
It enables you to upload YouTube metadata and revenue CSVs, ask business questions in natural language, and receive AI-driven insights, visualizations, and context-aware analytics.

---

## 🚀 Features

- **Upload YouTube metadata and revenue CSVs**
- **AI-powered Q&A**: Ask business questions in plain English
- **Context-aware conversation memory**
- **Automatic visualizations**: Charts and tables generated from your queries
- **Advanced filtering**: By label, artist, language, date, and view count
- **Export**: Download filtered data and conversation history

---

## 🏁 Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/HungamaBI.git
   cd HungamaBI
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Add your API key:**  
   Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

4. **Add your YouTube metadata:**  
   Place your `youtube_metadata.json` file in the project root.

5. **Run the app:**
   ```sh
   streamlit run app.py
   ```

---

## 📁 File Structure

- `app.py` — Main Streamlit app
- `youtube_metadata.json` — Your YouTube video metadata (required)
- `requirements.txt` — Python dependencies

---

## 💡 Sample Questions to Try

- What is our top performing video by revenue per view?
- How does engagement vary by language?
- Show me the monthly revenue trend for the past year
- Which artists generate the most views?

---

## 📝 License

MIT License

---

*Built with ❤️ using Streamlit and OpenAI/DeepSeek LLMs.*