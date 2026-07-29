# 📚 My Free RAG Application

A complete Retrieval-Augmented Generation (RAG) app built with 100% free tools.
Perfect for students and beginners!

## 🎯 What This App Does

1. **Upload PDFs** - Add your study notes, research papers, or any PDF documents
2. **Ask Questions** - Type questions in plain English
3. **Get Answers** - AI answers based ONLY on your uploaded documents

## 🛠️ Tech Stack (All Free!)

| Component | Tool | Cost |
|-----------|------|------|
| Web UI | Streamlit | Free |
| AI Brain | Gemini 2.0 Flash | Free tier (1,500 req/day) |
| Text → Vectors | Gemini Embedding | Free tier |
| Vector Search | FAISS | Open source (free) |
| PDF Reading | PyPDF | Open source (free) |
| Hosting | Streamlit Community Cloud | Free |

## 🚀 Quick Start (Run Locally)

### Step 1: Get Your Free API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your key

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Secrets
Create a folder named `.streamlit` and inside it create `secrets.toml`:
```toml
GEMINI_API_KEY = "your-actual-api-key-here"
```

### Step 4: Run the App
```bash
streamlit run app.py
```

Your app will open at `http://localhost:8501`

## 🌐 Deploy to the Internet (Free URL!)

### Step 1: Push to GitHub
1. Create a free GitHub account at [github.com](https://github.com)
2. Create a new repository (name it `my-rag-app`)
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `.gitignore` (optional)

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. **IMPORTANT**: Click "Advanced settings" → "Secrets"
6. Add: `GEMINI_API_KEY = "your-api-key"`
7. Click "Deploy!"

🎉 You'll get a free URL like `https://yourname-my-rag-app.streamlit.app`

## 📁 Project Structure

```
my-rag-app/
├── app.py              # Main application code
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml    # API keys (NEVER commit this!)
└── README.md           # This file
```

## ⚠️ Important Notes

- **Free Tier Limits**: Gemini free tier allows ~15 requests per minute and ~1,500 per day. Perfect for school projects!
- **In-Memory Storage**: This app uses FAISS in-memory. If the app restarts, you'll need to re-upload documents. This is fine for demos.
- **PDF Quality**: Scanned/image PDFs won't work well. Use text-based PDFs.
- **No Credit Card**: None of these services require a credit card for the free tier.

## 🎓 How RAG Works (Simple Explanation)

```
Your Question → Find Similar Text in PDFs → AI Reads That Text → Answer
```

1. **Chunking**: Your PDF is split into small paragraphs
2. **Embedding**: Each paragraph is converted to a number vector (like a fingerprint)
3. **Indexing**: All vectors are stored in FAISS for fast searching
4. **Retrieval**: Your question is also converted to a vector, and FAISS finds the closest matches
5. **Generation**: Gemini reads only those matching paragraphs and answers your question

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "API Key not found" | Check your secrets.toml or Streamlit Cloud secrets |
| "Error reading PDF" | Make sure it's a text-based PDF, not a scanned image |
| App is slow | First upload takes time to create embeddings. Be patient! |
| Wrong answers | Try asking more specific questions. The AI only knows your uploaded docs |

## 📚 Learn More

- [Streamlit Docs](https://docs.streamlit.io)
- [Gemini API Docs](https://ai.google.dev)
- [What is RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)

---
**Built with ❤️ for students who want to learn AI without spending money!**
