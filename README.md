# 📄 AI-Powered PDF Summarizer

🚀 **AI-Powered PDF Summarizer** is a tool that extracts and summarizes **research papers** from **Uploaded PDFs** using **gemini-1.5-flash**. The system provides structured, downloadable summaries to help researchers and professionals quickly grasp key findings.

![PDF Summarizer UI](https://github.com/farhan11166/MP2-rag-intraiit/blob/main/PDF_SUMMARIZER.png)

---

## 🛠 Features

- 🌐 **Upload a research paper pdf** to fetch and summarize papers.
- 📑 **Extracts technical content** (architecture, implementation, results).
- 🔍 **Optimized for large text processing** with **parallel summarization**.
- 🎨 **Modern UI** built with **Streamlit**.
- 📥 **Download summary as a Markdown file**.

---

## 🚀 Tech Stack

| Component         | Technology |
|------------------|------------|
| **Frontend**     | [Streamlit](https://streamlit.io/) |
| **Backend**      | [FastAPI](https://fastapi.tiangolo.com/) |
| **LLM Model**    | [Google Gemma 3](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/1-5-flash) |
| **PDF Processing** | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) |
| **Text Chunking** | [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/) |
---

## 🎬 Demo

1️⃣ **Upload a pdf from your local device **  
2️⃣ **Click "Summarize PDF"** 🚀  
3️⃣ **Get a structured summary** with **technical insights** 📝  
4️⃣ **Download summary as Pdf** 📥  

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/farhan11166/MP2-rag-intraiit
cd MP2-rag-intraiit

```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ SET GOOGLE GEMINI AND EXA API KEY in a.env file


GOOGLE_API_KEY=[YOUR-API-KEY]
EXA_API_KEY=[YOUR-API-KEY]

```

### 3️⃣ Start the Backend (FastAPI)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4️⃣ Start the Frontend (Streamlit)

```bash
streamlit run frontend.py
```

---

## 📜 API Endpoints

### 🔹 Health Check

```http
GET /health
```

Response:
```json
{"status": "ok", "message": "FastAPI backend is running!"}
```

### 🔹 Summarize
Summarize 
```
POST /summarize/
```
Request Body:
```
{
  "filename": "example.pdf",
  "content": "<base64-encoded-pdf>"
}
```
Response:
```
{
  "summary": "Structured summary of the research paper..."
}
```
