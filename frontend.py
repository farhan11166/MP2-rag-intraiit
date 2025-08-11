import streamlit as st
import requests
import time
import markdown
from weasyprint import HTML
import io
# Streamlit UI setup
st.set_page_config(page_title="📄 AI-Powered PDF Summarizer", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #282c34; /* Darker background */
            color: #abb2bf; /* Lighter, less harsh text */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Modern font */
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #61afef; /* Softer blue */
            background-color: #3e4451; /* Darker input background */
            color: #d1d5db; /* Light gray input text */
        }
        .stButton>button {
            background-color: #61afef; /* Softer blue button */
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 28px;
            border-radius: 8px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #569cd6; /* Slightly darker on hover */
            transform: translateY(-2px);
        }
        .stMarkdown, .stSubheader {
            color: #e06c75; /* Soft red for headers */
            font-weight: bold;
        }
        .summary-section {
            background-color: #3e4451;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #61afef;
        }
        .section-title {
            color: #61afef;
            font-size: 24px;
            margin-bottom: 15px;
        }
        .section-content {
            color: #d1d5db;
            font-size: 16px;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

# Professional header
st.title("📄 AI-Powered PDF Summarizer")
st.markdown("Extract and summarize research papers with AI-powered efficiency.")

# Input for PDF URL
#pdf_url = st.text_input("🔗 Add the .pdf file or arxiv", 
#                        placeholder="")
pdf_file=st.file_uploader("Choose a file", type=["pdf", "txt", "csv"])

# Placeholder for status messages
status_placeholder = st.empty()

def format_section(title, content):
    """Format a section of the summary with consistent styling"""
    return f"""
    <div class="summary-section">
        <div class="section-title">{title}</div>
        <div class="section-content">{content}</div>
    </div>
    """

# Add a spinner and professional feedback system
if st.button("🚀 Summarize PDF"):
    if pdf_file:
        with st.spinner("⏳ Processing... This may take a few minutes."):
            status_placeholder.info("⏳ Fetching and summarizing the document...")

              
            
            try:
                files = {"file": (pdf_file.name, pdf_file.getvalue(), pdf_file.type)}

                response = requests.post(
                    "http://localhost:8000/summarize_arxiv/",
                    files=files,
                    timeout=3600
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "error" in data:
                        status_placeholder.error(f"❌ {data['error']}")
                    else:
                        summary = data.get("summary", "No summary generated.")
                        status_placeholder.success("✅ Summary Ready!")
                        
                        # Split the summary into sections and display them
                        sections = summary.split("#")[1:]  # Skip empty first split
                        
                        for section in sections:
                            if section.strip():
                                # Split section into title and content
                                parts = section.split("\n", 1)
                                if len(parts) == 2:
                                    title, content = parts
                                    st.markdown(
                                        format_section(title.strip(), content.strip()),
                                        unsafe_allow_html=True
                                    )
                        
                        # Add download button for the summary
                        html = markdown.markdown(summary)

# Convert HTML to PDF bytes using WeasyPrint
                        pdf_bytes = HTML(string=html).write_pdf()
                

# Create an in-memory binary stream for the PDF
                        pdf_buffer = io.BytesIO(pdf_bytes)

                        st.download_button(
                            "⬇️ Download Summary",
                            data=pdf_buffer,
                            file_name=f"{pdf_file.name}_summary.pdf",
                            mime="application/markdown"
                        )
                else:
                    status_placeholder.error("❌ Failed to process the PDF. Please check the URL and try again.")
            except requests.exceptions.Timeout:
                status_placeholder.error("⚠️ Request timed out. Please try again later.")
            except Exception as e:
                status_placeholder.error(f"⚠️ An error occurred: {str(e)}")
    else:
        status_placeholder.warning("⚠️ Please enter a valid Arxiv PDF URL.")

# Add helpful instructions at the bottom
st.markdown("---")
st.markdown("""
### 📝 Notes:
- Processing typically takes 3-5 minutes depending on paper length
- Only Arxiv PDF URLs are supported
- The summary is structured into key sections for better readability
- You can download the summary as a markdown file
""")
