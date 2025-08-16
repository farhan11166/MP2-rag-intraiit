import streamlit as st
import requests
import time
import markdown
from weasyprint import HTML
import io
import json
from crewai import LLM
# Streamlit UI setup
st.set_page_config(page_title="📚 AI-Powered PDF Summarizer", layout="wide")

# Enhanced CSS styling
st.markdown("""
    <style>
        .main > div {
            padding: 1rem 2rem;
        }
        
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #61afef;
            background-color: #3e4451;
            color: #d1d5db;
        }
        
        .stButton>button {
            background-color: #61afef;
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 28px;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
            max-width: 300px;
        }
        
        .stButton>button:hover {
            background-color: #569cd6;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .stButton>button:disabled {
            background-color: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        .summary-section {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #61afef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section-title {
            color: #2c3e50;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            border-bottom: 2px solid #61afef;
            padding-bottom: 5px;
        }
        
        .section-content {
            color: #34495e;
            font-size: 16px;
            line-height: 1.7;
            white-space: pre-wrap;
        }
        
        .citations-container {
            background-color: #e8f4f8;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border-left: 5px solid #17a2b8;
        }
        
        .citation-item {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 3px solid #17a2b8;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .citation-title {
            font-weight: bold;
            color: #0056b3;
            margin-bottom: 5px;
        }
        
        .citation-url {
            color: #6c757d;
            font-size: 14px;
            word-break: break-all;
        }
        
        .citation-description {
            color: #495057;
            margin-top: 8px;
            font-style: italic;
        }
        
        .chat-container {
            border-top: 2px solid #e9ecef;
            margin-top: 2rem;
            padding-top: 1rem;
        }
        
        .source-tag {
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin: 2px;
            display: inline-block;
        }
        
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            color: #856404;
        }
        
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            color: #721c24;
        }
        
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            color: #155724;
        }
        
        .info-highlight {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main > div {
                padding: 0.5rem 1rem;
            }
            
            .section-title {
                font-size: 18px;
            }
            
            .section-content {
                font-size: 14px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📚 AI-Powered Document Summarizer with Citations")
st.markdown("Extract, summarize, and research documents with AI-powered analysis and web citations.")

# Input validation function
def validate_file(file):
    """Validate uploaded file"""
    if file is None:
        return False, "Please upload a file"
    
    # Check file size (max 50MB)
    if file.size > 50 * 1024 * 1024:
        return False, "File size too large. Maximum 50MB allowed."
    
    # Check file type more strictly
    allowed_types = ['application/pdf', 'text/plain', 'text/csv']
    if file.type not in allowed_types:
        return False, f"Invalid file type: {file.type}. Only PDF, TXT, and CSV files are allowed."
    
    return True, "File valid"

def display_citations(citations):
    """Display citations in a formatted container"""
    if not citations:
        return
    
    citations_html = '<div class="citations-container"><h3>📚 Web Citations & Related Sources</h3>'
    
    for i, citation in enumerate(citations, 1):
        title = citation.get('title', f'Source {i}')
        url = citation.get('url', '#')
        description = citation.get('description', 'No description available')
        
        citations_html += f'''
        <div class="citation-item">
            <div class="citation-title">[{i}] {title}</div>
            <div class="citation-url">🔗 <a href="{url}" target="_blank">{url}</a></div>
            <div class="citation-description">{description}</div>
        </div>
        '''
    
    citations_html += '</div>'
    st.markdown(citations_html, unsafe_allow_html=True)

def safe_api_call(url, **kwargs):
    """Make API call with comprehensive error handling"""
    try:
        response = requests.post(url, **kwargs)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to backend server. Please ensure the server is running on localhost:8000")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. The document might be too large or the server is overloaded")
    except requests.exceptions.HTTPError as e:
        if response.status_code == 422:
            raise Exception("Invalid request format. Please try again")
        elif response.status_code == 500:
            raise Exception("Server internal error. Please try again later")
        else:
            raise Exception(f"Server error (HTTP {response.status_code}): {str(e)}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")

# Initialize all session state variables
def initialize_session_state():
    """Initialize all session state variables"""
    if "textforbot" not in st.session_state:
        st.session_state.textforbot = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "current_document_name" not in st.session_state:
        st.session_state.current_document_name = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "citations" not in st.session_state:
        st.session_state.citations = []

initialize_session_state()

# File upload section
col1, col2 = st.columns([3, 1])
with col1:
    pdf_file = st.file_uploader(
        "Choose a file", 
        type=["pdf", "txt", "csv"],
        help="Upload PDF, TXT, or CSV files (max 50MB)"
    )

# Validate file when uploaded
file_valid = False
if pdf_file:
    is_valid, message = validate_file(pdf_file)
    if is_valid:
        file_valid = True
        st.success(f"✅ File loaded: {pdf_file.name} ({pdf_file.size / 1024:.1f} KB)")
    else:
        st.error(f"❌ {message}")

# Placeholder for status messages
status_placeholder = st.empty()

# Process document button
process_button = st.button(
    "🚀 Analyze Document & Generate Citations", 
    disabled=not file_valid or st.session_state.processing,
    help="Upload a valid file to enable analysis with web citations"
)

if process_button and file_valid and not st.session_state.processing:
    st.session_state.processing = True
    
    with st.spinner("⏳ Processing document with AI agents... This may take several minutes as we analyze your document and find relevant citations."):
        status_placeholder.info("⏳ AI agents are analyzing document and searching for citations...")

        try:
            # Reset previous data when processing new document
            if st.session_state.current_document_name != pdf_file.name:
                st.session_state.textforbot = []
                st.session_state.chat_history = []
                st.session_state.document_processed = False
                st.session_state.citations = []
            
            # Prepare file data
            files = {"file": (pdf_file.name, pdf_file.getvalue(), pdf_file.type)}
            
            # Make API call with longer timeout for agent processing
            response = safe_api_call(
                "http://localhost:8000/summarize/",
                files=files,
                timeout=600  # 10 minutes timeout for agent processing
            )

            data = response.json()
            
            if "error" in data:
                status_placeholder.error(f"❌ Server Error: {data['error']}")
            else:
                summary = data.get("summary", "No summary generated.")
                citations = data.get("citations", [])
                status_placeholder.success("✅ Document analyzed successfully!")

                # Process and store textforbot data
                textforbot_data = data.get("textforbot", [])
                
                # Handle different data types that might come from backend
                if isinstance(textforbot_data, str):
                    st.session_state.textforbot = [textforbot_data]
                elif isinstance(textforbot_data, list):
                    # Convert all items to strings and filter empty ones
                    st.session_state.textforbot = [
                        str(item).strip() 
                        for item in textforbot_data 
                        if item and str(item).strip()
                    ]
                else:
                    st.warning("⚠️ Unexpected data format from server. Chat functionality may be limited.")
                    st.session_state.textforbot = []

                # Store citations
                st.session_state.citations = citations

                # Mark as processed
                st.session_state.document_processed = True
                st.session_state.current_document_name = pdf_file.name

                # Display summary with better error handling
                try:
                    if summary and summary.strip():
                        # Display the summary as markdown
                        st.markdown("## 📋 Document Analysis Summary")
                        st.markdown(summary)
                        
                        # Display citations
                        if citations:
                            display_citations(citations)
                            st.success(f"✅ Found {len(citations)} relevant web citations!")
                        else:
                            st.info("ℹ️ No specific web citations were found for this document.")
                    else:
                        st.warning("⚠️ No summary content received from server")

                except Exception as display_error:
                    st.error(f"Error displaying summary: {str(display_error)}")
                    # Fallback: display raw summary
                    st.text_area("Raw Summary", summary, height=300)

                # Enhanced PDF download functionality
                try:
                    if summary and summary.strip():
                        html_content = markdown.markdown(summary)
                        
                        # Add citations to HTML
                        citations_html = ""
                        if citations:
                            citations_html = "<h2>References and Citations</h2>"
                            for i, citation in enumerate(citations, 1):
                                citations_html += f"""
                                <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #007bff; background-color: #f8f9fa;">
                                    <p><strong>[{i}] {citation.get('title', 'Unknown Title')}</strong></p>
                                    <p><em>URL:</em> <a href="{citation.get('url', '#')}" target="_blank">{citation.get('url', 'No URL')}</a></p>
                                    <p>{citation.get('description', '')}</p>
                                </div>
                                """
                        
                        # Add basic HTML structure for better PDF rendering
                        full_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>Document Analysis with Citations - {pdf_file.name}</title>
                            <style>
                                body {{ 
                                    font-family: 'Arial', 'Helvetica', sans-serif; 
                                    margin: 2cm; 
                                    line-height: 1.6; 
                                    color: #333;
                                }}
                                h1, h2, h3 {{ 
                                    color: #2c3e50; 
                                    border-bottom: 2px solid #3498db;
                                    padding-bottom: 10px;
                                }}
                                h1 {{ font-size: 24px; }}
                                h2 {{ font-size: 20px; margin-top: 30px; }}
                                h3 {{ font-size: 18px; }}
                                p {{ margin-bottom: 1em; }}
                                .citation {{ 
                                    background-color: #f8f9fa; 
                                    padding: 15px; 
                                    margin: 15px 0; 
                                    border-left: 4px solid #007bff;
                                    border-radius: 5px;
                                }}
                                .header {{
                                    text-align: center;
                                    margin-bottom: 30px;
                                    padding: 20px;
                                    background-color: #f1f2f6;
                                    border-radius: 10px;
                                }}
                                a {{ color: #007bff; text-decoration: none; }}
                                a:hover {{ text-decoration: underline; }}
                            </style>
                        </head>
                        <body>
                            <div class="header">
                                <h1>📚 Document Analysis Report</h1>
                                <p><strong>Document:</strong> {pdf_file.name}</p>
                                <p><strong>Generated on:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                            </div>
                            {html_content}
                            <div class="citation">
                                {citations_html}
                            </div>
                        </body>
                        </html>
                        """
                        
                        pdf_bytes = HTML(string=full_html).write_pdf()
                        pdf_buffer = io.BytesIO(pdf_bytes)

                        st.download_button(
                            "⬇️ Download Complete Analysis as PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=f"{pdf_file.name.rsplit('.', 1)[0]}_analysis_with_citations.pdf",
                            mime="application/pdf",
                            help="Download the complete analysis with web citations as a formatted PDF document"
                        )
                    
                except Exception as pdf_error:
                    st.error(f"❌ PDF generation failed: {str(pdf_error)}")
                    # Offer text download as fallback
                    text_content = summary
                    if citations:
                        text_content += "\n\nReferences and Citations:\n" + "="*50 + "\n"
                        for i, citation in enumerate(citations, 1):
                            text_content += f"\n[{i}] {citation.get('title', 'Unknown')}\n"
                            text_content += f"URL: {citation.get('url', 'No URL')}\n"
                            text_content += f"Description: {citation.get('description', '')}\n"
                            text_content += "-" * 40 + "\n"
                    
                    st.download_button(
                        "⬇️ Download Analysis as Text",
                        data=text_content,
                        file_name=f"{pdf_file.name.rsplit('.', 1)[0]}_analysis.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            status_placeholder.error(f"❌ {str(e)}")
            st.session_state.document_processed = False
        
        finally:
            st.session_state.processing = False

# Chat Section
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.subheader("💬 Ask Questions About Your Document")

if not st.session_state.document_processed:
    st.info("📄 Process a document first to enable the enhanced chat functionality with web context.")
elif not st.session_state.textforbot:
    st.warning("⚠️ No content available for chat. The document processing may have failed.")
else:
    # Show available citations as context
    if st.session_state.citations:
        with st.expander("📚 Available Citation Context"):
            for i, citation in enumerate(st.session_state.citations[:5], 1):
                st.write(f"**[{i}]** {citation.get('title', 'Unknown Title')}")
                st.write(f"🔗 {citation.get('url', 'No URL')}")
                if citation.get('description'):
                    st.write(f"📝 {citation.get('description')}")
                st.write("---")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for role, msg, sources in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)
                if role == "assistant" and sources:
                    st.write("**Sources used:**")
                    for source in sources:
                        if source.get("url"):
                            st.markdown(f'<span class="source-tag">🔗 <a href="{source["url"]}" target="_blank">{source.get("title", "Unknown")}</a></span>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="source-tag">📄 {source.get("title", "Document")}</span>', 
                                      unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input(
        "Ask a question about the document... (Enhanced with web context)",
        disabled=not st.session_state.textforbot
    )

    if user_question and user_question.strip():
        # Validate question length
        if len(user_question) > 500:
            st.error("❌ Question too long. Please keep it under 500 characters.")
        else:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_question, []))
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)

            # Prepare and send request
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking with enhanced context..."):
                    try:
                        payload = {
                            "messages": st.session_state.textforbot,
                            "question": user_question
                        }
                        
                        response = safe_api_call(
                            "http://localhost:8000/chat/",
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=120  # 2 minutes for enhanced chat
                        )
                        
                        result = response.json()
                        backend_answer = result.get("answer", "I don't know")
                        sources = result.get("sources", [])
                        
                        # Validate response
                        if not backend_answer or backend_answer.strip() == "":
                            backend_answer = "I received an empty response. Please try rephrasing your question."
                        
                    except Exception as e:
                        backend_answer = f"⚠️ Chat Error: {str(e)}"
                        sources = []

                    # Display response
                    st.markdown(backend_answer)
                    
                    # Display sources if available
                    if sources:
                        st.write("**Sources used in this response:**")
                        for source in sources:
                            if source.get("url"):
                                st.markdown(f'<span class="source-tag">🔗 <a href="{source["url"]}" target="_blank">{source.get("title", "Unknown")}</a></span>', 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span class="source-tag">📄 {source.get("title", "Document")}</span>', 
                                          unsafe_allow_html=True)
                    
                    # Store response with sources
                    st.session_state.chat_history.append(("assistant", backend_answer, sources))

st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Debug information
if st.checkbox("Show Debug Info", help="For troubleshooting purposes"):
    with st.expander("Debug Information"):
        st.write("**Session State:**")
        st.write(f"- Document processed: {st.session_state.document_processed}")
        st.write(f"- Current document: {st.session_state.current_document_name}")
        st.write(f"- Processing: {st.session_state.processing}")
        st.write(f"- Textforbot chunks: {len(st.session_state.textforbot)}")
        st.write(f"- Chat history length: {len(st.session_state.chat_history)}")
        st.write(f"- Citations available: {len(st.session_state.citations)}")
        
        if st.session_state.textforbot:
            st.write("**First few characters of first chunk:**")
            st.code(st.session_state.textforbot[0][:200] + "..." if len(st.session_state.textforbot[0]) > 200 else st.session_state.textforbot[0])
        
        if st.session_state.citations:
            st.write("**Available Citations:**")
            for i, citation in enumerate(st.session_state.citations, 1):
                st.write(f"{i}. {citation.get('title', 'No title')}: {citation.get('url', 'No URL')}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    📚 AI-Powered Document Analysis | 🔗 Web Citations | 🤖 CrewAI Powered<br>
    Advanced document summarization with intelligent web research and citation integration.
</div>
""", unsafe_allow_html=True)