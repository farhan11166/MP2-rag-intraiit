import streamlit as st
import requests
import time
import markdown
from weasyprint import HTML
import io
import json

# Streamlit UI setup
st.set_page_config(page_title="📄 AI-Powered PDF Summarizer", layout="wide")

# Custom CSS with better responsive design and accessibility
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
        
        .chat-container {
            border-top: 2px solid #e9ecef;
            margin-top: 2rem;
            padding-top: 1rem;
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

# Professional header
st.title("📄 AI-Powered PDF Summarizer")
st.markdown("Extract and summarize research papers with AI-powered efficiency.")

# Input validation function
def validate_file(file):
    """Validate uploaded file"""
    if file is None:
        return False, "Please upload a file"
    
    # Check file size (max 50MB)
    if file.size > 50 * 1024 * 1024:
        return False, "File size too large. Maximum 50MB allowed."
    
    # Check file type more strictly
    allowed_types = ['application/pdf', 'text/plain', 'text/csv', 'application/vnd.ms-excel']
    if file.type not in allowed_types:
        return False, f"Invalid file type: {file.type}. Only PDF, TXT, and CSV files are allowed."
    
    return True, "File valid"

def format_section(title, content):
    """Format a section of the summary with consistent styling"""
    # Escape HTML characters to prevent XSS
    title = title.replace('<', '&lt;').replace('>', '&gt;')
    content = content.replace('<', '&lt;').replace('>', '&gt;')
    
    return f"""
    <div class="summary-section">
        <div class="section-title">{title}</div>
        <div class="section-content">{content}</div>
    </div>
    """

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

# Process PDF button with proper state management
process_button = st.button(
    "🚀 Summarize Document", 
    disabled=not file_valid or st.session_state.processing,
    help="Upload a valid file to enable summarization"
)

if process_button and file_valid and not st.session_state.processing:
    st.session_state.processing = True
    
    with st.spinner("⏳ Processing... This may take a few minutes."):
        status_placeholder.info("⏳ Uploading and processing document...")

        try:
            # Reset previous data when processing new document
            if st.session_state.current_document_name != pdf_file.name:
                st.session_state.textforbot = []
                st.session_state.chat_history = []
                st.session_state.document_processed = False
            
            # Prepare file data
            files = {"file": (pdf_file.name, pdf_file.getvalue(), pdf_file.type)}
            
            # Make API call
            response = safe_api_call(
                "http://localhost:8000/summarize_arxiv/",
                files=files,
                timeout=1800  # 30 minutes timeout
            )

            data = response.json()
            
            if "error" in data:
                status_placeholder.error(f"❌ Server Error: {data['error']}")
            else:
                summary = data.get("summary", "No summary generated.")
                status_placeholder.success("✅ Document processed successfully!")

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

                # Mark as processed
                st.session_state.document_processed = True
                st.session_state.current_document_name = pdf_file.name

                # Display summary with better error handling
                try:
                    if summary and summary.strip():
                        # Split the summary into sections more robustly
                        if "#" in summary:
                            sections = [s.strip() for s in summary.split("#") if s.strip()]
                        else:
                            # If no markdown headers, treat as single section
                            sections = [f"Summary\n{summary}"]
                        
                        for section in sections:
                            if section:
                                lines = section.split("\n", 1)
                                if len(lines) >= 2:
                                    title, content = lines[0].strip(), lines[1].strip()
                                else:
                                    title, content = "Summary", lines[0] if lines else "No content"
                                
                                if title and content:
                                    st.markdown(
                                        format_section(title, content),
                                        unsafe_allow_html=True
                                    )
                    else:
                        st.warning("⚠️ No summary content received from server")

                except Exception as display_error:
                    st.error(f"Error displaying summary: {str(display_error)}")
                    # Fallback: display raw summary
                    st.text_area("Raw Summary", summary, height=300)

                # PDF download functionality with better error handling
                try:
                    if summary and summary.strip():
                        html_content = markdown.markdown(summary)
                        
                        # Add basic HTML structure for better PDF rendering
                        full_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>Document Summary</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 2cm; line-height: 1.6; }}
                                h1, h2, h3 {{ color: #2c3e50; }}
                                p {{ margin-bottom: 1em; }}
                            </style>
                        </head>
                        <body>
                            <h1>Document Summary: {pdf_file.name}</h1>
                            {html_content}
                        </body>
                        </html>
                        """
                        
                        pdf_bytes = HTML(string=full_html).write_pdf()
                        pdf_buffer = io.BytesIO(pdf_bytes)

                        st.download_button(
                            "⬇️ Download Summary as PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=f"{pdf_file.name.rsplit('.', 1)[0]}_summary.pdf",
                            mime="application/pdf",
                            help="Download the summary as a formatted PDF document"
                        )
                    
                except Exception as pdf_error:
                    st.error(f"❌ PDF generation failed: {str(pdf_error)}")
                    # Offer text download as fallback
                    st.download_button(
                        "⬇️ Download Summary as Text",
                        data=summary,
                        file_name=f"{pdf_file.name.rsplit('.', 1)[0]}_summary.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            status_placeholder.error(f"❌ {str(e)}")
            st.session_state.document_processed = False
        
        finally:
            st.session_state.processing = False

# Chat Section with better UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.subheader("💬 Ask Questions About Your Document")

if not st.session_state.document_processed:
    st.info("📝 Process a document first to enable the chat functionality.")
elif not st.session_state.textforbot:
    st.warning("⚠️ No content available for chat. The document processing may have failed.")
else:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

    # Chat input
    user_question = st.chat_input(
        "Ask a question about the document...",
        disabled=not st.session_state.textforbot
    )

    if user_question and user_question.strip():
        # Validate question length
        if len(user_question) > 500:
            st.error("❌ Question too long. Please keep it under 500 characters.")
        else:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_question))
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)

            # Prepare and send request
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        payload = {
                            "messages": st.session_state.textforbot,
                            "question": user_question
                        }
                        
                        response = safe_api_call(
                            "http://localhost:8000/chat/",
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=120  # 2 minutes for chat
                        )
                        
                        result = response.json()
                        backend_answer = result.get("answer", "I don't know")
                        
                        # Validate response
                        if not backend_answer or backend_answer.strip() == "":
                            backend_answer = "I received an empty response. Please try rephrasing your question."
                        
                    except Exception as e:
                        backend_answer = f"⚠️ Chat Error: {str(e)}"

                    # Display and store response
                    assistant_message = f"🤖 {backend_answer}"
                    st.markdown(assistant_message)
                    st.session_state.chat_history.append(("assistant", assistant_message))

st.markdown('</div>', unsafe_allow_html=True)

# Debug information (collapsible, only in development)
if st.checkbox("Show Debug Info", help="For troubleshooting purposes"):
    with st.expander("Debug Information"):
        st.write("**Session State:**")
        st.write(f"- Document processed: {st.session_state.document_processed}")
        st.write(f"- Current document: {st.session_state.current_document_name}")
        st.write(f"- Processing: {st.session_state.processing}")
        st.write(f"- Textforbot chunks: {len(st.session_state.textforbot)}")
        st.write(f"- Chat history length: {len(st.session_state.chat_history)}")
        
        if st.session_state.textforbot:
            st.write("**First few characters of first chunk:**")
            st.code(st.session_state.textforbot[0][:200] + "..." if len(st.session_state.textforbot[0]) > 200 else st.session_state.textforbot[0])
