"""
CrewAI Tasks for document processing workflow
"""
from crewai import Task

def create_document_analysis_task(document_text: str, filename: str, agent):
    """Create document analysis task"""
    return Task(
        description=f"""
        Analyze the following document and provide a comprehensive analysis:
        
        **Document:** {filename}
        **Content:** {document_text[:3000]}...
        
        Please analyze and extract:
        1. **Main Purpose & Objectives** - What is this document about?
        2. **Key Findings & Results** - What are the main discoveries or outcomes?
        3. **Methodology & Approach** - What methods or techniques were used?
        4. **Important Concepts** - What are the key terms and concepts?
        5. **Technical Details** - Any important technical information
        6. **Conclusions** - What are the main takeaways?
        
        Also extract 5-7 key topics that would be good for web research to find 
        related sources and citations.
        
        Provide a detailed analysis that will help create a comprehensive summary.
        """,
        agent=agent,
        expected_output="Detailed document analysis with key topics for web research"
    )

def create_web_research_task(agent):
    """Create web research task"""
    return Task(
        description=f"""
        Based on the document analysis, search for relevant online sources that:
        
        1. **Support the main findings** - Find sources that validate or provide context
        2. **Explain methodologies** - Find sources about the methods used
        3. **Provide background** - Find sources that give context to the topic
        4. **Show recent developments** - Find current research in the field
        5. **Offer additional insights** - Find complementary information
        
        For each source you find, make sure it's:
        - From a credible, authoritative source
        - Relevant to the document's content
        - Recent and up-to-date when possible
        - Adds value to understanding the document
        
        Focus on finding 3-5 high-quality sources rather than many low-quality ones.
        """,
        agent=agent,
        expected_output="List of relevant, credible web sources with descriptions"
    )

def create_summary_synthesis_task(filename: str, agent):
    """Create summary synthesis task"""
    return Task(
        description=f"""
        Create a comprehensive summary that integrates the document analysis 
        with the web research findings.
        
        **Requirements:**
        1. Use the document analysis as the primary content foundation
        2. Integrate web sources to support and expand key points
        3. Structure using clear Markdown formatting
        4. Include proper citations and references
        5. Make it detailed, informative, and well-organized
        
        **Structure the summary as follows:**
        
        # Document Summary: {filename}
        
        ## Executive Summary
        Brief overview of the document's main points and significance
        
        ## Key Findings and Results  
        Main discoveries, outcomes, and important results
        
        ## Methodology and Approach
        Methods, techniques, and approaches used in the document
        
        ## Technical Details and Implementation
        Important technical information, details, and specifics
        
        ## Related Research and Context
        How this relates to current research (supported by web citations)
        
        ## Conclusions and Implications
        Final thoughts, implications, and future considerations
        
        ## References and Citations
        List of web sources and citations used
        
        **Citation Format:** 
        - Use numbered references like [1], [2], etc.
        - Include full citation details at the end
        - Make sure citations are relevant and add value
        """,
        agent=agent,
        expected_output="Comprehensive markdown summary with integrated web citations"
    )