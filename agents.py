# agents.py
import logging
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from typing import Dict, Any, List
import traceback
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from exa_py import Exa


from llm_setup import crew_llm, chat_model, basic_summarization
from helpers import extract_key_topics

logger = logging.getLogger(__name__)

load_dotenv("a.env")
api_key = os.getenv("GOOGLE_API_KEY")
exa_api_key = os.getenv("EXA_API_KEY")
exa_client = Exa(api_key=exa_api_key)
@tool("exa_search")
def search_web_content(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for relevant content using Exa API.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
    
    Returns:
        List[Dict]: List of search results with title, URL, and content
    """
    try:
        logger.info(f"Searching web for: {query}")
        search_response = exa_client.search_and_contents(
            query=query,
            num_results=num_results,
            text=True,
            highlights=True,
            summary=True
        )
        
        results = []
        for result in search_response.results:
            results.append({
                "title": result.title or "Untitled",
                "url": result.url or "",
                "content": result.text[:800] if result.text else "",
                "summary": getattr(result, 'summary', '') or "",
                "highlights": getattr(result, 'highlights', []) or []
            })
        
        logger.info(f"Found {len(results)} web results")
        return results
    
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return []

def create_agents():
    """Create CrewAI agents with error handling"""
    try:
        if crew_llm is None:
            logger.error("CrewAI LLM not initialized")
            return None, None, None
            
        # Document Analysis Agent
        document_analyzer = Agent(
            role="Document Analyst",
            goal="Analyze and extract key information from documents for comprehensive summarization",
            backstory="""You are an expert document analyst with years of experience in technical 
            document review. You excel at identifying key concepts, methodologies, and findings 
            from complex documents.""",
            verbose=False,
            allow_delegation=False,
            llm=crew_llm
        )

        # Web Research Agent
        web_researcher = Agent(
            role="Web Research Specialist",
            goal="Find relevant online sources and citations that complement document analysis",
            backstory="""You are a skilled researcher who knows how to find credible online sources 
            that provide additional context and validation for technical documents. You excel at 
            identifying authoritative sources and relevant citations.""",
            verbose=False,
            allow_delegation=False,
            tools=[search_web_content],
            llm=crew_llm
        )

        # Summary Synthesizer Agent
        summary_synthesizer = Agent(
            role="Technical Writer and Synthesizer",
            goal="Create comprehensive summaries that integrate document content with web citations",
            backstory="""You are an expert technical writer who specializes in creating detailed, 
            well-structured summaries that combine multiple sources of information. You excel at 
            presenting complex information in a clear, organized manner with proper citations.""",
            verbose=False,
            allow_delegation=False,
            llm=crew_llm
        )
        
        return document_analyzer, web_researcher, summary_synthesizer
    except Exception as e:
        logger.error(f"Error creating agents: {e}")
        return None, None, None


async def enhanced_summarization_with_citations(full_text: str, filename: str) -> Dict[str, Any]:
    """Enhanced summarization using CrewAI agents with web citations"""
    try:
        logger.info("Starting enhanced summarization with citations...")
        
        # Create agents
        document_analyzer, web_researcher, summary_synthesizer = create_agents()
        
        if not all([document_analyzer, web_researcher, summary_synthesizer]):
            logger.error("Failed to create agents, falling back to basic summarization")
            return {
                "summary": await basic_summarization(full_text[:5000]),
                "citations": []
            }
        
        # Extract key topics for web research
        key_topics = extract_key_topics(full_text)
        logger.info(f"Extracted key topics: {key_topics}")
        
        # Task 1: Document Analysis
        document_analysis_task = Task(
            description=f"""
            Analyze the following document and extract key information:
            
            Document: {filename}
            Content (first 3000 chars): {full_text[:3000]}
            
            Focus on:
            1. Main objectives and purpose
            2. Key findings and results
            3. Methodologies and approaches used
            4. Important concepts and terminology
            5. Conclusions and implications
            
            Provide a comprehensive analysis of the document's content and structure.
            """,
            agent=document_analyzer,
            expected_output="Detailed analysis of document content, methodology, and key findings"
        )
        
        # Task 2: Web Research
        web_research_task = Task(
            description=f"""
            Research relevant information online for the following topics derived from the document:
            
            Topics: {', '.join(key_topics)}
            
            For each topic, find:
            1. Authoritative sources and recent research
            2. Related methodologies and best practices
            3. Current developments in the field
            4. Supporting evidence and validation
            
            Focus on finding credible, academic, and industry sources that complement the document.
            """,
            agent=web_researcher,
            expected_output="List of relevant web sources with titles, URLs, and descriptions"
        )
        
        # Task 3: Summary Synthesis
        synthesis_task = Task(
            description=f"""
            Create a comprehensive summary that integrates the document analysis with web research findings.
            
            Requirements:
            1. Start with the document analysis as the primary content
            2. Integrate relevant web citations to support and expand on key points
            3. Structure the summary using clear Markdown formatting
            4. Include proper citations and references
            5. Make it detailed, informative, and well-organized
            
            Structure the summary as follows:
            # Document Summary: {filename}
            
            ## Executive Summary
            [Brief overview of the document's main points]
            
            ## Key Findings and Results
            [Main discoveries and outcomes]
            
            ## Methodology and Approach
            [Methods and techniques used]
            
            ## Technical Details and Implementation
            [Technical aspects and details]
            
            ## Related Research and Context
            [How this relates to current research, supported by web citations]
            
            ## Conclusions and Implications
            [Final thoughts and implications]
            
            ## References and Citations
            [List of web sources and citations]
            """,
            agent=summary_synthesizer,
            expected_output="Comprehensive markdown summary with integrated web citations",
            context=[document_analysis_task, web_research_task]
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[document_analyzer, web_researcher, summary_synthesizer],
            tasks=[document_analysis_task, web_research_task, synthesis_task],
            verbose=True
        )
        
        logger.info("Running CrewAI agents for enhanced summarization...")
        result = crew.kickoff()
        
        # Extract citations from web research results
        citations = []
        try:
            # Get web results from the search tool calls
            web_results = search_web_content(" ".join(key_topics[:2]), num_results=5)
            for idx, result in enumerate(web_results, 1):
                if result.get('title') and result.get('url'):
                    citations.append({
                        "title": result['title'],
                        "url": result['url'],
                        "description": result.get('summary', result.get('content', '')[:200] + "...")
                    })
        except Exception as e:
            logger.warning(f"Could not extract citations: {e}")
        
        logger.info(f"Enhanced summarization completed with {len(citations)} citations")
        
        return {
            "summary": str(result),
            "citations": citations
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced summarization: {e}")
        logger.error(traceback.format_exc())
        # Fallback to basic summarization
        return {
            "summary": await basic_summarization(full_text[:5000]),
            "citations": []
        }
