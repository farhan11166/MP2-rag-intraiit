
import logging
from typing import List, Dict, Any
from crewai import Agent, Task, Crew
from config import crew_llm
from tools import search_web_content

logger = logging.getLogger(__name__)

def create_agents():
    """Create all CrewAI agents."""
    if crew_llm is None:
        logger.error("CrewAI LLM not initialized")
        return None, None, None
    
    # Document Analysis Agent
    document_analyzer = Agent(
        role="Document Analyst",
        goal="Analyze and extract key information from documents",
        backstory="""You are an expert document analyst who excels at identifying 
        key concepts, methodologies, and findings from complex documents.""",
        verbose=False,
        allow_delegation=False,
        llm=crew_llm
    )

    # Web Research Agent
    web_researcher = Agent(
        role="Web Research Specialist",
        goal="Find relevant online sources and citations",
        backstory="""You are a skilled researcher who finds credible online sources 
        that provide additional context for technical documents.""",
        verbose=False,
        allow_delegation=False,
        tools=[search_web_content],
        llm=crew_llm
    )

    # Summary Synthesizer Agent
    summary_synthesizer = Agent(
        role="Technical Writer",
        goal="Create comprehensive summaries with web citations",
        backstory="""You are an expert technical writer who creates detailed, 
        well-structured summaries with proper citations.""",
        verbose=False,
        allow_delegation=False,
        llm=crew_llm
    )
    
    return document_analyzer, web_researcher, summary_synthesizer

async def run_summarization_crew(full_text: str, filename: str, key_topics: List[str]) -> str:
    """Run the summarization crew and return result."""
    try:
        # Create agents
        document_analyzer, web_researcher, summary_synthesizer = create_agents()
        
        if not all([document_analyzer, web_researcher, summary_synthesizer]):
            raise ValueError("Failed to create agents")
        
        # Task 1: Document Analysis
        document_task = Task(
            description=f"""
            Analyze this document and extract key information:
            
            Document: {filename}
            Content: {full_text[:3000]}
            
            Focus on: objectives, findings, methodology, concepts, conclusions.
            """,
            agent=document_analyzer,
            expected_output="Detailed analysis of document content and findings"
        )
        
        # Task 2: Web Research
        research_task = Task(
            description=f"""
            Research online for topics: {', '.join(key_topics)}
            
            Find authoritative sources, recent research, and supporting evidence.
            """,
            agent=web_researcher,
            expected_output="List of relevant web sources with descriptions"
        )
        
        # Task 3: Summary Synthesis
        synthesis_task = Task(
            description=f"""
            Create a comprehensive summary integrating document analysis with web research.
            
            Structure as:
            # Document Summary: {filename}
            ## Executive Summary
            ## Key Findings and Results
            ## Methodology and Approach
            ## Related Research and Context
            ## Conclusions and Implications
            ## References and Citations
            """,
            agent=summary_synthesizer,
            expected_output="Comprehensive markdown summary with citations",
            context=[document_task, research_task]
        )
        
        # Create and run crew
        crew = Crew(
            agents=[document_analyzer, web_researcher, summary_synthesizer],
            tasks=[document_task, research_task, synthesis_task],
            verbose=True
        )
        
        logger.info("Running CrewAI summarization...")
        result = crew.kickoff()
        return str(result)
        
    except Exception as e:
        logger.error(f"Error in crew summarization: {e}")
        raise

def get_citations_from_topics(key_topics: List[str]) -> List[Dict[str, str]]:
    """Get web citations for topics."""
    citations = []
    try:
        web_results = search_web_content(" ".join(key_topics[:2]), num_results=5)
        for result in web_results:
            if result.get('title') and result.get('url'):
                citations.append({
                    "title": result['title'],
                    "url": result['url'],
                    "description": result.get('summary', result.get('content', '')[:200] + "...")
                })
    except Exception as e:
        logger.warning(f"Could not get citations: {e}")
    
    return citations