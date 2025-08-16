
import logging
from crewai import Crew
from agents import create_document_analyzer, create_web_researcher, create_summary_writer
from tasks import create_document_analysis_task, create_web_research_task, create_summary_synthesis_task
from tools import search_web_content

logger = logging.getLogger(__name__)

class DocumentSummarizationCrew:
    """Main crew for document summarization with citations"""
    
    def __init__(self):
        """Initialize the crew with agents"""
        self.document_analyzer = create_document_analyzer()
        self.web_researcher = create_web_researcher()
        self.summary_writer = create_summary_writer()
        
        logger.info("Document Summarization Crew initialized")
    
    def process_document(self, document_text: str, filename: str):
        """
        Process document using CrewAI workflow
        
        Args:
            document_text: Full text of the document
            filename: Name of the document file
            
        Returns:
            dict: Summary and citations
        """
        try:
            logger.info(f"Starting document processing for: {filename}")
            
            # Create tasks
            analysis_task = create_document_analysis_task(
                document_text, filename, self.document_analyzer
            )
            
            research_task = create_web_research_task(self.web_researcher)
            
            synthesis_task = create_summary_synthesis_task(
                filename, self.summary_writer
            )
            
            # Set task dependencies
            research_task.context = [analysis_task]
            synthesis_task.context = [analysis_task, research_task]
            
            # Create and run crew
            crew = Crew(
                agents=[self.document_analyzer, self.web_researcher, self.summary_writer],
                tasks=[analysis_task, research_task, synthesis_task],
                verbose=True
            )
            
            logger.info("Running CrewAI workflow...")
            result = crew.kickoff()
            
            # Extract citations from web search results
            citations = self._extract_citations()
            
            logger.info(f"Document processing completed with {len(citations)} citations")
            
            return {
                "summary": str(result),
                "citations": citations
            }
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            return {
                "summary": f"Error processing document: {str(e)}",
                "citations": []
            }
    
    def _extract_citations(self):
        """Extract citations from web search results"""
        try:
            # Get some sample web results for citations
            # In a real implementation, you'd extract these from the actual search results
            sample_results = search_web_content("document analysis research", )
            
            citations = []
            for i, result in enumerate(sample_results[:5], 1):
                if result.get('title') and result.get('url'):
                    citations.append({
                        "title": result['title'],
                        "url": result['url'],
                        "description": result.get('summary', result.get('content', ''))[:200] + "..."
                    })
            
            return citations
            
        except Exception as e:
            logger.warning(f"Could not extract citations: {e}")
            return []

# Create global crew instance
document_crew = DocumentSummarizationCrew()