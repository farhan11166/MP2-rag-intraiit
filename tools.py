
import logging
from typing import List, Dict, Any
from crewai.tools import tool
from config import exa_client

logger = logging.getLogger(__name__)

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