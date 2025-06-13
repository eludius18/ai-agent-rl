# src/rag/web_rag.py

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.agents.tools import Tool
from typing import List

def get_web_search_tool() -> List[Tool]:
    search = DuckDuckGoSearchRun()
    return [
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful to search recent news and current events from the web"
        )
    ]