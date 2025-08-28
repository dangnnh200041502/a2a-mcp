"""Query Analyzer: LLM-based query analysis and task planning.

This module provides intelligent query analysis capabilities:
- Analyzes user queries using LLM
- Splits queries into meaningful subtasks
- Groups related questions
- Ignores filler phrases
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import os
from langchain_groq import ChatGroq


class QueryAnalyzer:
    """LLM-based query analyzer for intelligent task planning."""
    
    def __init__(self):
        self._llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
        )
    
    def analyze_query(self, raw: str) -> List[str]:
        """Sử dụng LLM để phân tích và tách query thành các subtasks."""
        raw = (raw or "").strip()
        if not raw:
            return []
        
        prompt = f"""
You are a query analyzer. Your task is to break down a user query into separate, independent subtasks.

IMPORTANT RULES:
1. Only split into subtasks if there are MULTIPLE DISTINCT QUESTIONS or REQUESTS
2. Ignore filler phrases like "can I ask you that", "by the way", "also", etc.
3. Each subtask must be a complete, meaningful question or request
4. If there's only one main question, return it as a single subtask
5. Focus on the actual information being requested
6. GROUP RELATED QUESTIONS: If multiple questions are about the same topic/entity, combine them into one subtask

Examples:
Query: "What is 2+2 and what's the weather?"
Subtasks:
What is 2+2?
What's the weather?

Query: "Calculate 10*5 and explain machine learning"
Subtasks:
Calculate 10*5
Explain machine learning

Query: "What is artificial intelligence?"
Subtasks:
What is artificial intelligence?

Query: "When was GreenGrow Innovations founded? and Where is it headquartered?"
Subtasks:
When was GreenGrow Innovations founded and where is it headquartered?

Query: "What is the weather today and what is the result of 98 + 126, can I ask you that When was GreenGrow Innovations founded?"
Subtasks:
What is the weather today?
What is the result of 98 + 126?
When was GreenGrow Innovations founded?

Query: "What is the capital of France and what is its population?"
Subtasks:
What is the capital of France and what is its population?

Query: "Hello, how are you doing today?"
Subtasks:
Hello, how are you doing today?

Now analyze this query: "{raw}"

Subtasks:
"""
        
        try:
            response = self._llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            
            # Parse the response
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Filter out non-subtask lines (like "Subtasks:" header)
            subtasks = []
            for line in lines:
                if line.lower() in ['subtasks:', 'subtask:', 'tasks:', 'task:']:
                    continue
                if line.startswith('Query:') or line.startswith('Now analyze'):
                    continue
                subtasks.append(line)
            
            # If LLM didn't split or returned empty, return original query
            if not subtasks:
                return [raw]
            
            return subtasks
            
        except Exception as e:
            # Fallback to original query if LLM fails
            print(f"LLM analysis failed: {e}")
            return [raw]
    
    def is_multi_intent(self, query: str) -> bool:
        """Check if query contains multiple intents."""
        subtasks = self.analyze_query(query)
        return len(subtasks) > 1
    
    def get_subtasks(self, query: str) -> List[str]:
        """Get subtasks for a query."""
        return self.analyze_query(query)


# Convenience functions for easy integration
def create_analyzer() -> QueryAnalyzer:
    """Create a new query analyzer."""
    return QueryAnalyzer()


def analyze_single_query(query: str) -> List[str]:
    """Quick function to analyze a single query."""
    analyzer = create_analyzer()
    return analyzer.analyze_query(query)
