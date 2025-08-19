import os
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Cấu hình Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def expand_query(original_query: str) -> List[str]:
    """
    Tạo 3 query tương tự từ query gốc sử dụng Gemini 1.5 Flash
    """
    prompt = f"""
    ### ROLE:
    You are an AI assistant specialized in query optimization and semantic analysis.

    ### TASKS:
    - From the original query, generate **three** new search queries.  
    - Preserve the main idea while varying the expression and enriching with domain-specific keywords.
    - Ensure a semantic similarity of at least 85%.

    ### GUIDELINES:  
    - Retain the original subject, predicate, verbs, and key adjectives.  
    - You may reorder sentence components and add extra terms or phrases to enrich content without altering the original meaning.  
    - Return three distinct queries, each on its own line.

    ### INPUT:
    Original query: "{original_query}"

    ### OUTPUT:
    Generate exactly 3 queries, one per line:
    """

    try:
        response = model.generate_content(prompt)
        # Tách response thành các dòng và loại bỏ dòng trống
        expanded_queries = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        
        # Đảm bảo có đúng 3 queries
        if len(expanded_queries) >= 3:
            return expanded_queries[:3]
        elif len(expanded_queries) > 0:
            # Nếu không đủ 3, lặp lại query cuối
            while len(expanded_queries) < 3:
                expanded_queries.append(expanded_queries[-1])
            return expanded_queries
        else:
            # Fallback nếu không có response
            return [original_query] * 3
            
    except Exception as e:
        print(f"Error in query expansion: {e}")
        # Fallback: trả về query gốc 3 lần
        return [original_query] * 3

def get_all_queries(original_query: str) -> List[str]:
    """
    Trả về danh sách tất cả queries (gốc + 3 expanded)
    """
    expanded_queries = expand_query(original_query)
    all_queries = [original_query] + expanded_queries
    return all_queries

 
