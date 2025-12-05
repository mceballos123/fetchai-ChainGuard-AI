from uagents import Model
from pydantic import Field
from typing import List, Optional


class FinancialRequest(Model):
    """Request model for financial risk analysis - country will be scraped from B Corp page"""

    request_id: str
    supplier_name: str
    industry: str
    b_corp_profile_url: Optional[str] = None  # B Corp URL to scrape country from
    timestamp: str = ""


class FinancialResponse(Model):
    """Response model for financial risk analysis"""

    request_id: str
    supplier_name: str
    financial_score: float  # 0-100, higher is better (lower risk)
    financial_details: str = ""  # Summary of tariffs, inflation, financial risks
    risk_factors: List[str] = []  # List of identified financial risks
    timestamp: str = ""


"""
Financial Agent Flow:
    B Corp Page --> Web Scraping --> Extract Country --> Country-Based Financial Analysis --> Pinecone --> RAG Analysis
    
    Process:
    1. Receive b_corp_profile_url from orchestrator
    2. Scrape B Corp page to extract supplier country
    3. Apply country-specific financial risk analysis based on US trade relationships
    4. Store analysis in Pinecone with embeddings
    5. RAG analysis with LLM
    
    Financial Agent analyzes and returns:
    - Financial score (0-100, threshold: 60)
    - Financial details (country trade relationship, tariff considerations)
    - Risk factors (trade-specific factors based on country)
    
    Financial Response --> Orchestrator --> Combined with Compliance & Risk --> User (ASI:1)
"""
