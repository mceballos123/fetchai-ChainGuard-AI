from uagents import Model
from pydantic import Field
from typing import List


class FinancialRequest(Model):
    """Request model for financial risk analysis based on country tariff/inflation data"""

    request_id: str
    supplier_name: str
    supplier_country: str  # Country where supplier operates - used for Trade War Tracker lookup
    industry: str
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
    Trade War Tracker URL --> Web Scraping --> Country Tariff/Inflation Data --> Pinecone --> RAG Analysis
    
    Process:
    1. Receive supplier_country from find_supplier_agent
    2. Scrape Trade War Tracker (tradewartracker.com) for country-specific data
    3. Extract tariff rates, inflation info, trade restrictions
    4. Store in Pinecone with embeddings
    5. RAG analysis with LLM
    
    Financial Agent analyzes and returns:
    - Financial score (0-100, threshold: 60)
    - Financial details (tariffs, inflation impact)
    - Risk factors (specific tariff percentages, trade restrictions)
    
    Financial Response --> Orchestrator --> Combined with Compliance & Risk --> User (ASI:1)
"""
