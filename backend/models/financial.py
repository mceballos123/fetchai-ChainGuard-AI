from uagents import Model
from pydantic import Field
from typing import List


class FinancialRequest(Model):
    """Request model for financial risk analysis"""

    request_id: str
    supplier_name: str
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
    Documents (financial_files/) --> RAG + LlamaIndex --> Financial Analysis
    
    RAG + LlamaIndex extracts:
    - Tariff information
    - Inflation risks
    - Currency exchange risks
    - Economic stability data
    - Trade restrictions
    
    Financial Agent analyzes and returns:
    - Financial score (0-100)
    - Financial details summary
    - Risk factors list
    
    Financial Response --> Orchestrator --> Combined with Compliance --> User (ASI:1)
"""
