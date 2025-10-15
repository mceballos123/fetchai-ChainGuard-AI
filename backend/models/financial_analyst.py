from uagents import Model
from pydantic import Field
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC
from enum import Enum

class FinancialAnalysisRequest(Model):
    request_id: str = Field(..., description="Original request ID")
    supplier_name: str = Field(..., description="Name of supplier to analyze")
    industry: str = Field(..., description="Industry sector")
    timestamp: str = Field(default="", description="Request timestamp")

class FinancialAnalysisResponse(Model):
    request_id: str = Field(..., description="Original request ID")
    supplier_name: str = Field(..., description="Supplier analyzed")
    inflation_impact: str = Field(..., description="Inflation and tariff impact analysis")
    risk_factors: List[str] = Field(default_factory=list, description="Financial risk factors")
    timestamp: str = Field(default="", description="Response timestamp")
    supplier_finacial_score: float = Field(..., description="Supplier financial score 0-100", ge=0, le=100)
    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)

"""
    Prompt message for the financial analyst agent to analyze the financial health of the supplier through the supplier's fincial statement:

    Documents --> RAG + Llama index  --> Financial Analyst

    RAG + Llama index sorts the information from the documents and passes the important information to the financial analyst agent
    
    The financial analyst agent analyzes the information and passes the information like the risk factors that could come up and the inflation impact of the supplier's country and the financial health of the supplier

    Financial Analyst --> passes the information --> find supplier agent --> passes the info to the user(if it goes through)
"""