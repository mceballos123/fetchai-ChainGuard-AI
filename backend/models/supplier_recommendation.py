from uagents import Model
from pydantic import Field
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC
from enum import Enum

class SupplierRecommendation(Model):
    """Single supplier recommendation with scores"""
    supplier_name: str = Field(..., description="Name of the supplier")
    supplier_location: str = Field(..., description="Supplier's location")
    overall_score: float = Field(..., description="Overall compatibility score 0-100", ge=0, le=100)
    compliance_score: float = Field(..., description="Compliance score", ge=0, le=100)
    financial_score: float = Field(..., description="Financial health score", ge=0, le=100)
    risk_score: float = Field(..., description="Risk score (inverted for display)", ge=0, le=100)
    why_this_supplier: str = Field(..., description="Why this supplier is recommended")

class SupplierSearchResponse(Model):
    """Final response with supplier recommendations"""
    request_id: str = Field(..., description="Original request ID")
    timestamp: str = Field(default="", description="Response timestamp")
    top_supplier: Optional[SupplierRecommendation] = Field(None, description="Top recommended supplier")
    alternative_suppliers: List[SupplierRecommendation] = Field(
        default_factory=list, 
        description="Alternative supplier options"
    )
    search_summary: str = Field(..., description="Summary of search results")
    
    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)

class ErrorResponse(Model):
    """Error response model"""
    request_id: str = Field(..., description="Original request ID")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: str = Field(default="", description="Error timestamp")
    
    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)
    
"""
    Prompt message for the supplier recommendation agent to recommend the supplier:
    Documents --> RAG + Llama index  --> Supplier Recommendation

    RAG + Llama index sorts the information from the documents and passes the important information to the supplier recommendation agent

    The supplier recommendation agent analyzes the information and passes the information like the supplier name, location, overall score, compliance score, financial score, risk score and why this supplier is recommended

    Supplier Recommendation --> passes the information --> find supplier agent --> passes the info to the user(if it goes through)

"""