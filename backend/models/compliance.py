from uagents import Model
from pydantic import Field
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC
from enum import Enum

class ComplianceRequest(Model):
    request_id: str = Field(..., description="Original request ID")
    supplier_name: str = Field(..., description="Name of supplier to check")
    industry: str = Field(..., description="Industry sector")
    company_values: str = Field(..., description="Company values to validate against")
    timestamp: str = Field(default="", description="Request timestamp")

class ComplianceResponse(Model):
    request_id: str = Field(..., description="Original request ID")
    supplier_name: str = Field(..., description="Supplier checked")
    compliance_score: float = Field(..., description="Compliance score 0-100 based on ethics and sustainability", ge=0, le=100)
    violations: Optional[List[str]] = Field(default_factory=list, description="List of violations found")
    sustainability_info: Optional[str] = Field(None, description="Sustainability practices summary")
    ethics_info: Optional[str] = Field(None, description="Ethics practices summary")
    timestamp: str = Field(default="", description="Response timestamp")
    
    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)
    
"""
    Prompt message for the compliance agent to check the compliance of the supplier:
    Documents --> RAG + Llama index  --> Compliance

    RAG + Llama index sorts the information from the documents and passes the important information to the compliance agent

    The compliance agent analyzes the information and passes the information like the compliance score and the violations found

    Compliance --> passes the information --> find supplier agent --> passes the info to the user(if it goes through)

"""