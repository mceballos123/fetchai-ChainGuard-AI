from uagents import Model
from pydantic import Field
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC
from enum import Enum

class SupplierRiskRequest(Model):
    request_id: str = Field(..., description="Original request ID")
    supplier_name: str = Field(..., description="Name of supplier to assess")
    supplier_location: str = Field(..., description="Supplier's geographic location")
    industry: str = Field(..., description="Industry sector")
    timestamp: str = Field(default="", description="Request timestamp")

class SupplierRiskResponse(Model):
    request_id: str = Field(..., description="Original request ID")
    supplier_name: str = Field(..., description="Supplier assessed")
    risk_score: float = Field(..., description="Overall risk score 0-100 (lower is better)", ge=0, le=100)
    weather_risks: Optional[List[str]] = Field(default_factory=list, description="Weather-related risks")
    natural_disaster_risks: Optional[List[str]] = Field(default_factory=list, description="Natural disaster risks")
    capacity_constraints: str = Field(..., description="Capacity and operational constraints")
    geographic_risks: str = Field(..., description="Geographic and geopolitical risks")
    timestamp: str = Field(default="", description="Response timestamp")
    
    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)