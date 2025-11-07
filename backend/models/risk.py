from uagents import Model
from typing import List, Optional


class RiskRequest(Model):
    """Request model for risk management analysis"""

    request_id: str 
    supplier_name: str
    industry: str
    timestamp: str = ""

class RiskResponse(Model):
    """Response model for risk management analysis"""

    request_id: str
    supplier_name: str
    risk_score: float
    risk_details: str
    risk_factors: Optional[List[str]]
    timestamp: str = ""