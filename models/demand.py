from uagents import Model
from typing import List


class DemandRequest(Model):
    """Request model for demand forecast analysis"""

    request_id: str
    supplier_name: str
    timestamp: str = ""


class DemandResponse(Model):
    """Response model for demand forecast analysis"""

    request_id: str
    supplier_name: str
    delay_status: str
    delay_details: str
    shipping_status: str
    shipping_details: str
    quality_status: str
    quality_details: str
    overall_performance: str  # EXCELLENT, GOOD, FAIR, POOR
    alerts: List[str] = []
    timestamp: str = ""
