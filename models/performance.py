from uagents import Model
from typing import List, Optional


class PerformanceRequest(Model):
    """Request model for performance monitoring analysis"""

    request_id: str
    supplier_name: str
    product_category: str = ""  # food, clothing, electronics - determined from user query
    timestamp: str = ""


class PerformanceResponse(Model):
    """Response model for performance monitoring analysis"""

    request_id: str
    supplier_name: str
    weather_status: str
    weather_details: str
    strike_status: str
    strike_details: str
    political_status: str
    political_details: str
    legal_status: str
    legal_details: str
    overall_risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    alerts: List[str] = []
    timestamp: str = ""
