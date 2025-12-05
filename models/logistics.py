from uagents import Model
from typing import List, Optional


class LogisticsRequest(Model):
    """Request model for logistics/inventory monitoring"""

    request_id: str
    supplier_name: str
    product_category: str = ""  # food, clothing, electronics - determined from user query
    timestamp: str = ""


class LogisticsResponse(Model):
    """Response model for logistics/inventory monitoring"""

    request_id: str
    supplier_name: str
    current_inventory_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    current_inventory_units: int
    inventory_details: str
    predicted_inventory_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    predicted_inventory_units: int
    inventory_forecast: str
    restocking_status: str  # ON_TIME, DELAYED, CRITICAL
    restocking_details: str
    overall_logistics_status: str  # HEALTHY, WARNING, CRITICAL
    alerts: List[str] = []
    timestamp: str = ""

