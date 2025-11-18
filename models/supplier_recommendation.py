from uagents import Model
from typing import List


class SupplierRecommendation(Model):
    """Single supplier recommendation with scores"""

    supplier_name: str
    supplier_location: str
    overall_score: float
    compliance_score: float
    financial_score: float
    risk_score: float
    why_this_supplier: str


class SupplierSearchResponse(Model):
    """Final response with supplier recommendations"""

    request_id: str
    timestamp: str = ""
    top_supplier: SupplierRecommendation
    alternative_suppliers: List[SupplierRecommendation] = []
    search_summary: str


class ErrorResponse(Model):
    """Error response model"""

    request_id: str
    error: str
    error_type: str
    timestamp: str = ""


"""
    Prompt message for the supplier recommendation agent to recommend the supplier:
    Documents --> RAG + Llama index  --> Supplier Recommendation

    RAG + Llama index sorts the information from the documents and passes the important information to the supplier recommendation agent

    The supplier recommendation agent analyzes the information and passes the information like the supplier name, location, overall score, compliance score, financial score, risk score and why this supplier is recommended

    Supplier Recommendation --> passes the information --> find supplier agent --> passes the info to the user(if it goes through)

"""
