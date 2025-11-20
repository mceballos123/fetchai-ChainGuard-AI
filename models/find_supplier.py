from uagents import Model
from typing import Optional, List


class FindSupplierRequest(Model):
    """Request to find a supplier from B Corporation directory"""

    request_id: str
    user_query: str
    business_category: str


class SupplierSearchResult(Model):
    """Individual supplier search result from B Corp directory"""

    company_name: str
    location: str = ""
    industry: str = ""
    b_corp_profile_url: str = ""
    description: str = ""


class FindSupplierResponse(Model):
    """Response with the best supplier found from B Corp directory"""

    request_id: str
    success: bool
    best_supplier: Optional[SupplierSearchResult] = None
    alternative_suppliers: List[SupplierSearchResult] = []
    search_category: str = ""
    total_results_found: int = 0
    search_summary: str = ""
    error_message: str = ""
