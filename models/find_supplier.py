from uagents import Model
from typing import Optional, List
from uagents import Context
from typing_extensions import TypedDict

class FindSupplierRequest(Model):
    """Request to find suppliers from UK Companies House or US SEC API"""

    request_id: str
    user_query: str
    business_category: str
    country: str = ""  # US or UK
    timestamp: str = ""


class SupplierSearchResult(Model):
    """Individual supplier search result"""

    company_name: str
    location: str = ""
    industry: str = ""
    country: str = ""
    # UK-specific fields
    company_number: str = ""
    status: str = ""
    company_type: str = ""
    address: str = ""
    # US-specific fields
    ticker: str = ""
    cik: str = ""
    # Legacy field for backwards compatibility
    b_corp_profile_url: str = ""
    description: str = ""


class FindSupplierResponse(Model):
    """Response with suppliers found from UK or US APIs"""

    request_id: str
    success: bool
    suppliers: List[SupplierSearchResult] = []
    country: str = ""
    search_category: str = ""
    total_results_found: int = 0
    search_summary: str = ""
    error_message: str = ""
    timestamp: str = ""
    # Legacy fields for backwards compatibility
    best_supplier: Optional[SupplierSearchResult] = None
    alternative_suppliers: List[SupplierSearchResult] = []


class SupplierSearchState(TypedDict):
    """State for LangGraph workflow"""

    request_id: str
    user_query: str
    business_category: str
    country: Optional[str]
    search_query: str
    suppliers: List[SupplierSearchResult]
    success: bool
    error_message: Optional[str]
    search_summary: str
    score: int
    ctx: Optional[Context]
