from uagents import Model
from typing import Optional
from uagents import Context
from typing_extensions import TypedDict


class SupplierInfoRequest(Model):
    """Request to get detailed information about a specific supplier"""

    request_id: str
    company_name: str
    cik: str  # SEC CIK number
    country: str  # US or UK
    ticker: str = ""  # Optional ticker symbol
    timestamp: str = ""


class SupplierInfoResult(Model):
    """Detailed information about a supplier"""

    company_name: str
    cik: str
    country: str
    is_foreign: bool  # True if foreign company, False if US company
    filing_type: str  # "10-Q" for US companies, "20-F" for foreign companies
    company_description: str  # Summary of what the company does
    history: str = ""  # Company history and development
    latest_filing_url: str = ""  # URL to the latest filing
    error_message: str = ""


class SupplierInfoResponse(Model):
    """Response with detailed supplier information"""

    request_id: str
    success: bool
    supplier_info: Optional[SupplierInfoResult] = None
    error_message: str = ""
    timestamp: str = ""


class SupplierInfoState(TypedDict):
    """State for LangGraph workflow for supplier info extraction"""

    request_id: str
    company_name: str
    cik: str
    country: str
    ticker: str
    is_foreign: Optional[bool]
    filing_type: Optional[str]
    latest_filing_url: Optional[str]
    filing_content: Optional[str]
    company_description: Optional[str]
    history: Optional[str]
    success: bool
    error_message: Optional[str]
    score: int
    ctx: Optional[Context]
