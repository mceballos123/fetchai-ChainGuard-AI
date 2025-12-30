from uagents import Model
from typing import Optional
from uagents import Context
from typing_extensions import TypedDict


class FinanceInfoRequest(Model):
    """Request to get financial information about a specific supplier"""

    request_id: str
    company_name: str
    cik: str  # SEC CIK number
    country: str  # US or UK (for now only US)
    ticker: str = ""  # Optional ticker symbol
    timestamp: str = ""


class FinancialData(Model):
    """Financial information extracted from SEC filings"""

    company_name: str
    cik: str
    filing_type: str  # "10-Q" for US companies
    filing_date: str = ""

    # Key financial metrics
    overview: str = ""  # Company overview and business description
    financial_condition: str = ""  # Management's discussion of financial condition
    revenue_info: str = ""  # Revenue and growth information
    legal_proceedings: str = ""  # Legal and regulatory issues

    # Metadata
    filing_url: str = ""
    extraction_summary: str = ""


class FinanceInfoResponse(Model):
    """Response with detailed financial information"""

    request_id: str
    success: bool
    financial_data: Optional[FinancialData] = None
    error_message: str = ""
    timestamp: str = ""


class FinanceInfoState(TypedDict):
    """State for LangGraph workflow for financial info extraction"""

    request_id: str
    company_name: str
    cik: str
    country: str
    ticker: str
    filing_type: Optional[str]
    latest_filing_url: Optional[str]
    filing_date: Optional[str]

    # Extracted financial data
    overview: Optional[str]
    financial_condition: Optional[str]
    revenue_info: Optional[str]
    legal_proceedings: Optional[str]

    success: bool
    error_message: Optional[str]
    score: int
    ctx: Optional[Context]
