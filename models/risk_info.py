from uagents import Model
from typing import Optional, List
from uagents import Context
from typing_extensions import TypedDict


class RiskInfoRequest(Model):
    """Request to get risk factors information about a specific supplier"""

    request_id: str
    company_name: str
    cik: str  # SEC CIK number
    country: str  # US or UK
    ticker: str = ""  # Optional ticker symbol
    timestamp: str = ""


class RiskData(Model):
    """Risk factors extracted from SEC filings"""

    company_name: str
    cik: str
    filing_type: str  # "10-Q" for US companies, "20-F" for foreign
    filing_date: str = ""
    is_foreign: bool = False

    # Risk information
    summary_risk_factors: str = ""  # Summary of key risk factors
    operational_risks: str = ""  # Operational and business risks
    financial_risks: str = ""  # Financial and market risks
    legal_regulatory_risks: str = ""  # Legal and regulatory risks
    all_risk_factors: List[str] = []  # List of individual risk factors

    # Metadata
    filing_url: str = ""
    extraction_summary: str = ""


class RiskInfoResponse(Model):
    """Response with detailed risk information"""

    request_id: str
    success: bool
    risk_data: Optional[RiskData] = None
    error_message: str = ""
    timestamp: str = ""


class RiskInfoState(TypedDict):
    """State for LangGraph workflow for risk info extraction"""

    request_id: str
    company_name: str
    cik: str
    country: str
    ticker: str
    is_foreign: Optional[bool]
    filing_type: Optional[str]
    latest_filing_url: Optional[str]
    filing_date: Optional[str]

    # Extracted risk data
    summary_risk_factors: Optional[str]
    operational_risks: Optional[str]
    financial_risks: Optional[str]
    legal_regulatory_risks: Optional[str]
    all_risk_factors: Optional[List[str]]

    success: bool
    error_message: Optional[str]
    score: int
    ctx: Optional[Context]
