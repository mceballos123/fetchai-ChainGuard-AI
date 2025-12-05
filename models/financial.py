from uagents import Model
from pydantic import Field
from typing import List, Optional


class FinancialRequest(Model):
    """Request model for financial risk analysis - country will be scraped from B Corp page"""

    request_id: str
    supplier_name: str
    industry: str
    b_corp_profile_url: Optional[str] = None  # B Corp URL to scrape country from
    user_country: str = "United States"  # User's country (default: US) - extracted from query like "I'm in Germany..."
    timestamp: str = ""


class FinancialResponse(Model):
    """Response model for financial risk analysis"""

    request_id: str
    supplier_name: str
    financial_score: float  # 0-100, higher is better (lower risk)
    financial_details: str = ""  # Summary of tariffs, inflation, financial risks
    risk_factors: List[str] = []  # List of identified financial risks
    user_country: str = "United States"  # User's country used for analysis
    supplier_country: str = ""  # Supplier's country extracted from B Corp
    timestamp: str = ""


"""
Financial Agent Flow:
    B Corp Page --> Web Scraping --> Extract Supplier Country --> Country-Based Financial Analysis --> Response
    
    Process:
    1. Receive b_corp_profile_url and user_country from orchestrator
    2. Scrape B Corp page to extract supplier country from Headquarters section
    3. Apply country-specific financial risk analysis based on trade relationships
       between user_country and supplier_country
    4. Return financial score and details
    
    Example:
    - User query: "I'm in Germany and want to find a coffee supplier"
    - user_country = "Germany"
    - Supplier B Corp page shows "Headquarters: Catalonia, Spain"
    - supplier_country = "Spain"
    - Analysis: Germany-Spain trade relationship (both EU = favorable)
    
    Financial Agent analyzes and returns:
    - Financial score (0-100, threshold: 60)
    - Financial details (country trade relationship, tariff considerations)
    - Risk factors (trade-specific factors based on countries)
    
    Financial Response --> Orchestrator --> Combined with Compliance & Risk --> User
"""
