from uagents import Model
from pydantic import Field
from typing import Optional


class FindSupplierRequest(Model):
    """Request to find a supplier from B Corporation directory"""

    request_id: str = Field(..., description="Unique request identifier")
    user_query: str = Field(
        ..., description="User's search query describing what they're looking for"
    )
    business_category: str = Field(
        ..., description="Business category to search (e.g., coffee, pizza, food)"
    )
    timestamp: str = Field(default="", description="Request timestamp")


class SupplierSearchResult(Model):
    """Individual supplier search result from B Corp directory"""

    company_name: str = Field(..., description="Name of the B Corporation company")
    location: str = Field(default="", description="Company location")
    industry: str = Field(default="", description="Industry/business category")
    b_corp_profile_url: str = Field(default="", description="URL to B Corp profile")
    description: str = Field(default="", description="Brief description of the company")


class FindSupplierResponse(Model):
    """Response with the best supplier found from B Corp directory"""

    request_id: str = Field(..., description="Original request identifier")
    success: bool = Field(..., description="Whether search was successful")
    best_supplier: Optional[SupplierSearchResult] = Field(
        default=None, description="Top recommended supplier"
    )
    search_category: str = Field(default="", description="Category that was searched")
    total_results_found: int = Field(
        default=0, description="Total number of results found"
    )
    search_summary: str = Field(default="", description="Summary of the search results")
    timestamp: str = Field(default="", description="Response timestamp")
    error_message: str = Field(default="", description="Error message if search failed")
