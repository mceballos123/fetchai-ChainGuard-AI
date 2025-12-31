"""
LangGraph Workflow for Multi-Country Supplier Search
Searches for suppliers using US SEC API or UK Companies House API based on user location
"""

import os
import time
import requests
from typing import Optional, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from models.find_supplier import SupplierSearchResult, SupplierSearchState

load_dotenv()


# ========== Helper Functions ==========

def detect_country_from_query(message: str) -> Optional[str]:
    """
    Detect country from user query

    Args:
        message: User query string

    Returns:
        Country code ('US' or 'UK') or None if not detected
    """
    message_lower = message.lower()

    # Detect country
    if any(word in message_lower for word in ["united states", "usa", "us", "america", "american"]):
        return "US"
    elif any(word in message_lower for word in ["uk", "united kingdom", "britain", "british", "england"]):
        return "UK"

    return None


# ========== LangGraph Node Functions ==========

def parse_input_node(state: SupplierSearchState) -> SupplierSearchState:
    """
    Node 1: Parse user input to extract country and build search query
    """
    ctx = state["ctx"]
    user_query = state["user_query"]
    business_category = state["business_category"]

    ctx.logger.info(f"[LangGraph Node: parse_input] Parsing user query: {user_query}")

    # Detect country from query
    country = detect_country_from_query(user_query)

    if not country:
        ctx.logger.warning("Could not detect country from query")
        return {
            **state,
            "country": None,
            "success": False,
            "error_message": "Could not detect country. Please specify 'US' or 'UK' in your query.",
        }

    # Build search query based on business category
    search_query = f"{business_category} supplier" if business_category != "general" else "supplier"

    ctx.logger.info(f"Detected country: {country}")
    ctx.logger.info(f"Search query: {search_query}")

    return {
        **state,
        "country": country,
        "search_query": search_query,
    }


def search_uk_suppliers_node(state: SupplierSearchState) -> SupplierSearchState:
    """
    Node 2a: Search UK Companies House API for suppliers
    """
    ctx = state["ctx"]
    search_query = state["search_query"]

    ctx.logger.info(f"[LangGraph Node: search_uk] Searching UK Companies House API")

    api_key = os.getenv('GOV_UK_API_KEY', '').strip()

    if not api_key or api_key == "your_api_key_here":
        return {
            **state,
            "success": False,
            "error_message": "GOV_UK_API_KEY not configured in .env file",
        }

    base_url = "https://api.company-information.service.gov.uk/search/companies"
    params = {
        "q": search_query,
        "items_per_page": 4
    }

    try:
        # Add 10-second delay before making API call to respect rate limits
        ctx.logger.info("Waiting 10 seconds before calling UK API...")
        time.sleep(10)
        ctx.logger.info("Making UK API request now...")

        response = requests.get(
            base_url,
            params=params,
            auth=(api_key, ''),
            headers={'Accept': 'application/json'},
            timeout=10
        )

        if response.status_code == 401:
            return {
                **state,
                "success": False,
                "error_message": "Invalid UK API key. Please get a valid key from https://developer.company-information.service.gov.uk/",
            }

        response.raise_for_status()
        data = response.json()
        companies = data.get("items", [])

        # Format suppliers
        suppliers = []
        for company in companies[:4]:
            address = company.get("address", {})
            address_parts = [
                address.get("address_line_1", ""),
                address.get("address_line_2", ""),
                address.get("locality", ""),
                address.get("postal_code", "")
            ]
            formatted_address = ", ".join(filter(None, address_parts))

            suppliers.append(SupplierSearchResult(
                company_name=company.get("title", "Unknown"),
                company_number=company.get("company_number", ""),
                status=company.get("company_status", ""),
                company_type=company.get("company_type", ""),
                address=formatted_address,
                country="UK",
                industry=state["business_category"],
                description=f"UK registered company in {state['business_category']} industry"
            ))

        ctx.logger.info(f"✅ Found {len(suppliers)} UK suppliers")

        return {
            **state,
            "suppliers": suppliers,
            "success": True,
            "search_summary": f"Found {len(suppliers)} UK suppliers in {state['business_category']} category",
        }

    except Exception as e:
        ctx.logger.error(f"Error searching UK suppliers: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error searching UK suppliers: {str(e)}",
        }


def search_us_suppliers_node(state: SupplierSearchState) -> SupplierSearchState:
    """
    Node 2b: Search US SEC API for suppliers
    """
    ctx = state["ctx"]
    business_category = state["business_category"]

    ctx.logger.info(f"[LangGraph Node: search_us] Searching US SEC API")

    try:
        # Get SEC company tickers
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        # Add 10-second delay before making API call to respect rate limits
        ctx.logger.info("Waiting 10 seconds before calling US SEC API...")
        time.sleep(10)
        ctx.logger.info("Making US SEC API request now...")

        response = requests.get(tickers_url, headers=headers, timeout=10)
        response.raise_for_status()

        companies_data = response.json()

        # Search for companies matching the business category
        search_term = business_category.lower() if business_category != "general" else "supplier"

        matching_companies = []
        for _, company in companies_data.items():
            company_name = company.get("title", "").lower()
            if search_term in company_name or "supplier" in company_name or business_category.lower() in company_name:
                matching_companies.append(SupplierSearchResult(
                    company_name=company.get("title", "Unknown"),
                    ticker=company.get("ticker", ""),
                    cik=str(company.get("cik_str", "")).zfill(10),
                    country="US",
                    industry=state["business_category"],
                    description=f"US registered company in {state['business_category']} industry"
                ))

                if len(matching_companies) >= 4:
                    break

        ctx.logger.info(f"✅ Found {len(matching_companies)} US suppliers")

        return {
            **state,
            "suppliers": matching_companies,
            "success": True,
            "search_summary": f"Found {len(matching_companies)} US suppliers in {state['business_category']} category",
        }

    except Exception as e:
        ctx.logger.error(f"Error searching US suppliers: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error searching US suppliers: {str(e)}",
        }


def format_results_node(state: SupplierSearchState) -> SupplierSearchState:
    """
    Node 3: Format results for output
    """
    ctx = state["ctx"]
    suppliers = state["suppliers"]

    ctx.logger.info(f"[LangGraph Node: format_results] Formatting {len(suppliers)} suppliers")

    if not suppliers:
        return {
            **state,
            "success": False,
            "error_message": f"No suppliers found in {state['country']} for category: {state['business_category']}",
        }

    # Log supplier details
    for idx, supplier in enumerate(suppliers, 1):
        ctx.logger.info(f"  {idx}. {supplier.company_name} ({supplier.country})")

    return state


def error_node(state: SupplierSearchState) -> SupplierSearchState:
    """
    Node 4: Handle errors
    """
    ctx = state["ctx"]

    ctx.logger.error(f"[LangGraph Node: error] {state.get('error_message', 'Unknown error')}")

    return state


# ========== Routing Functions ==========

def route_by_country(state: SupplierSearchState) -> Literal["search_us", "search_uk", "error"]:
    """
    Route to appropriate API based on detected country
    """
    # Check for errors
    if state.get("error_message"):
        return "error"

    country = state.get("country")

    if country == "US":
        return "search_us"
    elif country == "UK":
        return "search_uk"
    else:
        return "error"


def should_continue_to_format(state: SupplierSearchState) -> Literal["format_results", "error"]:
    """
    Determine if we should format output or handle error
    """
    if state.get("error_message"):
        return "error"

    if state.get("suppliers") and len(state["suppliers"]) > 0:
        return "format_results"

    return "error"


# ========== Graph Construction ==========

def build_supplier_search_graph():
    """
    Build the LangGraph workflow for multi-country supplier search

    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(SupplierSearchState)

    # Add nodes
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("search_us", search_us_suppliers_node)
    workflow.add_node("search_uk", search_uk_suppliers_node)
    workflow.add_node("format_results", format_results_node)
    workflow.add_node("error", error_node)

    # Add edges
    workflow.add_edge(START, "parse_input")

    # Conditional routing based on country
    workflow.add_conditional_edges(
        "parse_input",
        route_by_country,
        {
            "search_us": "search_us",
            "search_uk": "search_uk",
            "error": "error",
        },
    )

    # After US search, check if we should continue or error
    workflow.add_conditional_edges(
        "search_us",
        should_continue_to_format,
        {
            "format_results": "format_results",
            "error": "error",
        },
    )

    # After UK search, check if we should continue or error
    workflow.add_conditional_edges(
        "search_uk",
        should_continue_to_format,
        {
            "format_results": "format_results",
            "error": "error",
        },
    )

    # End after formatting or error
    workflow.add_edge("format_results", END)
    workflow.add_edge("error", END)

    return workflow.compile()
