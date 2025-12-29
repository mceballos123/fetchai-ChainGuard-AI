from typing import Optional, Dict, Any, List, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from uagents import Agent, Context, Protocol
from models.find_supplier import (
    FindSupplierRequest,
    FindSupplierResponse,
    SupplierSearchResult,
    SupplierSearchState,
)
import os
import re
import requests
import nest_asyncio
import time

nest_asyncio.apply()
load_dotenv()

find_supplier_agent = Agent(
    name="find_supplier_agent",
    seed=os.getenv("FIND_SUPPLIER_SEED"),
    port=8006,
    mailbox=True,
)

find_supplier_protocol = Protocol(name="find_supplier_protocol", version="1.0")


def detect_country_and_query(message: str) -> tuple[Optional[str], str]:
    
    message_lower = message.lower()

    # Detect country
    country = None
    if any(word in message_lower for word in ["united states", "usa", "us", "america", "american"]):
        country = "US"
    elif any(word in message_lower for word in ["uk", "united kingdom", "britain", "british", "england"]):
        country = "UK"

    return country


def parse_input_node(state: SupplierSearchState) -> SupplierSearchState:
    """Node 1: Parse user input to extract country"""
    ctx = state["ctx"]
    user_query = state["user_query"]
    business_category = state["business_category"]

    ctx.logger.info(f"[LangGraph Node: parse_input] Parsing user query: {user_query}")

    country = detect_country_and_query(user_query)

    if not country:
        ctx.logger.warning("Could not detect country from query")
        return {
            **state,
            "country": None,
            "success": False,
            "error_message": "Could not detect country. Please specify 'US' or 'UK' in your query.",
            "score": 0,
        }

    # Build search query
    search_query = f"{business_category} supplier" if business_category != "general" else "supplier"

    ctx.logger.info(f"Detected country: {country}")
    ctx.logger.info(f"Search query: {search_query}")

    return {
        **state,
        "country": country,
        "search_query": search_query,
        "score": 100,
    }


def search_uk_suppliers_node(state: SupplierSearchState) -> SupplierSearchState:
    
    ctx = state["ctx"]
    search_query = state["search_query"]

    ctx.logger.info(f"[LangGraph Node: search_uk] Searching UK Companies House API")

    api_key = os.getenv('GOV_UK_API_KEY').strip()

    base_url = "https://api.company-information.service.gov.uk/search/companies"
    params = {
        "q": search_query,
        "items_per_page": 4  # Get 4 suppliers
    }

    try:
        # Add 10-second delay before making API call
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
                "score": 0,
            }

        response.raise_for_status()
        data = response.json()
        companies = data.get("items", [])

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

        score = 100 if len(suppliers) >= 3 else 75

        ctx.logger.info(f"✅ Found {len(suppliers)} UK suppliers")

        return {
            **state,
            "suppliers": suppliers,
            "success": True,
            "score": score,
            "search_summary": f"Found {len(suppliers)} UK suppliers in {state['business_category']} category",
        }

    except Exception as e:
        ctx.logger.error(f"Error searching UK suppliers: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error searching UK suppliers: {str(e)}",
            "score": 0,
        }


def search_us_suppliers_node(state: SupplierSearchState) -> SupplierSearchState:
    """Node 2b: Search US SEC API"""
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

        # Add 10-second delay before making API call
        ctx.logger.info("Waiting 10 seconds before calling US SEC API...")
        time.sleep(10)
        ctx.logger.info("Making US SEC API request now...")

        response = requests.get(tickers_url, headers=headers, timeout=10)
        response.raise_for_status()

        companies_data = response.json()

        # Search for companies matching the business category
        search_term = business_category.lower() if business_category != "general" else "supplier"

        matching_companies = []
        for key, company in companies_data.items():
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

        score = 100 if len(matching_companies) >= 3 else 75

        ctx.logger.info(f"✅ Found {len(matching_companies)} US suppliers")

        return {
            **state,
            "suppliers": matching_companies,
            "success": True,
            "score": score,
            "search_summary": f"Found {len(matching_companies)} US suppliers in {state['business_category']} category",
        }

    except Exception as e:
        ctx.logger.error(f"Error searching US suppliers: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error searching US suppliers: {str(e)}",
            "score": 0,
        }


def format_results_node(state: SupplierSearchState) -> SupplierSearchState:
    """Node 3: Format results for output"""
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
    """Node 4: Handle errors"""
    ctx = state["ctx"]

    ctx.logger.error(f"[LangGraph Node: error] {state.get('error_message', 'Unknown error')}")

    return state


def route_by_country(state: SupplierSearchState) -> str:
    """Route to appropriate API based on country"""
    # Check for errors
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    country = state.get("country")

    if country == "US":
        return "search_us"
    elif country == "UK":
        return "search_uk"
    else:
        return "error"


def should_continue_to_format(state: SupplierSearchState) -> str:
    """Determine if we should format output or handle error"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("suppliers") and len(state["suppliers"]) > 0:
        return "format_results"

    return "error"


def build_supplier_search_graph():
    """Build the LangGraph workflow for multi-country supplier search"""
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

    workflow.add_conditional_edges(
        "search_us",
        should_continue_to_format,
        {
            "format_results": "format_results",
            "error": "error",
        },
    )

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


supplier_search_graph = build_supplier_search_graph()


@find_supplier_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize find_supplier agent on startup"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("FIND SUPPLIER AGENT STARTING UP (MULTI-COUNTRY)")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Port: 8006")
    ctx.logger.info("Supported Countries: US (SEC API), UK (Companies House API)")
    ctx.logger.info("=" * 70)

    # Initialize storage
    ctx.storage.set("search_history", [])
    ctx.storage.set("processed_request_ids", [])

    ctx.logger.info(
        "Find Supplier Agent Ready - Listening for FindSupplierRequest messages..."
    )


@find_supplier_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("FIND SUPPLIER AGENT SHUTTING DOWN")
    ctx.logger.info("=" * 70)

    search_history = ctx.storage.get("search_history") or []
    ctx.logger.info(f"Total searches processed: {len(search_history)}")


def extract_business_category(user_query: str) -> str:
    """Extract business category from user query"""
    user_query_lower = user_query.lower()

    categories = [
        "coffee", "tea", "food", "restaurant", "pizza", "burger", "cafe",
        "clothing", "apparel", "fashion", "textile", "bakery", "chocolate",
        "beer", "wine", "beverage", "furniture", "design", "manufacturing",
        "technology", "software", "consulting", "cosmetics", "beauty",
        "wellness", "agriculture", "farming", "organic", "water"
    ]

    for category in categories:
        if category in user_query_lower:
            return category

    words = re.findall(r"\b[a-z]+\b", user_query_lower)
    if len(words) > 2:
        skip_words = {
            "i", "need", "want", "find", "looking", "for", "a", "an",
            "the", "some", "get", "me", "in", "from", "us", "uk"
        }
        for word in words:
            if word not in skip_words and len(word) > 3:
                return word

    return "general"


@find_supplier_protocol.on_message(
    model=FindSupplierRequest, replies=FindSupplierResponse
)
async def handle_find_supplier_request(
    ctx: Context, sender: str, msg: FindSupplierRequest
):
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 FIND SUPPLIER AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"User Query: {msg.user_query}")
    ctx.logger.info(f"Business Category: {msg.business_category}")
    ctx.logger.info("=" * 70)
    ctx.logger.info("")
    ctx.logger.info("🔍 Starting LangGraph Multi-Country Supplier Search...")
    ctx.logger.info("=" * 70)

    try:
        processed_ids = ctx.storage.get("processed_request_ids") or []
        if msg.request_id in processed_ids:
            ctx.logger.warning(f"Duplicate request: {msg.request_id}")
            return

        processed_ids.append(msg.request_id)
        ctx.storage.set("processed_request_ids", processed_ids)

        search_category = (
            msg.business_category
            if msg.business_category != "general"
            else extract_business_category(msg.user_query)
        )

        initial_state: SupplierSearchState = {
            "request_id": msg.request_id,
            "user_query": msg.user_query,
            "business_category": search_category,
            "country": None,
            "search_query": "",
            "suppliers": [],
            "success": False,
            "error_message": None,
            "search_summary": "",
            "score": 100,
            "ctx": ctx,
        }

        ctx.logger.info(f"🔄 Executing LangGraph workflow for category: {search_category}")

        final_state = supplier_search_graph.invoke(initial_state)

        ctx.logger.info("LangGraph Workflow Completed")

        if final_state["success"] and final_state["suppliers"]:
            suppliers = final_state["suppliers"]
            country = final_state["country"]

            ctx.logger.info(f"✅ Found {len(suppliers)} Suppliers in {country}!")
            for idx, supplier in enumerate(suppliers, 1):
                ctx.logger.info(f"  {idx}. {supplier.company_name}")

            response = FindSupplierResponse(
                request_id=msg.request_id,
                success=True,
                suppliers=suppliers,
                country=country,
                search_category=search_category,
                total_results_found=len(suppliers),
                search_summary=final_state["search_summary"],
                # Legacy fields for backwards compatibility
                best_supplier=suppliers[0] if suppliers else None,
                alternative_suppliers=suppliers[1:] if len(suppliers) > 1 else [],
            )

            search_history = ctx.storage.get("search_history") or []
            search_history.append(
                {
                    "request_id": msg.request_id,
                    "country": country,
                    "category": search_category,
                    "suppliers_found": len(suppliers),
                }
            )
            ctx.storage.set("search_history", search_history)

            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: True")
            ctx.logger.info(f"Suppliers: {len(suppliers)}")

            await ctx.send(sender, response)
        else:
            ctx.logger.warning("No suppliers found")
            ctx.logger.warning(
                f"Error: {final_state.get('error_message', 'No suppliers found')}"
            )

            error_response = FindSupplierResponse(
                request_id=msg.request_id,
                success=False,
                country=final_state.get("country", ""),
                search_category=search_category,
                total_results_found=0,
                search_summary=final_state.get("search_summary", "Suppliers not found"),
                error_message=final_state.get("error_message", "No suppliers found"),
            )

            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: False")
            ctx.logger.info(f"Error: {error_response.error_message}")
            await ctx.send(sender, error_response)

    except Exception as e:
        ctx.logger.error(f"Error processing request: {e}")
        import traceback

        traceback.print_exc()

        error_response = FindSupplierResponse(
            request_id=msg.request_id,
            success=False,
            search_category=msg.business_category,
            total_results_found=0,
            search_summary="Internal error during supplier search",
            error_message=str(e),
        )

        await ctx.send(sender, error_response)


find_supplier_agent.include(find_supplier_protocol, publish_manifest=True)

if __name__ == "__main__":
    find_supplier_agent.run()
