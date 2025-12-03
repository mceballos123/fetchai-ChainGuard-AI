from typing import Optional, Dict, Any, List, TypedDict
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langgraph.graph import StateGraph, START, END
from uagents import Agent, Context, Protocol
from models.find_supplier import (
    FindSupplierRequest,
    FindSupplierResponse,
    SupplierSearchResult,
)

import os
import re
import time
import asyncio
import requests
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

find_supplier_agent = Agent(
    name="find_supplier_agent",
    seed=os.getenv("FIND_SUPPLIER_SEED"),
    port=8006,
    mailbox=True,
)

find_supplier_protocol = Protocol(name="find_supplier_protocol", version="1.0")


class SupplierSearchState(TypedDict):
    """State for LangGraph workflow"""

    request_id: str
    user_query: str
    business_category: str
    search_category: str
    search_result: Optional[Dict[str, Any]]
    best_supplier: Optional[SupplierSearchResult]
    success: bool
    error_message: Optional[str]
    search_summary: str
    ctx: Optional[Context]


def extract_country_from_location(location_text: str) -> Optional[str]:
    """
    Extract country name from a location string.
    Example: "Buenos Aires Province, Argentina" -> "Argentina"
    """
    if not location_text:
        return None

    known_countries = [
        "Argentina",
        "Brazil",
        "Chile",
        "Colombia",
        "Mexico",
        "Peru",
        "Uruguay",
        "Venezuela",
        "Ecuador",
        "Bolivia",
        "Paraguay",
        "United States",
        "USA",
        "Canada",
        "United Kingdom",
        "UK",
        "Germany",
        "France",
        "Spain",
        "Italy",
        "Netherlands",
        "Belgium",
        "Switzerland",
        "Austria",
        "Portugal",
        "Sweden",
        "Norway",
        "Denmark",
        "Finland",
        "Ireland",
        "Australia",
        "New Zealand",
        "Japan",
        "China",
        "India",
        "South Korea",
        "Singapore",
        "Taiwan",
        "Thailand",
        "Vietnam",
        "Indonesia",
        "Malaysia",
        "Philippines",
        "South Africa",
        "Kenya",
        "Nigeria",
        "Egypt",
        "Morocco",
        "Israel",
        "United Arab Emirates",
        "Saudi Arabia",
    ]

    location_lower = location_text.lower()

    for country in known_countries:
        if country.lower() in location_lower:
            return country

    # If no known country found, try to get the last part after comma
    parts = location_text.split(",")
    if len(parts) > 1:
        potential_country = parts[-1].strip()
        if len(potential_country) > 2:
            return potential_country

    return None


async def search_b_corp_directory(
    ctx: Context, category: str, max_results: int = 1
) -> Dict[str, Any]:
    """
    Search B Corp directory and extract supplier name + country.

    Process:
    1. Search B Corp directory
    2. Click on first company profile
    3. Extract company name and COUNTRY from profile (Headquarters section)
    4. Country will be sent to financial agent for tariff/inflation analysis
    """
    driver = None
    try:
        ctx.logger.info(f"🔍 Searching B Corp directory for: {category}")

        search_url = f"https://www.bcorporation.net/en-us/find-a-b-corp/?query={category}&sortBy=companies-production-en-us"

        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(60)
        driver.get(search_url)
        time.sleep(8)

        first_company = None
        company_country = None
        profile_url = None

        # Try to find and click first company profile
        try:
            # Use WebDriverWait for better control
            wait = WebDriverWait(driver, 15)
            
            company_links = driver.find_elements(
                By.CSS_SELECTOR, 'a[href*="/find-a-b-corp/company/"]'
            )

            if company_links:
                profile_url = company_links[0].get_attribute("href")
                ctx.logger.info(f"📍 Found company profile: {profile_url}")
                driver.get(profile_url)
                time.sleep(5)
                ctx.logger.info("🌐 Loaded company profile page")
        except Exception as e:
            ctx.logger.warning(f"Could not navigate to profile: {e}")
            # Check if driver is still alive
            try:
                if not driver or driver.service.process is None:
                    ctx.logger.error("Driver died while navigating, stopping")
                    raise
            except:
                raise

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract company name
        h1_tag = soup.find("h1")
        if h1_tag:
            potential_name = h1_tag.get_text(strip=True)
            if potential_name and len(potential_name) > 2 and len(potential_name) < 100:
                first_company = potential_name
                ctx.logger.info(f"✅ Found company name: {first_company}")

        if not first_company:
            company_span = soup.find("span", {"data-testid": "company-name-desktop"})
            if company_span:
                first_company = company_span.get_text(strip=True)

        if not first_company:
            company_span = soup.find("span", {"data-testid": "company-name-mobile"})
            if company_span:
                first_company = company_span.get_text(strip=True)

        if not first_company:
            pattern = r'data-testid="company-name[^"]*"[^>]*>([^<]+)</span>'
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                first_company = matches[0].strip()

        # Extract country - CRITICAL for financial analysis
        ctx.logger.info("🌍 Extracting supplier country...")

        # Method 1: Look for Headquarters section
        all_text = soup.get_text(separator="|", strip=True)

        hq_patterns = [
            r"Headquarters\|([^|]+)",
            r"Headquarters([A-Za-z\s,]+(?:Argentina|Brazil|Chile|Colombia|Mexico|Peru|United States|Canada|United Kingdom|Germany|France|Spain|Italy|Australia|Japan|China|India))",
        ]

        for pattern in hq_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                headquarters_text = match.group(1).strip()
                ctx.logger.info(f"   Found Headquarters: {headquarters_text}")
                company_country = extract_country_from_location(headquarters_text)
                if company_country:
                    ctx.logger.info(f"   ✅ Country: {company_country}")
                    break

        # Method 2: Search for known countries in context
        if not company_country:
            known_countries = [
                ("Argentina", ["Buenos Aires", "Argentina"]),
                ("Brazil", ["Brazil", "São Paulo"]),
                ("Chile", ["Chile", "Santiago"]),
                ("Colombia", ["Colombia", "Bogotá"]),
                ("Mexico", ["Mexico", "México"]),
                ("Peru", ["Peru", "Lima"]),
                ("United States", ["United States", "USA"]),
                ("Canada", ["Canada"]),
                ("United Kingdom", ["United Kingdom", "UK"]),
            ]

            for country, keywords in known_countries:
                for keyword in keywords:
                    if keyword.lower() in html_content.lower():
                        context_patterns = [
                            rf"Headquarters[^<]*{keyword}",
                            rf"{keyword}[^<]*Province",
                            rf"Province[^<]*{keyword}",
                        ]
                        for ctx_pattern in context_patterns:
                            if re.search(ctx_pattern, html_content, re.IGNORECASE):
                                company_country = country
                                ctx.logger.info(
                                    f"   ✅ Country (via context): {country}"
                                )
                                break
                        if company_country:
                            break
                if company_country:
                    break

        # Method 3: Search in visible text
        if not company_country:
            visible_text = soup.get_text()
            for country in [
                "Argentina",
                "Brazil",
                "Chile",
                "Colombia",
                "Mexico",
                "Peru",
                "United States",
                "Canada",
            ]:
                if country in visible_text:
                    company_country = country
                    ctx.logger.info(f"   Found country in text: {country}")
                    break

        if not company_country:
            company_country = "Unknown"
            ctx.logger.warning("⚠️ Could not determine country - defaulting to Unknown")

        if not first_company:
            ctx.logger.warning("❌ Could not extract company name")
            return {
                "success": False,
                "results": [],
                "total_found": 0,
                "search_url": search_url,
                "error": "No company found",
            }

        ctx.logger.info(f"✅ Supplier: {first_company} from {company_country}")

        result = SupplierSearchResult(
            company_name=first_company,
            location=company_country,  # COUNTRY for financial analysis
            industry=category.title(),
            b_corp_profile_url=profile_url or search_url,
            description=f"B Corporation certified company specializing in {category}",
        )

        return {
            "success": True,
            "results": [result],
            "total_found": 1,
            "search_url": search_url,
        }

    except Exception as e:
        ctx.logger.error(f"Error during search: {e}")
        import traceback
        traceback.print_exc()

        return {"success": False, "results": [], "total_found": 0, "error": str(e)}
    
    finally:
        # Always cleanup driver in finally block
        if driver:
            try:
                driver.quit()
                ctx.logger.info("B Corp search WebDriver cleaned up")
            except Exception as e:
                ctx.logger.warning(f"Error closing driver: {e}")


def find_supplier_node(state: SupplierSearchState) -> SupplierSearchState:
    """Node 1: Find supplier by searching B Corp directory"""
    ctx = state["ctx"]
    search_category = state["search_category"]

    ctx.logger.info(f"[LangGraph Node: find_supplier] Searching for: {search_category}")

    search_result = asyncio.run(
        search_b_corp_directory(ctx, search_category, max_results=1)
    )

    ctx.logger.info(f"Search result: {search_result}")

    if not search_result.get("success"):
        return {
            **state,
            "success": False,
            "error_message": search_result.get("error", "Unknown error occurred"),
            "search_summary": "Failed to search B Corporation directory",
        }

    return {
        **state,
        "search_result": search_result,
        "success": search_result.get("success", False),
    }


def process_results_node(state: SupplierSearchState) -> SupplierSearchState:
    """Node 2: Process search results and select best supplier"""
    ctx = state["ctx"]
    search_result = state["search_result"]

    ctx.logger.info("[LangGraph Node: process_results] Processing search results")

    if not search_result or not search_result.get("success"):
        error_msg = (
            search_result.get("error", "Unknown error occurred")
            if search_result
            else "Search failed"
        )
        return {
            **state,
            "success": False,
            "error_message": error_msg,
            "search_summary": "Failed to search B Corporation directory",
        }

    results = search_result.get("results", [])

    if not results:
        return {
            **state,
            "success": False,
            "error_message": f"No suppliers found in category: {state['search_category']}",
            "search_summary": f"No B Corporation certified companies found for '{state['search_category']}'",
        }

    best_supplier = select_best_supplier(ctx, results, state["user_query"])
    search_summary = f"Found B Corporation certified company in the {state['search_category']} category: {best_supplier.company_name}"

    return {
        **state,
        "best_supplier": best_supplier,
        "success": True,
        "search_summary": search_summary,
    }


def supplier_found_node(state: SupplierSearchState) -> SupplierSearchState:
    """Node 3: Handle successful supplier found"""
    ctx = state["ctx"]
    best_supplier = state["best_supplier"]

    ctx.logger.info(
        f"[LangGraph Node: supplier_found] ✓ Supplier found: {best_supplier.company_name}"
    )
    ctx.logger.info(f"  Location: {best_supplier.location}")
    ctx.logger.info(f"  Industry: {best_supplier.industry}")
    ctx.logger.info(f"  Summary: {state['search_summary']}")

    return state


def supplier_not_found_node(state: SupplierSearchState) -> SupplierSearchState:
    """Node 4: Handle supplier not found"""
    ctx = state["ctx"]

    ctx.logger.warning(f"[LangGraph Node: supplier_not_found] ✗ Supplier not found")
    ctx.logger.warning(f"  Category: {state['search_category']}")
    ctx.logger.warning(f"  Error: {state.get('error_message', 'No results')}")

    return state


def route_search_results(state: SupplierSearchState) -> str:
    """Conditional routing: supplier found or not found"""
    if state.get("success") and state.get("best_supplier"):
        return "supplier_found"
    else:
        return "supplier_not_found"


def build_supplier_search_graph():
    """Build the LangGraph workflow for supplier search"""
    workflow = StateGraph(SupplierSearchState)

    workflow.add_node("find_supplier", find_supplier_node)
    workflow.add_node("process_results", process_results_node)
    workflow.add_node("supplier_found", supplier_found_node)
    workflow.add_node("supplier_not_found", supplier_not_found_node)

    workflow.add_edge(START, "find_supplier")
    workflow.add_edge("find_supplier", "process_results")

    workflow.add_conditional_edges(
        "process_results",
        route_search_results,
        {
            "supplier_found": "supplier_found",
            "supplier_not_found": "supplier_not_found",
        },
    )

    workflow.add_edge("supplier_found", END)
    workflow.add_edge("supplier_not_found", END)

    return workflow.compile()


supplier_search_graph = build_supplier_search_graph()


@find_supplier_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize find_supplier agent on startup"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("FIND SUPPLIER AGENT STARTING UP")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Port: 8006")
    ctx.logger.info("=" * 70)

    # Initialize storage for tracking searches
    ctx.storage.set("search_history", [])
    ctx.storage.set("processed_request_ids", [])

    ctx.logger.info(
        "✅ Find Supplier Agent Ready - Listening for FindSupplierRequest messages..."
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
    user_query_lower = user_query.lower()

    categories = [
        "water",
        "coffee",
        "tea",
        "food",
        "restaurant",
        "pizza",
        "burger",
        "cafe",
        "clothing",
        "apparel",
        "fashion",
        "textile",
        "bakery",
        "chocolate",
        "beer",
        "wine",
        "beverage",
        "furniture",
        "design",
        "manufacturing",
        "technology",
        "software",
        "consulting",
        "cosmetics",
        "beauty",
        "wellness",
        "agriculture",
        "farming",
        "organic",
    ]

    for category in categories:
        if category in user_query_lower:
            return category

    words = re.findall(r"\b[a-z]+\b", user_query_lower)
    if len(words) > 2:
        skip_words = {
            "i",
            "need",
            "want",
            "find",
            "looking",
            "for",
            "a",
            "an",
            "the",
            "some",
            "get",
            "me",
        }
        for word in words:
            if word not in skip_words and len(word) > 3:
                return word

    return "general"


# was async, changed to sync


def select_best_supplier(
    ctx: Context, results: List[SupplierSearchResult], user_query: str
) -> Optional[SupplierSearchResult]:
    if not results:
        return None

    best_supplier = results[0]
    ctx.logger.info(f"Selected supplier: {best_supplier.company_name}")

    return best_supplier


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
    ctx.logger.info("🔍 Starting LangGraph Supplier Search Workflow...")
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
            "business_category": msg.business_category,
            "search_category": search_category,
            "search_result": None,
            "best_supplier": None,
            "success": False,
            "error_message": None,
            "search_summary": "",
            "ctx": ctx,
        }

        ctx.logger.info(
            f"🔄 Executing LangGraph workflow for category: {search_category}"
        )
        ctx.logger.info(f"Will search B Corp directory...")

        final_state = supplier_search_graph.invoke(initial_state)

        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("✅ LangGraph Workflow Completed")
        ctx.logger.info("=" * 70)

        if final_state["success"] and final_state["best_supplier"]:
            best_supplier = final_state["best_supplier"]

            ctx.logger.info("✅ Supplier Found Successfully!")
            ctx.logger.info(f"Company Name: {best_supplier.company_name}")
            ctx.logger.info(f"Location: {best_supplier.location}")
            ctx.logger.info(f"Industry: {best_supplier.industry}")
            ctx.logger.info(f"B Corp Profile: {best_supplier.b_corp_profile_url}")

            response = FindSupplierResponse(
                request_id=msg.request_id,
                success=True,
                best_supplier=best_supplier,
                alternative_suppliers=[],
                search_category=search_category,
                total_results_found=1,
                search_summary=final_state["search_summary"],
            )

            search_history = ctx.storage.get("search_history") or []
            search_history.append(
                {
                    "request_id": msg.request_id,
                    "category": search_category,
                    "selected_supplier": best_supplier.company_name,
                }
            )
            ctx.storage.set("search_history", search_history)

            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("📤 FIND SUPPLIER AGENT: SENDING RESPONSE")
            ctx.logger.info("=" * 70)
            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: True")
            ctx.logger.info(f"Supplier: {best_supplier.company_name}")
            ctx.logger.info("=" * 70)

            await ctx.send(sender, response)
        else:
            ctx.logger.warning("❌ No supplier found")
            ctx.logger.warning(
                f"Error: {final_state.get('error_message', 'No suppliers found')}"
            )

            error_response = FindSupplierResponse(
                request_id=msg.request_id,
                success=False,
                search_category=search_category,
                total_results_found=0,
                search_summary=final_state.get("search_summary", "Supplier not found"),
                error_message=final_state.get("error_message", "No suppliers found"),
            )

            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("📤 FIND SUPPLIER AGENT: SENDING ERROR RESPONSE")
            ctx.logger.info("=" * 70)
            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: False")
            ctx.logger.info(f"Error: {error_response.error_message}")
            ctx.logger.info("=" * 70)

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
