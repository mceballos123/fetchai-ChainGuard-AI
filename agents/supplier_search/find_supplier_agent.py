from uagents import Agent, Context, Protocol
import os
import re
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests

# Import models
from models.find_supplier import (
    FindSupplierRequest,
    FindSupplierResponse,
    SupplierSearchResult,
)

load_dotenv()

find_supplier_agent = Agent(
    name="find_supplier_agent",
    seed=os.getenv("FIND_SUPPLIER_SEED"),
    port=8006,
    mailbox=True,
)

find_supplier_protocol = Protocol(name="find_supplier_protocol", version="1.0")


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


async def search_b_corp_directory(
    ctx: Context, category: str, max_results: int = 1
) -> Dict[str, Any]:
    driver = None
    try:
        ctx.logger.info(f"Searching B Corp directory for: {category}")

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
        driver.set_page_load_timeout(30)
        driver.get(search_url)
        time.sleep(5)

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")

        first_company = None

        company_span = soup.find("span", {"data-testid": "company-name-desktop"})
        if company_span:
            first_company = company_span.get_text(strip=True)

        if not first_company:
            company_span = soup.find("span", {"data-testid": "company-name-mobile"})
            if company_span:
                first_company = company_span.get_text(strip=True)

        if not first_company:
            company_spans = soup.find_all(
                "span", {"data-testid": re.compile("company-name")}
            )
            if company_spans:
                first_company = company_spans[0].get_text(strip=True)

        if not first_company:
            pattern = r'data-testid="company-name[^"]*"[^>]*>([^<]+)</span>'
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                first_company = matches[0].strip()

        if not first_company:
            company_patterns = [
                r'<span[^>]*class="[^"]*text-xl[^"]*font-medium[^"]*"[^>]*>([^<]+)</span>',
                r'"name"\s*:\s*"([^"]+)"',
                r'"companyName"\s*:\s*"([^"]+)"',
            ]

            for pattern in company_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                if matches:
                    for match in matches:
                        cleaned = match.strip()
                        if cleaned and len(cleaned) > 3 and len(cleaned) < 100:
                            skip_words = [
                                "showing",
                                "sort",
                                "filter",
                                "location",
                                "ownership",
                                "more filters",
                                "find a b corp",
                            ]
                            if not any(skip in cleaned.lower() for skip in skip_words):
                                first_company = cleaned
                                break
                if first_company:
                    break

        if not first_company:
            json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}'
            json_matches = re.findall(json_pattern, html_content)
            if json_matches:
                for match in json_matches:
                    if match and len(match) > 3 and len(match) < 100:
                        first_company = match.strip()
                        break

        unique_companies = []
        if first_company:
            unique_companies = [first_company]
            ctx.logger.info(f"Found company: {first_company}")
        else:
            ctx.logger.warning("Could not extract company name from page")

        if driver:
            driver.quit()
            driver = None

        if not unique_companies:
            ctx.logger.warning("No companies found")
            return {
                "success": True,
                "results": [],
                "total_found": 0,
                "search_url": search_url,
            }

        results = []
        for idx, company_name in enumerate(unique_companies[:max_results]):
            result = SupplierSearchResult(
                company_name=company_name,
                location="United States",
                industry=category.title(),
                b_corp_profile_url=f"https://www.bcorporation.net/en-us/find-a-b-corp/?query={category}",
                description=f"B Corporation certified company specializing in {category}",
            )
            results.append(result)

        return {
            "success": True,
            "results": results,
            "total_found": len(unique_companies),
            "search_url": search_url,
        }

    except Exception as e:
        ctx.logger.error(f"Error during search: {e}")
        import traceback

        traceback.print_exc()

        if driver:
            try:
                driver.quit()
            except:
                pass

        return {"success": False, "results": [], "total_found": 0, "error": str(e)}


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
    ctx.logger.info(f"Received request {msg.request_id}: {msg.user_query}")

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

        search_result = await search_b_corp_directory(
            ctx, search_category, max_results=1
        )

        if not search_result["success"]:
            error_response = FindSupplierResponse(
                request_id=msg.request_id,
                success=False,
                search_category=search_category,
                total_results_found=0,
                search_summary="Failed to search B Corporation directory",
                error_message=search_result.get("error", "Unknown error occurred"),
            )

            await ctx.send(sender, error_response)
            ctx.logger.error(f"Search failed: {error_response.error_message}")
            return

        results = search_result["results"]
        total_found = search_result["total_found"]

        if not results:
            no_results_response = FindSupplierResponse(
                request_id=msg.request_id,
                success=False,
                search_category=search_category,
                total_results_found=0,
                search_summary=f"No B Corporation certified companies found for '{search_category}'",
                error_message=f"No suppliers found in category: {search_category}",
            )

            await ctx.send(sender, no_results_response)
            ctx.logger.warning(f"No results for: {search_category}")
            return

        best_supplier = select_best_supplier(ctx, results, msg.user_query)
        search_summary = f"Found B Corporation certified company in the {search_category} category: {best_supplier.company_name}"

        response = FindSupplierResponse(
            request_id=msg.request_id,
            success=True,
            best_supplier=best_supplier,
            alternative_suppliers=[],
            search_category=search_category,
            total_results_found=1,
            search_summary=search_summary,
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

        ctx.logger.info(f"Sending response: {best_supplier.company_name}")
        await ctx.send(sender, response)

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
