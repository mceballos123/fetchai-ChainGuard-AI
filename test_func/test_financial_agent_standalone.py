"""
Standalone test for Financial Agent
Tests the financial analysis workflow using the EXACT same logic as the actual agent
Scrapes B Corp page to get supplier country (no hardcoded data)

Usage:
    python -m test_func.test_financial_agent_standalone
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Import the actual models used by the agent
from models.financial import FinancialRequest, FinancialResponse


class MockContext:
    """Mock Context that mimics uagents Context"""

    def __init__(self):
        self.logs = []
        self._storage = {}

    class Storage:
        def __init__(self, parent):
            self.parent = parent

        def get(self, key: str, default=None):
            return self.parent._storage.get(key, default)

        def set(self, key: str, value):
            self.parent._storage[key] = value

    class Logger:
        def __init__(self, parent):
            self.parent = parent

        def info(self, msg):
            print(f"[INFO] {msg}")
            self.parent.logs.append(("INFO", msg))

        def error(self, msg):
            print(f"[ERROR] {msg}")
            self.parent.logs.append(("ERROR", msg))

        def warning(self, msg):
            print(f"[WARNING] {msg}")
            self.parent.logs.append(("WARNING", msg))

    @property
    def logger(self):
        return self.Logger(self)

    @property
    def storage(self):
        return self.Storage(self)


def extract_country_from_location(location_text: str) -> Optional[str]:
    """
    Extract country from location text (e.g., "Catalonia, Spain" -> "Spain")
    Same logic as find_supplier_agent.py
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


async def scrape_country_from_bcorp(b_corp_url: str) -> Optional[str]:
    """
    Scrape country from B Corp profile page
    Same logic as find_supplier_agent.py
    """
    driver = None
    try:
        print(f"🌐 Scraping country from B Corp page: {b_corp_url}")

        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )

        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(60)

        print("   Loading B Corp page...")
        driver.get(b_corp_url)
        time.sleep(5)

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")

        company_country = None

        # Method 1: Look for Headquarters section
        print("   Searching for Headquarters information...")
        all_text = soup.get_text(separator="|", strip=True)

        hq_patterns = [
            r"Headquarters\|([^|]+)",
            r"Headquarters([A-Za-z\s,]+(?:Argentina|Brazil|Chile|Colombia|Mexico|Peru|United States|Canada|United Kingdom|Germany|France|Spain|Italy|Australia|Japan|China|India))",
        ]

        for pattern in hq_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                headquarters_text = match.group(1).strip()
                print(f"   Found Headquarters: {headquarters_text}")
                company_country = extract_country_from_location(headquarters_text)
                if company_country:
                    print(f"   ✅ Extracted Country: {company_country}")
                    break

        # Method 2: Search for "Operates In" field
        if not company_country:
            print("   Trying 'Operates In' field...")
            operates_match = re.search(r"Operates In\|([^|]+)", all_text, re.IGNORECASE)
            if operates_match:
                operates_text = operates_match.group(1).strip()
                print(f"   Found Operates In: {operates_text}")
                company_country = extract_country_from_location(operates_text)
                if company_country:
                    print(f"   ✅ Extracted Country: {company_country}")

        # Method 3: Search in visible text
        if not company_country:
            print("   Searching in visible text...")
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
                "Spain",
                "Germany",
                "France",
                "Italy",
                "United Kingdom",
            ]:
                if country in visible_text:
                    company_country = country
                    print(f"   ✅ Found country in text: {country}")
                    break

        if not company_country:
            print("   ⚠️ Could not determine country - defaulting to Unknown")
            company_country = "Unknown"

        return company_country

    except Exception as e:
        print(f"   ❌ Error scraping B Corp page: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if driver:
            try:
                driver.quit()
                print("   ✅ WebDriver cleaned up")
            except Exception as e:
                print(f"   ⚠️ Error closing driver: {e}")


class TestFinancialAgent:
    """Test Financial Agent using EXACT same logic as actual agent"""

    def __init__(self):
        self.ctx = MockContext()
        self.rag_system = None
        self.workflow = None

    async def initialize(self):
        """Initialize RAG system and workflow (mimics agent startup)"""
        from langgraph_logic.financial_langgraph import (
            FinancialRAGSystem,
            build_financial_workflow,
        )

        print("🔧 Initializing Financial Agent (like agent startup)...")

        # Initialize RAG system
        if self.rag_system is None:
            self.rag_system = FinancialRAGSystem()

        success = await self.rag_system.initialize(self.ctx)

        if success:
            self.ctx.logger.info("Financial Agent ready with RAG system!")

            # Build LangGraph workflow
            if self.workflow is None:
                self.workflow = build_financial_workflow(self.rag_system, self.ctx)
                self.ctx.logger.info("LangGraph financial workflow built!")
        else:
            self.ctx.logger.warning("Financial Agent running in fallback mode (no RAG)")

        # Initialize state storage (like agent does)
        self.ctx.storage.set("supplier_history", [])
        self.ctx.storage.set("current_supplier", None)
        self.ctx.storage.set("previous_supplier", None)
        self.ctx.storage.set("processed_request_ids", [])

        print("✅ Financial Agent initialized\n")

    async def handle_financial_request(
        self, msg: FinancialRequest
    ) -> FinancialResponse:
        """
        Handle financial request using EXACT same logic as actual agent
        (Mirrors: agents/supplier_search/financial_agent.py::handle_financial_request)
        """
        sender = "test_orchestrator"

        # === LOGGING (same as actual agent) ===
        self.ctx.logger.info("")
        self.ctx.logger.info("=" * 70)
        self.ctx.logger.info("📥 FINANCIAL AGENT: RECEIVED REQUEST")
        self.ctx.logger.info("=" * 70)
        self.ctx.logger.info(f"From: {sender}")
        self.ctx.logger.info(f"Request ID: {msg.request_id}")
        self.ctx.logger.info(f"Supplier Name: {msg.supplier_name}")
        self.ctx.logger.info(f"Supplier Country: {msg.supplier_country}")
        self.ctx.logger.info(f"Industry: {msg.industry}")
        self.ctx.logger.info("=" * 70)

        # === STATE MANAGEMENT: Move current to previous ===
        previous_supplier = self.ctx.storage.get("current_supplier")
        if previous_supplier:
            self.ctx.storage.set("previous_supplier", previous_supplier)

        # === STATE MANAGEMENT: Set new current supplier ===
        current_supplier_info = {
            "request_id": msg.request_id,
            "supplier_name": msg.supplier_name,
            "supplier_country": msg.supplier_country,
            "industry": msg.industry,
            "timestamp": msg.timestamp,
            "sender": sender,
        }
        self.ctx.storage.set("current_supplier", current_supplier_info)

        try:
            # === USING LANGGRAPH WORKFLOW WITH RAG (same as actual agent) ===
            from langgraph_logic.state_schemas import SupplierWorkflowState

            if self.workflow is None:
                self.ctx.logger.error("❌ Financial workflow not initialized!")
                raise RuntimeError("Financial workflow not ready")

            self.ctx.logger.info("🔄 Starting LangGraph financial workflow...")
            self.ctx.logger.info(
                f"Will analyze tariff/inflation data for: {msg.supplier_country}"
            )

            # Create workflow state from request (EXACT same as actual agent)
            workflow_state = SupplierWorkflowState(
                request_id=msg.request_id,
                timestamp=msg.timestamp,
                user_input="",
                business_type="",
                company_values="",
                industry=msg.industry,
                product_needed="",
                supplier_name=msg.supplier_name,
                supplier_location=None,
                supplier_country=msg.supplier_country,  # Critical for Trade War Tracker
                b_corp_profile_url=None,
                retrieved_documents=None,
                rag_context=None,
                compliance_score=None,
                ethics_info=None,
                sustainability_info=None,
                violations=None,
                financial_score=None,
                financial_info=None,
                risk_score=None,
                risk_details=None,
                risk_factors=None,
                current_step="financial_check",
                error_message=None,
                should_continue=True,
                messages=[],
            )

            # Invoke workflow (same as actual agent)
            result = self.workflow.invoke(workflow_state)

            self.ctx.logger.info("✅ LangGraph workflow completed")

            self.ctx.logger.info(f"Result variable on line 380: {result}")

            # Extract results (same as actual agent)
            financial_score = result.get("financial_score", 70.0)
            financial_details = result.get("financial_info", "Analysis complete")
            risk_factors = result.get("risk_factors", [])

            self.ctx.logger.info(f"Financial Analysis Results:")
            self.ctx.logger.info(f"  - Score: {financial_score}/100")
            self.ctx.logger.info(f"  - Risk Factors: {len(risk_factors)}")
            self.ctx.logger.info(
                f"  - Status: {'APPROVED' if financial_score >= 60 else 'REJECTED'}"
            )

            # Build response (same as actual agent)
            response = FinancialResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                financial_score=financial_score,
                financial_details=financial_details,
                risk_factors=risk_factors if risk_factors else [],
                timestamp=datetime.now().isoformat(),
            )

            self.ctx.logger.info(f"Response variable on line 404: {response}")

            # Update state with results (same as actual agent)
            current_step = (
                "financial_approved" if financial_score >= 60 else "financial_rejected"
            )
            current_supplier_info["financial_score"] = response.financial_score
            current_supplier_info["financial_details"] = response.financial_details
            current_supplier_info["risk_factors"] = response.risk_factors
            current_supplier_info["current_step"] = current_step
            self.ctx.storage.set("current_supplier", current_supplier_info)

            # Add to history
            supplier_history = self.ctx.storage.get("supplier_history") or []
            supplier_history.append(current_supplier_info)
            self.ctx.storage.set("supplier_history", supplier_history)

            self.ctx.logger.info("")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info("📤 FINANCIAL AGENT: SENDING RESPONSE")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info(f"To: {sender}")
            self.ctx.logger.info(f"Request ID: {msg.request_id}")
            self.ctx.logger.info(f"Financial Score: {response.financial_score}/100")
            self.ctx.logger.info(f"Risk Factors: {len(response.risk_factors)}")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info("")

            return response

        except Exception as e:
            self.ctx.logger.error(f"❌ Error in financial analysis: {e}")
            import traceback

            traceback.print_exc()

            # Return error response
            return FinancialResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                financial_score=0.0,
                financial_details=f"Error: {str(e)}",
                risk_factors=["analysis_error"],
                timestamp=datetime.now().isoformat(),
            )


async def run_tests():
    """Run financial agent test with real B Corp scraping (NO hardcoded data)"""
    tester = TestFinancialAgent()

    # Initialize agent (like startup event)
    await tester.initialize()

    # Test: NOMAD COFFEE SL - Scrape country from B Corp page
    print("\n" + "#" * 80)
    print("TEST: NOMAD COFFEE SL (Real B Corp - Scraping Country)")
    print("#" * 80)

    # B Corp URL for NOMAD COFFEE SL
    b_corp_url = (
        "https://www.bcorporation.net/en-us/find-a-b-corp/company/nomad-coffee-sl/"
    )
    supplier_name = "NOMAD COFFEE SL"
    industry = "Coffee"

    print(f"\n📍 Step 1: Scraping country from B Corp page...")
    print(f"   URL: {b_corp_url}")

    # Scrape the country from the B Corp page (NO hardcoding)
    supplier_country = await scrape_country_from_bcorp(b_corp_url)

    if not supplier_country or supplier_country == "Unknown":
        print("\n❌ Failed to scrape country from B Corp page")
        print("   Cannot proceed with financial analysis without country information")
        return

    print(f"\n✅ Country scraped successfully: {supplier_country}")
    print(
        f"\n📍 Step 2: Running financial analysis for {supplier_name} from {supplier_country}..."
    )

    # Create FinancialRequest with scraped country
    request = FinancialRequest(
        request_id=str(uuid.uuid4()),
        supplier_name=supplier_name,
        supplier_country=supplier_country,  # This is the SCRAPED country, not hardcoded
        industry=industry,
        timestamp=datetime.now().isoformat(),
    )

    # Handle the financial request
    response = await tester.handle_financial_request(request)

    # Display results
    print("\n" + "=" * 80)
    print("📊 FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Supplier: {supplier_name}")
    print(f"Country: {supplier_country} (scraped from B Corp)")
    print(f"Industry: {industry}")
    print(f"Financial Score: {response.financial_score}/100")
    print(
        f"Status: {'✅ PASS' if response.financial_score >= 60 else '❌ FAIL'} (Threshold: 60)"
    )
    print(f"\nRisk Factors:")
    if response.risk_factors:
        for factor in response.risk_factors:
            print(f"  • {factor}")
    else:
        print("  • None identified")
    print(f"\nFinancial Details:")
    print(f"  {response.financial_details}")
    print("=" * 80)

    print("\n🎉 FINANCIAL AGENT TEST COMPLETED")
    print(
        f"Processed {len(tester.ctx.storage.get('supplier_history', []))} supplier(s) total"
    )


if __name__ == "__main__":
    print("Starting Financial Agent Standalone Tests...")
    asyncio.run(run_tests())
