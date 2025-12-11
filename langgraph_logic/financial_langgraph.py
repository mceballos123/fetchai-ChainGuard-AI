"""
Financial LangGraph Workflow - With Google Gemini LLM Reasoning (No Pinecone/RAG)

This module handles financial risk analysis by:
1. Extracting supplier country from B Corp Headquarters section
2. Analyzing trade relationship between user's country and supplier's country
3. Using Google Gemini LLM for reasoning and generating detailed financial analysis
4. Returning financial score and risk factors

Uses Gemini for LLM reasoning, but no vector stores or RAG systems.
"""

from typing import Dict, Any, List, Optional, Literal
from models.financial import FinancialRequest, FinancialResponse
from langgraph_logic.state_schemas import SupplierWorkflowState
import os
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver

from dotenv import load_dotenv
from uagents import Context
from langgraph.graph import StateGraph, START, END
from prompts.finance_prompt import finance_prompt
import google.generativeai as genai

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class FinancialAnalysisSystem:
    """
    Financial analysis system - extracts country from B Corp page and analyzes trade relationships.

    Uses Google Gemini LLM for reasoning and detailed analysis generation.
    No Pinecone or RAG needed - uses rule-based country analysis with LLM-generated details.
    """

    def __init__(self):
        self.initialized = False

        # Initialize Gemini model with token optimization
        self.gemini_model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "max_output_tokens": 1400,
                "temperature": 0.3,
            },
        )

    def _extract_country_from_text(self, text: str, ctx: Context) -> Optional[str]:
        """
        Extract country name from text content.

        For headquarters like "Catalonia, Spain" -> extracts "Spain" (the country, not the city/region).

        Args:
            text: Text to search for country (e.g., "Catalonia, Spain" or "Catalonia|Spain")
            ctx: Context for logging

        Returns:
            Country name or None
        """
        if not text:
            return None

        # List of known countries
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
            "Greece",
            "Poland",
            "Czech Republic",
            "Hungary",
            "Romania",
            "Turkey",
            "Russia",
            "Ukraine",
        ]

        ctx.logger.info(f"Extracting country from text: {text}")

        text_lower = text.lower()

        # PRIORITY 1: Check for known countries in the text
        for country in known_countries:
            if country.lower() in text_lower:
                if country == "USA":
                    return "United States"
                if country == "UK":
                    return "United Kingdom"
                return country

        # PRIORITY 2: Extract from comma-separated location
        parts = text.split(",")
        if len(parts) > 1:
            potential_country = parts[-1].strip()
            if len(potential_country) > 2 and potential_country[0].isupper():
                return potential_country

        # PRIORITY 3: Try pipe-separated
        parts = text.split("|")
        if len(parts) > 1:
            for part in reversed(parts):
                part_clean = part.strip()
                for country in known_countries:
                    if country.lower() == part_clean.lower():
                        return country

        return None

    async def scrape_bcorp_country(
        self, ctx: Context, supplier_name: str, b_corp_url: str
    ) -> Dict[str, str]:
        """
        Scrape B Corp page ONLY to extract country from Headquarters section.

        Args:
            ctx: Context for logging
            supplier_name: Name of supplier
            b_corp_url: B Corp profile URL

        Returns:
            Dictionary with supplier_name, country, and url
        """
        driver = None
        try:
            ctx.logger.info(f"Scraping B Corp page to extract country: {supplier_name}")

            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(
                "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )

            # Use Remote Selenium if SELENIUM_REMOTE_URL is set (Docker), else local Chrome
            selenium_url = os.getenv("SELENIUM_REMOTE_URL")
            if selenium_url:
                ctx.logger.info(f"Using remote Selenium at: {selenium_url}")
                driver = webdriver.Remote(
                    command_executor=selenium_url, options=chrome_options
                )
            else:
                driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(60)

            ctx.logger.info(f"Loading B Corp page: {b_corp_url}")
            driver.get(b_corp_url)
            time.sleep(5)

            scraped_data = {
                "supplier_name": supplier_name,
                "country": "",
                "url": b_corp_url,
            }

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # METHOD 1: Try to find Headquarters section directly in DOM
            headquarters_found = False
            hq_spans = soup.find_all(
                "span", string=re.compile(r"Headquarters", re.IGNORECASE)
            )
            for hq_span in hq_spans:
                parent = hq_span.parent
                if parent:
                    full_text = parent.get_text(separator=" ", strip=True)
                    location_text = full_text.replace("Headquarters", "").strip()
                    if location_text:
                        ctx.logger.info(
                            f"Found Headquarters location (DOM method): {location_text}"
                        )
                        scraped_data["country"] = self._extract_country_from_text(
                            location_text, ctx
                        )
                        if scraped_data["country"]:
                            headquarters_found = True
                            break

            # METHOD 2: Fallback to text parsing if DOM method didn't work
            if not headquarters_found:
                all_text = soup.get_text(separator="|", strip=True)
                hq_patterns = [
                    r"Headquarters\|([^|]+)\|?,?\|?([^|]*)",
                    r"Headquarters\|([^|]+)",
                    r"Headquarters[:\s]+([^|]+)",
                ]
                for pattern in hq_patterns:
                    match = re.search(pattern, all_text, re.IGNORECASE)
                    if match:
                        headquarters_text = " ".join(
                            g for g in match.groups() if g
                        ).strip()
                        headquarters_text = headquarters_text.replace("|", " ").replace(
                            "  ", " "
                        )
                        ctx.logger.info(
                            f"Found Headquarters (text method): {headquarters_text}"
                        )
                        scraped_data["country"] = self._extract_country_from_text(
                            headquarters_text, ctx
                        )
                        if scraped_data["country"]:
                            headquarters_found = True
                        break

            # Fallback: Look for "Operates In" if headquarters didn't work
            if not scraped_data["country"]:
                all_text = soup.get_text(separator="|", strip=True)
                operates_pattern = r"Operates In\|([^|]+)"
                match = re.search(operates_pattern, all_text, re.IGNORECASE)
                if match:
                    operates_text = match.group(1).strip()
                    scraped_data["country"] = self._extract_country_from_text(
                        operates_text, ctx
                    )
                    if scraped_data["country"]:
                        pass

            # Default if no country found
            if not scraped_data["country"]:
                scraped_data["country"] = "Unknown"

            return scraped_data

        except Exception as e:
            ctx.logger.error(f"Error scraping B Corp page: {e}")
            return {
                "supplier_name": supplier_name,
                "country": "Unknown",
                "url": b_corp_url,
                "error": str(e),
            }

        finally:
            if driver:
                try:
                    driver.quit()
                    ctx.logger.info("WebDriver cleaned up")
                except Exception as e:
                    ctx.logger.warning(f"Error closing driver: {e}")

    async def initialize(self, ctx: Context) -> bool:
        """Initialize the financial analysis system with Google Gemini LLM"""
        try:
            # Skip test call to save tokens
            ctx.logger.info("Financial Analysis System initialized")
            self.initialized = True
            return True
        except Exception as e:
            ctx.logger.error(f"Failed to initialize: {e}")
            return False

    def _query_llm_for_analysis(
        self,
        ctx: Context,
        prompt: str,
        supplier_name: str,
        supplier_country: str,
        user_country: str,
    ) -> str:
        """Query Google Gemini LLM for detailed financial analysis with caching"""
        try:
            # Build the full prompt with system context
            full_prompt = f"""You are a financial risk analyst specializing in international trade and supply chain finance. Provide concise, factual analysis.

{prompt}"""

            response = self.gemini_model.generate_content(full_prompt)
            llm_response = response.text
            return llm_response

        except Exception as e:
            return f"Financial analysis for {supplier_name} ({supplier_country}) trading with {user_country}."

    async def analyze_financial_risk(
        self,
        ctx: Context,
        supplier_name: str,
        industry: str,
        b_corp_url: str,
        user_country: str = "United States",
    ) -> Dict[str, Any]:
        """
        Analyze financial risk based on trade relationship between user's country and supplier's country.

        Args:
            ctx: Context for logging
            supplier_name: Name of the supplier
            industry: Industry/sector
            b_corp_url: B Corp profile URL to scrape country from
            user_country: User's country (default: United States)

        Returns:
            Dictionary with financial_score, financial_details, risk_factors, supplier_country, user_country
        """
        if not self.initialized:
            ctx.logger.error("System not initialized")
            return self._generate_fallback_response(
                supplier_name, "Unknown", user_country
            )

        try:
            # Step 1: Scrape B Corp page to extract supplier country
            if not b_corp_url:
                ctx.logger.error("No B Corp URL provided")
                return self._generate_fallback_response(
                    supplier_name, "Unknown", user_country
                )

            ctx.logger.info(f"Scraping B Corp page for supplier country...")
            scraped_data = await self.scrape_bcorp_country(
                ctx, supplier_name, b_corp_url
            )

            if "error" in scraped_data:
                ctx.logger.error(f"Scraping failed: {scraped_data.get('error')}")
                return self._generate_fallback_response(
                    supplier_name, "Unknown", user_country
                )

            supplier_country = scraped_data.get("country", "Unknown")
            ctx.logger.info(f"Supplier Country: {supplier_country}")
            ctx.logger.info(f"User Country: {user_country}")

            # Step 2: Analyze trade relationship (rule-based scoring)
            analysis = analyze_country_financial_risk(
                supplier_country=supplier_country,
                user_country=user_country,
                ctx=ctx,
            )

            # Step 3: Generate prompt and query LLM for detailed analysis
            prompt = finance_prompt(
                supplier_name, industry, supplier_country, user_country
            )
            ctx.logger.info(
                f"Generated finance prompt for {user_country}-{supplier_country} trade"
            )

            # Query Gemini LLM for detailed reasoning
            llm_analysis = self._query_llm_for_analysis(
                ctx=ctx,
                prompt=prompt,
                supplier_name=supplier_name,
                supplier_country=supplier_country,
                user_country=user_country,
            )

            # Build financial details combining rule-based score with LLM analysis
            if supplier_country.lower() == user_country.lower():
                financial_details = (
                    f"Domestic trade analysis for {supplier_name}. "
                    f"Both buyer and supplier are in {user_country}. "
                    f"No import tariffs or international trade barriers apply. "
                    f"Score: {analysis['financial_score']}/100. "
                    f"\n\nLLM Analysis: {llm_analysis}"
                )
            else:
                financial_details = (
                    f"International trade analysis for {supplier_name} from {supplier_country}. "
                    f"Buyer located in {user_country}. "
                    f"Trade relationship: {analysis.get('trade_relationship', 'Standard international')}. "
                    f"Score: {analysis['financial_score']}/100. "
                    f"\n\nLLM Analysis: {llm_analysis}"
                )

            return {
                "financial_score": analysis["financial_score"],
                "financial_details": financial_details,
                "risk_factors": analysis["risk_factors"],
                "supplier_country": supplier_country,
                "user_country": user_country,
            }

        except Exception as e:
            ctx.logger.error(f"Analysis failed: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(
                supplier_name, "Unknown", user_country
            )

    def _generate_fallback_response(
        self, supplier_name: str, supplier_country: str, user_country: str
    ) -> Dict[str, Any]:
        """Generate fallback response when analysis fails"""
        return {
            "financial_score": 50.0,
            "financial_details": f"Unable to analyze trade relationship between {user_country} and {supplier_country}",
            "risk_factors": ["Analysis unavailable"],
            "supplier_country": supplier_country,
            "user_country": user_country,
        }


# ============================================================================
# COUNTRY-SPECIFIC ANALYSIS
# ============================================================================


def analyze_country_financial_risk(
    supplier_country: str,
    user_country: str,
    ctx: Context,
) -> Dict[str, Any]:
    """
    Analyze financial risk based on trade relationship between user's country and supplier's country.

    Scoring guidance:
    - 75-100: minimal financial risks (same country, free trade agreements, same trade bloc)
    - 60-74: moderate risk (some tariffs, different trade blocs but stable)
    - 40-59: significant risk (high tariffs, trade tensions)
    - Below 40: high risk (sanctions, severe trade restrictions)

    Args:
        supplier_country: Country where supplier operates
        user_country: Country where user/buyer is located
        ctx: Context for logging

    Returns:
        Dictionary with financial_score, risk_factors, trade_relationship
    """
    supplier_lower = supplier_country.lower() if supplier_country else ""
    user_lower = user_country.lower() if user_country else ""

    # Start with moderate baseline score
    score = 65.0
    risk_factors = []
    trade_relationship = "Standard international trade"

    # Define trade blocs and agreements
    eu_countries = [
        "spain",
        "germany",
        "france",
        "italy",
        "netherlands",
        "belgium",
        "portugal",
        "greece",
        "ireland",
        "austria",
        "poland",
        "sweden",
        "denmark",
        "finland",
        "czech republic",
        "hungary",
        "romania",
        "bulgaria",
        "croatia",
        "slovakia",
        "slovenia",
        "estonia",
        "latvia",
        "lithuania",
        "luxembourg",
        "malta",
        "cyprus",
    ]

    usmca_countries = ["united states", "usa", "canada", "mexico"]

    latin_american_countries = [
        "argentina",
        "brazil",
        "chile",
        "colombia",
        "peru",
        "uruguay",
        "venezuela",
        "ecuador",
        "bolivia",
        "paraguay",
    ]

    high_tariff_countries = ["china", "russia"]

    sanctioned_countries = ["russia", "north korea", "iran", "syria", "cuba"]

    # Normalize country names
    user_is_eu = any(c in user_lower for c in eu_countries)
    user_is_usmca = any(c in user_lower for c in usmca_countries)

    supplier_is_eu = any(c in supplier_lower for c in eu_countries)
    supplier_is_usmca = any(c in supplier_lower for c in usmca_countries)
    supplier_is_latin = any(c in supplier_lower for c in latin_american_countries)
    supplier_is_high_tariff = any(c in supplier_lower for c in high_tariff_countries)
    supplier_is_sanctioned = any(c in supplier_lower for c in sanctioned_countries)

    # CASE 1: Same country (domestic trade)
    if (
        supplier_lower == user_lower
        or ("united states" in supplier_lower and "united states" in user_lower)
        or ("usa" in supplier_lower and "usa" in user_lower)
    ):
        score = 90.0
        risk_factors = ["no_import_tariffs", "domestic_supply_chain"]
        trade_relationship = "Domestic trade"
        return {
            "financial_score": score,
            "risk_factors": risk_factors,
            "trade_relationship": trade_relationship,
        }

    # CASE 2: Sanctioned country supplier
    if supplier_is_sanctioned:
        score = 20.0
        risk_factors = ["international_sanctions", "trade_restrictions", "high_risk"]
        trade_relationship = "Sanctioned country - HIGH RISK"
        return {
            "financial_score": score,
            "risk_factors": risk_factors,
            "trade_relationship": trade_relationship,
        }

    # CASE 3: EU to EU trade (no tariffs within EU)
    if user_is_eu and supplier_is_eu:
        score = 88.0
        risk_factors = ["eu_single_market", "no_tariffs", "free_movement"]
        trade_relationship = "EU Single Market - Free Trade"
        return {
            "financial_score": score,
            "risk_factors": risk_factors,
            "trade_relationship": trade_relationship,
        }

    # CASE 4: USMCA trade (US-Canada-Mexico free trade)
    if user_is_usmca and supplier_is_usmca:
        score = 85.0
        risk_factors = ["usmca_agreement", "low_tariffs", "integrated_supply_chain"]
        trade_relationship = "USMCA Free Trade Agreement"
        return {
            "financial_score": score,
            "risk_factors": risk_factors,
            "trade_relationship": trade_relationship,
        }

    # CASE 5: High tariff country supplier (China, etc.)
    if supplier_is_high_tariff:
        if "china" in supplier_lower:
            score = 50.0
            risk_factors = [
                "elevated_tariffs",
                "trade_tensions",
                "supply_chain_considerations",
            ]
            trade_relationship = f"High tariff environment ({user_country}-China)"
        else:
            score = 45.0
            risk_factors = ["trade_restrictions", "economic_considerations"]
            trade_relationship = "Elevated trade restrictions"
        return {
            "financial_score": score,
            "risk_factors": risk_factors,
            "trade_relationship": trade_relationship,
        }

    # CASE 6: US/USMCA user importing from EU
    if user_is_usmca and supplier_is_eu:
        score = 72.0
        risk_factors = [
            "standard_tariffs",
            "stable_trade_relationship",
            "currency_exchange",
        ]
        trade_relationship = "USMCA-EU Standard Trade"
        return {
            "financial_score": score,
            "risk_factors": risk_factors,
            "trade_relationship": trade_relationship,
        }

    # CASE 7: EU user importing from US/USMCA
    if user_is_eu and supplier_is_usmca:
        score = 72.0
        risk_factors = [
            "standard_tariffs",
            "stable_trade_relationship",
            "currency_exchange",
        ]
        trade_relationship = "EU-USMCA Standard Trade"
        return {
            "financial_score": score,
            "risk_factors": risk_factors,
            "trade_relationship": trade_relationship,
        }

    # CASE 8: Latin American supplier
    if supplier_is_latin:
        # Check for specific trade agreements
        if "chile" in supplier_lower or "peru" in supplier_lower:
            score = 68.0
            risk_factors = ["bilateral_trade_agreement", "emerging_market"]
            trade_relationship = "Bilateral trade agreement"
        elif "argentina" in supplier_lower:
            score = 58.0
            risk_factors = ["economic_volatility", "currency_risk", "emerging_market"]
            trade_relationship = "Emerging market with volatility"
        else:
            score = 62.0
            risk_factors = ["emerging_market", "currency_considerations"]
            trade_relationship = "Latin American trade"
            return {
                "financial_score": score,
                "risk_factors": risk_factors,
                "trade_relationship": trade_relationship,
            }

    # CASE 9: Developed economies (Japan, South Korea, Australia, Singapore)
    developed_economies = [
        "japan",
        "south korea",
        "australia",
        "singapore",
        "new zealand",
    ]
    if any(c in supplier_lower for c in developed_economies):
        score = 74.0
        risk_factors = ["stable_economy", "reliable_trade_partner", "standard_tariffs"]
        trade_relationship = "Developed economy - stable trade"
        return {
            "financial_score": score,
            "risk_factors": risk_factors,
            "trade_relationship": trade_relationship,
        }

    # CASE 10: Default - standard international trade
    score = 64.0
    risk_factors = [
        "standard_international_tariffs",
        "currency_exchange",
        "import_duties",
    ]
    trade_relationship = f"Standard international ({user_country}-{supplier_country})"

    return {
        "financial_score": max(0.0, min(100.0, score)),
        "risk_factors": risk_factors,
        "trade_relationship": trade_relationship,
    }


# ============================================================================
# LANGGRAPH WORKFLOW NODES
# ============================================================================


def financial_check_node(
    state: SupplierWorkflowState,
    analysis_system: FinancialAnalysisSystem,
    ctx: Context,
) -> SupplierWorkflowState:
    """Node 1: Run financial risk check"""
    try:
        supplier_name = state.get("supplier_name", "Unknown")
        b_corp_url = state.get("b_corp_profile_url", "")
        user_country = state.get("user_country", "United States")

        if not b_corp_url:
            ctx.logger.error("No B Corp URL provided for financial analysis")
            state["current_step"] = "financial_check_failed"
            state["error_message"] = "No B Corp URL available"
            state["financial_score"] = 0.0
            return state

        # Run analysis synchronously
        import asyncio
        import concurrent.futures

        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    analysis_system.analyze_financial_risk(
                        ctx=ctx,
                        supplier_name=supplier_name,
                        industry=state.get("industry", ""),
                        b_corp_url=b_corp_url,
                        user_country=user_country,
                    )
                )
            finally:
                new_loop.close()

        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(run_async)
                result = future.result(timeout=180)  # 3 min timeout for scraping + LLM
        except RuntimeError:
            result = asyncio.run(
                analysis_system.analyze_financial_risk(
                    ctx=ctx,
                    supplier_name=supplier_name,
                    industry=state.get("industry", ""),
                    b_corp_url=b_corp_url,
                    user_country=user_country,
                )
            )

        # Update state with results
        state["financial_score"] = result.get("financial_score", 0.0)
        state["financial_info"] = result.get("financial_details", "N/A")
        state["risk_factors"] = result.get("risk_factors", [])
        state["supplier_country"] = result.get("supplier_country", "Unknown")
        state["user_country"] = result.get("user_country", user_country)
        state["current_step"] = "financial_check_complete"

        ctx.logger.info(f"Financial score: {state['financial_score']}/100")
        return state

    except Exception as e:
        ctx.logger.error(f"Error in financial check: {e}")
        state["current_step"] = "financial_check_failed"
        state["error_message"] = f"Financial check failed: {str(e)}"
        state["financial_score"] = 0.0
        state["risk_factors"] = [
            f"Analysis failed: {str(e)}"
        ]  # Set to prevent NoneType error
        state["financial_info"] = "Analysis could not be completed due to an error"
        return state


def financial_router(
    state: SupplierWorkflowState,
) -> Literal["success_node", "error_node"]:
    """
    Node 2: Conditional routing based on financial score.

    Routes to:
    - "success_node" if score >= 60 (APPROVED)
    - "error_node" if score < 60 OR error occurred (REJECTED)
    """
    financial_score = state.get("financial_score", 0.0)
    error_message = state.get("error_message")

    if error_message:
        return "error_node"

    if financial_score >= 60:
        return "success_node"
    else:
        state["error_message"] = (
            f"Financial score {financial_score}/100 below threshold (60)."
        )
        return "error_node"


def success_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """Node 3a: Success path - Financial requirements met."""
    state["current_step"] = "financial_approved"
    state["should_continue"] = False
    return state


def error_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """Node 3b: Error path - Financial requirements not met."""
    state["current_step"] = "financial_rejected"
    state["should_continue"] = False
    return state


# ============================================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================================


def build_financial_workflow(
    analysis_system: FinancialAnalysisSystem, ctx: Context
) -> StateGraph:
    """Build the LangGraph financial workflow"""
    workflow = StateGraph(SupplierWorkflowState)

    # Add nodes
    workflow.add_node(
        "financial_check",
        lambda state: financial_check_node(state, analysis_system, ctx),
    )
    workflow.add_node("success_node", lambda state: success_node(state, ctx))
    workflow.add_node("error_node", lambda state: error_node(state, ctx))

    # Add edges
    workflow.add_edge(START, "financial_check")
    workflow.add_conditional_edges(
        "financial_check",
        financial_router,
        {
            "success_node": "success_node",
            "error_node": "error_node",
        },
    )
    workflow.add_edge("success_node", END)
    workflow.add_edge("error_node", END)

    return workflow.compile()


# Backwards compatibility alias
FinancialRAGSystem = FinancialAnalysisSystem
