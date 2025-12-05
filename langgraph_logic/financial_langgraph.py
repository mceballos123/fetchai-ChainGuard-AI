from typing import Dict, Any, List, Optional, Literal
from models.financial import FinancialRequest, FinancialResponse
from langgraph_logic.state_schemas import SupplierWorkflowState
import os
import json
import re
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from dotenv import load_dotenv
from uagents import Context
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from pinecone import Pinecone, ServerlessSpec
from ollama import Client
from llama_index.llms.ollama import Ollama
from langgraph.graph import StateGraph, START, END
from prompts.finance_prompt import finance_prompt

load_dotenv()

PINECONE_INDEX_NAME_FINANCIAL = os.getenv(
    "PINECONE_INDEX_NAME_FINANCIAL", "financial-risk-index"
)
EMBEDDING_DIMENSION = 768


class FinancialRAGSystem:
    """RAG system for financial risk analysis - extracts country from B Corp Headquarters section only"""

    def __init__(self):
        self.initialized = False
        self.index = None
        self.query_engine = None

        # Initialize Ollama Client (direct connection)
        self.ollama_client = Client(host="http://127.0.0.1:11434", timeout=300)
        self.llm_type = "ollama"
        self.llm_model = "llama3.2:1b"

        # Initialize Ollama embeddings (still using LlamaIndex for embeddings)
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Configure Settings for LLamaIndex (embeddings only, BEFORE any Pinecone initialization!)
        Settings.embed_model = self.embed_model
        Settings.llm = Ollama(model=self.llm_model, request_timeout=300)

        print(f"Settings: {Settings}")
        print(f"Setting up Financial RAG System: {Settings.llm}")

        self.documents = []
        self.scraped_tariff_data = {}

    def _extract_country_from_text(self, text: str, ctx: Context) -> Optional[str]:
        """
        Extract country name from text content.

        Args:
            text: Text to search for country
            ctx: Context for logging

        Returns:
            Country name or None
        """
        print(f"Text on line 75: {text}")
        print(f"Known countries on line 76: {known_countries}")
        if not text:
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

        text_lower = text.lower()

        # Check for each known country
        for country in known_countries:
            if country.lower() in text_lower:
                # Special handling for USA -> United States
                if country == "USA":
                    return "United States"
                return country

        # If no known country found, try to extract from comma-separated location
        parts = text.split(",")
        if len(parts) > 1:
            potential_country = parts[-1].strip()
            if len(potential_country) > 2 and potential_country[0].isupper():
                return potential_country

        return None

    async def scrape_bcorp_financial_page(
        self, ctx: Context, supplier_name: str, b_corp_url: str
    ) -> Dict[str, str]:
        """
        Scrape B Corp page ONLY to extract country from Headquarters section.

        Args:
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

            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)

            ctx.logger.info(f"Loading B Corp page: {b_corp_url}")
            driver.get(b_corp_url)
            time.sleep(5)

            scraped_data = {
                "supplier_name": supplier_name,
                "country": "",
                "url": b_corp_url,
            }

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # ONLY extract headquarters to get country - nothing else
            all_text = soup.get_text(separator="|", strip=True)

            # Look for Headquarters section (e.g., "Headquarters|Catalonia, Spain")
            hq_patterns = [
                r"Headquarters\|([^|]+)",
                r"Headquarters[:\s]+([^|]+)",
            ]

            for pattern in hq_patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match:
                    headquarters_text = match.group(1).strip()
                    ctx.logger.info(f"Found Headquarters: {headquarters_text}")

                    # Extract country from headquarters (e.g., "Catalonia, Spain" -> "Spain")
                    scraped_data["country"] = self._extract_country_from_text(
                        headquarters_text, ctx
                    )
                    if scraped_data["country"]:
                        ctx.logger.info(
                            f"✅ Country extracted: {scraped_data['country']}"
                        )
                    break

            # Fallback: Look for "Operates In" if headquarters didn't work
            print(f"Scraped data on line 221: {scraped_data}")
            print(f"All text on line 222: {all_text}")
            if not scraped_data["country"]:
                operates_pattern = r"Operates In\|([^|]+)"
                match = re.search(operates_pattern, all_text, re.IGNORECASE)
                if match:
                    operates_text = match.group(1).strip()
                    scraped_data["country"] = self._extract_country_from_text(
                        operates_text, ctx
                    )
                    if scraped_data["country"]:
                        ctx.logger.info(
                            f"✅ Country from 'Operates In': {scraped_data['country']}"
                        )

            # Default if no country found
            if not scraped_data["country"]:
                scraped_data["country"] = "Unknown"
                ctx.logger.warning(
                    "⚠️ Could not determine country - defaulting to Unknown"
                )

            ctx.logger.info(f"Country extraction complete: {scraped_data['country']}")
            return scraped_data

        except Exception as e:
            ctx.logger.error(f"Error scraping B Corp page: {e}")
            import traceback

            traceback.print_exc()

            return {
                "supplier_name": supplier_name,
                "country": "Unknown",
                "url": b_corp_url,
                "error": str(e),
            }

        finally:
            # Always cleanup driver in finally block
            if driver:
                try:
                    driver.quit()
                    ctx.logger.info("B Corp financial scraping WebDriver cleaned up")
                except Exception as e:
                    ctx.logger.warning(f"Error closing driver: {e}")

    async def initialize(self, ctx: Context) -> bool:
        try:
            if not await self._setup_pinecone(ctx):
                return False

            self.initialized = True
            return True

        except Exception as e:
            ctx.logger.error(f"Failed to initialize RAG system: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _setup_pinecone(self, ctx: Context) -> bool:
        """Setup Pinecone vector store for financial data"""
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                ctx.logger.error("PINECONE_API_KEY not set")
                return False

            # Initialize Pinecone
            pc = Pinecone(api_key=pinecone_api_key)

            # Check if index exists
            existing_indexes = [idx.name for idx in pc.list_indexes()]

            if PINECONE_INDEX_NAME_FINANCIAL not in existing_indexes:
                ctx.logger.info(
                    f"Creating new Pinecone index: {PINECONE_INDEX_NAME_FINANCIAL}"
                )
                pc.create_index(
                    name=PINECONE_INDEX_NAME_FINANCIAL,
                    dimension=int(EMBEDDING_DIMENSION),
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                ctx.logger.info(
                    f"Using existing Pinecone index: {PINECONE_INDEX_NAME_FINANCIAL}"
                )

            # Get Pinecone index
            pinecone_index = pc.Index(PINECONE_INDEX_NAME_FINANCIAL)

            # Create LlamaIndex vector store
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

            # Create index from vector store
            self.index = VectorStoreIndex.from_vector_store(vector_store)

            # Create query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=1, response_mode="compact", verbose=True
            )

            return True

        except Exception as e:
            ctx.logger.error(f"Error setting up Pinecone: {e}")
            return False

    async def _index_scraped_document(self, ctx: Context, document: Document) -> bool:
        """Index a single scraped document in Pinecone"""
        try:
            if not self.index:
                ctx.logger.error("Index not initialized")
                return False

            # Parse document into nodes (chunks)
            parser = SimpleNodeParser.from_defaults(
                chunk_size=512,
                chunk_overlap=20,
            )

            nodes = parser.get_nodes_from_documents([document])

            # Index nodes in Pinecone
            for node in nodes:
                try:
                    self.index.insert_nodes([node])
                except Exception as e:
                    ctx.logger.warning(f"Error indexing node: {e}")

            return True

        except Exception as e:
            ctx.logger.error(f"Error indexing document: {e}")
            return False

    async def query_financial_documents(
        self,
        ctx: Context,
        supplier_name: str,
        industry: str,
        b_corp_url: str = "",
    ) -> Dict[str, Any]:
        """
        Query RAG system for financial risk analysis - extract country from Headquarters only.

        Process:
        1. Scrape B Corp page ONLY for Headquarters section to extract country
        2. Apply country-specific financial risk analysis based on US trade relationships
        3. Create simple document with country and analysis
        4. Store in Pinecone and query for financial risk
        5. Return structured financial data

        Returns:
        {
            "financial_score": float (0-100, higher is better/lower risk),
            "financial_details": str,
            "risk_factors": List[str],
            "retrieved_context": str
        }
        """
        print(
            f"Querying financial documents on line 385: {supplier_name}, {industry}, {b_corp_url}"
        )
        if not self.initialized:
            ctx.logger.error("RAG system not initialized")
            return self._generate_fallback_response(supplier_name, "Unknown")

        try:
            # Step 1: Scrape B Corp page to extract country from Headquarters section ONLY

            if not b_corp_url:
                ctx.logger.error("No B Corp URL provided for financial analysis")
                return self._generate_fallback_response(supplier_name, "Unknown")

            # Scrape B Corp page to extract country from Headquarters section
            ctx.logger.info(
                f"Scraping B Corp Headquarters section to extract country..."
            )
            scraped_data = await self.scrape_bcorp_financial_page(
                ctx, supplier_name, b_corp_url
            )

            if "error" in scraped_data:
                ctx.logger.error(f"B Corp scraping failed: {scraped_data.get('error')}")
                return self._generate_fallback_response(supplier_name, "Unknown")

            # Extract country from scraped data
            country = scraped_data.get("country", "Unknown")
            ctx.logger.info(f"📍 Country extracted from B Corp page: {country}")

            # Step 2: Apply country-specific financial risk analysis
            country_analysis = analyze_country_financial_risk(
                scraped_data, country, ctx
            )

            # Step 3: Create simple document with country and analysis
            financial_text = f"""
            Supplier: {supplier_name}
            Operating Country: {country}
            Industry: {industry}
            
            FINANCIAL RISK ANALYSIS:
            Based on US trade relationship with {country}:
            Financial Score: {country_analysis['financial_score']}/100
            Risk Factors: {', '.join(country_analysis['risk_factors'])}
            """

            print(f"Financial text on line 430: {financial_text}")

            # Create Document object
            doc = Document(
                text=financial_text,
                metadata={
                    "supplier_name": supplier_name,
                    "country": country,
                    "source": "bcorp_financial_analysis",
                    "url": b_corp_url or "",
                },
            )

            print(f"Document on line 443: {doc}")

            # Step 4: Index the document in Pinecone
            await self._index_scraped_document(ctx, doc)

            # Step 5: Query RAG system
            query = finance_prompt(supplier_name, industry, country)

            # Query using the indexed data
            rag_response = self.query_engine.query(query)
            response_text = str(rag_response)

            # Step 6: Parse response and merge with country-specific analysis
            result = await self._parse_rag_response(
                ctx, response_text, supplier_name, country
            )

            # Use country-specific score and risk factors
            result["financial_score"] = country_analysis["financial_score"]
            result["risk_factors"] = country_analysis["risk_factors"]

            # Build country-specific financial details
            is_us = "united states" in country.lower() or "usa" in country.lower()
            if is_us:
                result["financial_details"] = (
                    f"US domestic supplier analysis for {supplier_name}. No import tariffs apply. "
                    f"Analysis based on US economic conditions."
                )
            else:
                result["financial_details"] = (
                    f"International supplier analysis for {supplier_name} from {country}. "
                    f"Financial risk evaluated based on US-{country} trade relationship. "
                    f"Score: {country_analysis['financial_score']}/100."
                )

            return result

        except Exception as e:
            ctx.logger.error(f"Query failed: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name, "Unknown")

    async def _parse_rag_response(
        self, ctx: Context, response_text: str, supplier_name: str, country: str
    ) -> Dict[str, Any]:
        """Parse LLM response and extract structured financial data"""
        try:
            # Initialize defaults
            financial_score = 50.0
            financial_details = "Insufficient financial data"
            risk_factors = []

            # Parse response
            lines = response_text.split("\n")

            print(f"Lines on line 499: {lines}")

            for i, line in enumerate(lines):
                line_lower = line.lower()

                if "financial_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        financial_score = max(0, min(100, float(score_str)))
                    except (ValueError, IndexError):
                        pass

                elif "financial_details:" in line_lower:
                    # Extract content after "financial_details:"
                    content = line.split(":", 1)[-1].strip()

                    # If this line is minimal, look ahead for more content
                    if not content or len(content) < 30:
                        for next_line in lines[i + 1 :]:
                            next_line_lower = next_line.lower()
                            # Stop if we hit another field marker
                            if ":" in next_line and any(
                                field in next_line_lower
                                for field in ["risk_factors", "financial_score"]
                            ):
                                break
                            if next_line.strip():
                                content += " " + next_line.strip()
                            else:
                                break

                    # Fix concatenation issues
                    content = re.sub(r"(\w)([A-Z][a-z])", r"\1 \2", content)
                    content = re.sub(r"\s+", " ", content).strip()

                    if content:
                        financial_details = content

                elif "risk_factors:" in line_lower:
                    factors_str = line.split(":", 1)[-1].strip().lower()
                    if (
                        factors_str
                        and "minimal" not in factors_str
                        and "none" not in factors_str
                    ):
                        risk_factors = [f.strip() for f in factors_str.split(",")]

            print(f"Financial score on line 547: {financial_score}")
            print(f"Financial details on line 548: {financial_details}")
            print(f"Risk factors on line 549: {risk_factors}")
            print(f"Retrieved context on line 550: {response_text}")

            return {
                "financial_score": float(financial_score),
                "financial_details": financial_details
                or "No financial information available",
                "risk_factors": risk_factors,
                "retrieved_context": response_text,
            }

        except Exception as e:
            ctx.logger.error(f"Error parsing RAG response: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name, "Unknown")

    def _generate_fallback_response(
        self, supplier_name: str, country: str = "Unknown"
    ) -> Dict[str, Any]:
        """Generate fallback response when RAG is unavailable"""
        return {
            "financial_score": 50.0,  # Neutral score
            "financial_details": f"Unable to retrieve tariff/inflation information for {country}",
            "risk_factors": ["Data retrieval unavailable"],
            "retrieved_context": "Fallback mode - RAG system unavailable",
        }

    def query_financial_documents_sync(
        self,
        supplier_name: str,
        industry: str,
        b_corp_url: str = "",
    ) -> Dict[str, Any]:
        """Synchronous wrapper for query_financial_documents for use in LangGraph nodes"""
        import asyncio

        # Create a mock context for sync execution
        class MockContext:
            def __init__(self):
                self.logs = []

            class Logger:
                def __init__(self, parent):
                    self.parent = parent

                def info(self, msg):
                    pass

                def error(self, msg):
                    pass

                def warning(self, msg):
                    pass

            @property
            def logger(self):
                return self.Logger(self)

        try:
            mock_ctx = MockContext()
            try:
                # Try to get current running loop
                loop = asyncio.get_running_loop()
                # If we're here, a loop is running, use a different approach
                import concurrent.futures

                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.query_financial_documents(
                                ctx=mock_ctx,
                                supplier_name=supplier_name,
                                industry=industry,
                                b_corp_url=b_corp_url,
                            )
                        )
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(run_async)
                    result = future.result(timeout=600)  # 10 minute timeout
                    return result
            except RuntimeError:
                # No event loop running, use asyncio.run()
                result = asyncio.run(
                    self.query_financial_documents(
                        ctx=mock_ctx,
                        supplier_name=supplier_name,
                        industry=industry,
                        b_corp_url=b_corp_url,
                    )
                )
                return result
        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name, "Unknown")


def financial_check_node(
    state: SupplierWorkflowState, rag_system: FinancialRAGSystem, ctx: Context
) -> SupplierWorkflowState:
    """
    Node 1: Run financial risk check - extract country from Headquarters only.

    Process:
    1. Get supplier name and B Corp URL from state
    2. Scrape B Corp page ONLY for Headquarters section to extract country
    3. Apply country-specific financial risk analysis based on US trade relationships
    4. Store in Pinecone and analyze with RAG
    5. Return financial score and details

    Input: supplier_name, industry, b_corp_profile_url
    Output: financial_score, financial_info, risk_factors
    """
    try:
        supplier_name = state.get("supplier_name", "Unknown")
        b_corp_url = state.get("b_corp_profile_url", "")

        if not b_corp_url:
            ctx.logger.error("No B Corp URL provided for financial analysis")
            state["current_step"] = "financial_check_failed"
            state["error_message"] = "No B Corp URL available to extract country"
            state["financial_score"] = 0.0
            return state

        # Run RAG query for financial analysis (extracts country from Headquarters ONLY)
        ctx.logger.info(
            f"Financial analysis will scrape B Corp Headquarters section to extract country..."
        )
        rag_result = rag_system.query_financial_documents_sync(
            supplier_name=supplier_name,
            industry=state.get("industry", ""),
            b_corp_url=b_corp_url,
        )

        # Update state with financial results including risk_factors
        state["financial_score"] = rag_result.get("financial_score", 0.0)
        state["financial_info"] = rag_result.get("financial_details", "N/A")
        state["risk_factors"] = rag_result.get("risk_factors", [])
        state["current_step"] = "financial_check_complete"

        ctx.logger.info(f"Financial analysis complete")
        ctx.logger.info(f"Score: {state['financial_score']}/100")
        ctx.logger.info(f"Risk factors: {state['risk_factors']}")

        return state

    except Exception as e:
        ctx.logger.error(f"Error in financial check: {e}")
        import traceback

        traceback.print_exc()

        state["current_step"] = "financial_check_failed"
        state["error_message"] = f"Financial check failed: {str(e)}"
        state["financial_score"] = 0.0
        return state


def financial_router(
    state: SupplierWorkflowState,
) -> Literal["success_node", "error_node"]:
    """
    Node 2: Conditional routing based on financial score.

    Routes to:
    - "success_node" if score >= 70 (APPROVED - lower risk)
    - "error_node" if score < 70 OR error occurred (REJECTED - higher risk)
    """
    financial_score = state.get("financial_score", 0.0)
    error_message = state.get("error_message")

    # Check for errors first
    if error_message:
        return "error_node"

    # Check financial threshold (60+ means acceptable financial health)
    if financial_score >= 60:
        return "success_node"
    else:
        state["error_message"] = (
            f"Financial score {financial_score}/100 below threshold (60). Financial risk too high."
        )
        return "error_node"


def success_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3a: Success path - Country meets financial requirements.
    """
    state["current_step"] = "financial_approved"
    state["should_continue"] = False

    return state


def error_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3b: Error path - Country does not meet requirements.
    """
    state["current_step"] = "financial_rejected"
    state["should_continue"] = False

    return state


# ============================================================================
# COUNTRY-SPECIFIC ANALYSIS HELPERS
# ============================================================================


def analyze_country_financial_risk(
    scraped_data: Dict[str, Any], country: str, ctx: Context
) -> Dict[str, Any]:
    """
    Analyze financial risk based on country and US trade relationships.

    Follows finance_prompt scoring guidance:
    - 75-100: minimal financial risks (low tariffs, stable inflation)
    - 60-74: moderate risk (some tariffs, manageable inflation)
    - 40-59: significant risk (high tariffs or inflation concerns)
    - Below 40: high risk (severe tariffs, economic instability)

    Be fair in evaluation - don't be overly harsh on countries with some trade restrictions.
    Consider: tariff rates, trade agreements, currency stability, US relationship
    """
    country_lower = country.lower() if country else ""
    full_content = scraped_data.get("full_content", "").lower()

    # Start with moderate baseline score (per prompt guidance)
    score = 65.0  # Start in moderate range
    risk_factors = []

    # Check if supplier is in the United States
    is_us_supplier = (
        "united states" in country_lower
        or "usa" in country_lower
        or "u.s." in country_lower
    )

    # Define trade relationship categories

    # Countries with US Free Trade Agreements (lower tariff risk)
    usmca_countries = ["mexico", "canada"]  # USMCA (formerly NAFTA)

    # Latin American countries (emerging market risks)
    latin_american_countries = [
        "argentina",
        "brazil",
        "chile",
        "colombia",
        "mexico",
        "peru",
        "uruguay",
        "venezuela",
        "ecuador",
        "bolivia",
        "paraguay",
    ]

    # EU countries (generally favorable trade)
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
    ]

    # High tariff risk countries
    high_tariff_countries = ["china", "russia"]

    is_latin_american = any(c in country_lower for c in latin_american_countries)
    is_usmca = any(c in country_lower for c in usmca_countries)
    is_eu = any(c in country_lower for c in eu_countries)
    is_high_tariff = any(c in country_lower for c in high_tariff_countries)

    if is_us_supplier:
        # US Domestic Supplier Analysis
        # Score: 85-95 (minimal financial risks per prompt)
        ctx.logger.info(
            f"📍 US Supplier - No import tariffs, domestic economic analysis"
        )

        score = 88.0  # Minimal risk range (75-100)
        risk_factors.append("no_import_tariffs")
        risk_factors.append("domestic_supply_chain")

        # US suppliers - no tariff concerns, only domestic economic factors
        # B Corp certification indicates stable operations

    elif is_high_tariff:
        # High Tariff Countries (China, Russia, etc.)
        # Score: 40-55 (significant to high risk per prompt)
        ctx.logger.info(f"📍 {country} Supplier - Elevated tariff environment")

        if "china" in country_lower:
            score = 48.0  # Significant risk range (40-59)
            risk_factors.append("higher_us_tariffs_applicable")
            risk_factors.append("trade_relationship_considerations")
            # Note: Being fair per prompt - not subtracting 40 points
        elif "russia" in country_lower:
            score = 35.0  # High risk (below 40) due to sanctions
            risk_factors.append("international_sanctions")
            risk_factors.append("supply_chain_complexity")

    elif is_usmca:
        # USMCA Countries (Mexico, Canada) - Free Trade Agreement
        # Score: 80-90 (minimal financial risks per prompt)
        ctx.logger.info(f"📍 {country} Supplier - USMCA free trade partner")

        score = 82.0  # Minimal risk range (75-100)
        risk_factors.append("usmca_free_trade_agreement")
        risk_factors.append("low_to_no_tariffs")

        if "mexico" in country_lower:
            risk_factors.append("strong_trade_partner")
        elif "canada" in country_lower:
            risk_factors.append("stable_economic_partner")

    elif is_eu:
        # European Union Countries
        # Score: 70-80 (minimal to moderate risk per prompt)
        ctx.logger.info(f"📍 {country} (EU) Supplier - European trade analysis")

        score = 72.0  # Moderate to minimal range (60-74 / 75-100)
        risk_factors.append("eu_trade_relationship")
        risk_factors.append("standard_import_duties")

        # Major EU trading partners
        if any(c in country_lower for c in ["spain", "germany", "france", "italy"]):
            score = 76.0  # Bump to minimal risk
            risk_factors.append("major_trading_partner")

    elif is_latin_american:
        # Latin American Countries (non-USMCA)
        # Score: 55-70 (moderate to significant risk per prompt)
        ctx.logger.info(f"📍 {country} Supplier - Latin American emerging market")

        score = 62.0  # Moderate risk range (60-74)
        risk_factors.append("emerging_market_considerations")
        risk_factors.append("currency_exchange_factors")

        # Argentina has economic volatility
        if "argentina" in country_lower:
            score = 58.0  # Still moderate, not overly harsh
            risk_factors.append("economic_volatility_factors")

        # Chile and Peru have favorable trade agreements
        if any(c in country_lower for c in ["chile", "peru"]):
            score = 68.0  # Better moderate score
            risk_factors.append("bilateral_trade_agreement")

    else:
        # Other International Countries
        # Score: 60-70 (moderate risk per prompt)
        ctx.logger.info(f"📍 {country} Supplier - Standard international trade")

        score = 64.0  # Moderate risk range (60-74)
        risk_factors.append("standard_international_tariffs")
        risk_factors.append("import_duty_applicable")

        # Asia-Pacific developed economies (excluding China)
        if any(
            c in country_lower
            for c in ["japan", "south korea", "singapore", "australia"]
        ):
            score = 74.0  # Upper moderate range
            risk_factors.append("developed_economy")
            risk_factors.append("stable_trade_relations")

        # India - developing market
        if "india" in country_lower:
            score = 60.0  # Lower moderate range
            risk_factors.append("developing_market")
            risk_factors.append("tariff_considerations")

    # Ensure score is within bounds
    final_score = max(0.0, min(100.0, score))

    return {
        "financial_score": final_score,
        "risk_factors": risk_factors,
        "country_analyzed": country,
    }


# ============================================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================================


def build_financial_workflow(
    rag_system: FinancialRAGSystem, ctx: Context
) -> StateGraph:
    """
    Build the LangGraph financial workflow.

    Flow:
    START
        ↓
    financial_check_node (Extract country from Headquarters ONLY + Apply country-based analysis + Run RAG)
        ↓
    financial_router (Check score >= 60)
        ├─→ success_node (Score >= 60) → END
        └─→ error_node (Score < 60 or error) → END

    Returns:
        Compiled StateGraph workflow
    """
    # Create state graph
    workflow = StateGraph(SupplierWorkflowState)

    # Add nodes
    workflow.add_node(
        "financial_check", lambda state: financial_check_node(state, rag_system, ctx)
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

    # Compile workflow
    compiled_workflow = workflow.compile()
    print(f"Compiled workflow on line 987: {compiled_workflow}")

    return compiled_workflow
