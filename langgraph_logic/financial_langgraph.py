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
    """RAG system for financial risk analysis using B Corp page data and country-based tariff/inflation analysis"""

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

        print(f"Setting up Financial RAG System: {Settings.llm}")
        self.documents = []
        self.scraped_tariff_data = {}

    async def scrape_bcorp_financial_page(
        self, ctx: Context, supplier_name: str, b_corp_url: str, country: str
    ) -> Dict[str, str]:
        """
        Scrape B Corp page for financial-relevant data.

        Args:
            supplier_name: Name of supplier
            b_corp_url: B Corp profile URL
            country: Country where supplier operates

        Returns:
            Dictionary with scraped B Corp data relevant to financial analysis
        """
        driver = None
        try:
            ctx.logger.info(
                f"Scraping B Corp page for financial analysis: {supplier_name}"
            )

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
                "country": country,
                "url": b_corp_url,
                "headquarters": "",
                "size": "",
                "sector": "",
                "industry": "",
                "overall_score": "",
                "full_content": "",
            }

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Extract main content for analysis
            main_content = soup.get_text(separator=" ", strip=True)
            scraped_data["full_content"] = main_content[:5000]

            # Extract headquarters
            hq_patterns = [r"Headquarters[:\s]+([^|]+)", r"Operates In[:\s]+([^|]+)"]
            all_text = soup.get_text(separator="|", strip=True)
            for pattern in hq_patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match:
                    scraped_data["headquarters"] = match.group(1).strip()
                    break

            # Extract company size
            size_match = re.search(
                r"(\d+[-\s]+\d+)\s+employees", main_content, re.IGNORECASE
            )
            if size_match:
                scraped_data["size"] = size_match.group(1)

            # Extract sector and industry
            sector_match = re.search(r"Sector[:\s]+([^\n|]+)", all_text, re.IGNORECASE)
            if sector_match:
                scraped_data["sector"] = sector_match.group(1).strip()

            industry_match = re.search(
                r"Industry[:\s]+([^\n|]+)", all_text, re.IGNORECASE
            )
            if industry_match:
                scraped_data["industry"] = industry_match.group(1).strip()

            # Extract B Impact Score
            score_element = soup.find("div", class_=re.compile(".*score.*", re.I))
            if score_element:
                score_text = score_element.get_text(strip=True)
                scraped_data["overall_score"] = score_text

            ctx.logger.info(f"Successfully scraped B Corp data for {supplier_name}")
            return scraped_data

        except Exception as e:
            ctx.logger.error(f"Error scraping B Corp page: {e}")
            import traceback

            traceback.print_exc()

            return {
                "supplier_name": supplier_name,
                "country": country,
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
        country: str,
        industry: str,
        b_corp_url: str = "",
    ) -> Dict[str, Any]:
        """
        Query RAG system for financial risk analysis using B Corp data and country-based analysis.

        Process:
        1. Scrape B Corp page for supplier data
        2. Apply country-specific financial risk analysis based on US trade relationships
        3. Convert data to Document and create embeddings
        4. Store embeddings in Pinecone
        5. Query RAG system for financial risk analysis
        6. Return structured financial data

        Returns:
        {
            "financial_score": float (0-100, higher is better/lower risk),
            "financial_details": str,
            "risk_factors": List[str],
            "retrieved_context": str
        }
        """
        if not self.initialized:
            ctx.logger.error("RAG system not initialized")
            return self._generate_fallback_response(supplier_name, country)

        try:
            # Step 1: Validate inputs
            if not country:
                ctx.logger.error("No country provided for financial analysis")
                return self._generate_fallback_response(supplier_name, country)

            if not b_corp_url:
                ctx.logger.warning(
                    "No B Corp URL provided, using country-based analysis only"
                )
                scraped_data = {
                    "supplier_name": supplier_name,
                    "country": country,
                    "url": "",
                    "full_content": f"Supplier {supplier_name} operates in {country}. Industry: {industry}.",
                }
            else:
                # Scrape B Corp page for additional context
                scraped_data = await self.scrape_bcorp_financial_page(
                    ctx, supplier_name, b_corp_url, country
                )

                if "error" in scraped_data:
                    ctx.logger.warning(
                        "B Corp scraping failed, using country-based analysis only"
                    )
                    scraped_data["full_content"] = (
                        f"Supplier {supplier_name} operates in {country}. Industry: {industry}."
                    )

            # Step 2: Apply country-specific financial risk analysis
            country_analysis = analyze_country_financial_risk(
                scraped_data, country, ctx
            )

            # Step 3: Convert scraped data to LlamaIndex Document
            financial_text = f"""
            Supplier: {supplier_name}
            Operating Country: {country}
            Industry: {industry}
            
            B CORP INFORMATION:
            Headquarters: {scraped_data.get('headquarters', country)}
            Sector: {scraped_data.get('sector', 'Unknown')}
            Industry: {scraped_data.get('industry', industry)}
            Size: {scraped_data.get('size', 'Unknown')}
            B Impact Score: {scraped_data.get('overall_score', 'Unknown')}
            
            FINANCIAL RISK ANALYSIS:
            Based on US trade relationship with {country}:
            Financial Score: {country_analysis['financial_score']}/100
            Risk Factors: {', '.join(country_analysis['risk_factors'])}
            
            FULL CONTEXT:
            {scraped_data.get('full_content', 'No additional context')[:3000]}
            """

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
                    f"Analysis based on US economic conditions and B Corp certification standards."
                )
            else:
                result["financial_details"] = (
                    f"International supplier analysis for {supplier_name} from {country}. "
                    f"Financial risk evaluated based on US-{country} trade relationship and B Corp data. "
                    f"Score: {country_analysis['financial_score']}/100."
                )

            result["scraped_data"] = scraped_data

            return result

        except Exception as e:
            ctx.logger.error(f"Query failed: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name, country)

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
            return self._generate_fallback_response(supplier_name, country)

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
        country: str,
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
                                country=country,
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
                        country=country,
                        industry=industry,
                        b_corp_url=b_corp_url,
                    )
                )
                return result
        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name, country)


def financial_check_node(
    state: SupplierWorkflowState, rag_system: FinancialRAGSystem, ctx: Context
) -> SupplierWorkflowState:
    """
    Node 1: Run financial risk check using B Corp data and country-based trade analysis.

    Process:
    1. Get supplier country and B Corp URL from state
    2. Scrape B Corp page for supplier data
    3. Apply country-specific financial risk analysis based on US trade relationships
    4. Store in Pinecone and analyze with RAG
    5. Return financial score and details

    Input: supplier_name, supplier_country, industry, b_corp_profile_url
    Output: financial_score, financial_info, risk_factors
    """
    try:
        supplier_name = state.get("supplier_name", "Unknown")
        supplier_country = state.get("supplier_country", "")
        b_corp_url = state.get("b_corp_profile_url", "")

        if not supplier_country:
            ctx.logger.error("No supplier country provided for financial analysis")
            state["current_step"] = "financial_check_failed"
            state["error_message"] = "No country information available"
            state["financial_score"] = 0.0
            return state

        # Run RAG query for financial analysis (includes B Corp scraping and country analysis)
        rag_result = rag_system.query_financial_documents_sync(
            supplier_name=supplier_name,
            country=supplier_country,
            industry=state.get("industry", ""),
            b_corp_url=b_corp_url,
        )

        # Update state with financial results including risk_factors
        state["financial_score"] = rag_result.get("financial_score", 0.0)
        state["financial_info"] = rag_result.get("financial_details", "N/A")
        state["risk_factors"] = rag_result.get("risk_factors", [])
        state["current_step"] = "financial_check_complete"

        ctx.logger.info(f"Financial analysis complete for {supplier_country}")
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

    Analysis uses known trade facts and relationships between US and supplier countries:
    - US suppliers: No import tariffs, domestic economy focus
    - Countries with free trade agreements: Lower tariff risk
    - EU countries: Generally favorable trade, but potential VAT considerations
    - Latin American countries: Emerging market risks, currency volatility
    - China: High tariff risk due to ongoing trade tensions
    - Other international: Standard international trade analysis
    """
    country_lower = country.lower() if country else ""
    full_content = scraped_data.get("full_content", "").lower()

    # Base score
    score = 70.0
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
        ctx.logger.info(
            f"📍 US Supplier - No import tariffs, domestic economic analysis"
        )

        score += 20.0  # No import tariff risk - major advantage
        risk_factors.append("domestic_supplier_no_tariffs")

        # US suppliers only face domestic economic considerations
        # Note: Actual inflation data would require real-time economic APIs
        # For B Corp certified companies, assume stable operations

    elif is_high_tariff:
        # High Tariff Countries (China, Russia, etc.)
        ctx.logger.info(f"📍 {country} Supplier - High tariff environment")

        if "china" in country_lower:
            score -= 30.0  # Significant tariff exposure with China
            risk_factors.append("china_tariff_risk")
            risk_factors.append("trade_tension_impact")
        elif "russia" in country_lower:
            score -= 40.0  # Sanctions and trade restrictions
            risk_factors.append("sanctions_risk")
            risk_factors.append("supply_chain_instability")

    elif is_usmca:
        # USMCA Countries (Mexico, Canada) - Free Trade Agreement
        ctx.logger.info(f"📍 {country} Supplier - USMCA free trade partner")

        score += 15.0  # USMCA benefits
        risk_factors.append("usmca_free_trade")

        if "mexico" in country_lower:
            # Mexico: Strong trade partner, minimal tariffs
            risk_factors.append("favorable_trade_agreement")
        elif "canada" in country_lower:
            # Canada: Strong trade partner, stable economy
            risk_factors.append("stable_trade_partner")

    elif is_eu:
        # European Union Countries
        ctx.logger.info(f"📍 {country} (EU) Supplier - European trade analysis")

        score += 5.0  # Generally favorable trade with EU
        risk_factors.append("eu_trade_relationship")

        # Some EU products face lower tariffs
        # Spain, Germany, France are major trading partners
        if any(c in country_lower for c in ["spain", "germany", "france", "italy"]):
            score += 5.0
            risk_factors.append("major_eu_trading_partner")

    elif is_latin_american:
        # Latin American Countries (non-USMCA)
        ctx.logger.info(f"📍 {country} Supplier - Latin American emerging market")

        # Moderate tariff risk, currency volatility
        score -= 10.0
        risk_factors.append("emerging_market_risk")
        risk_factors.append("currency_volatility")

        # Argentina has specific economic challenges
        if "argentina" in country_lower:
            score -= 5.0
            risk_factors.append("argentina_economic_volatility")

        # Chile and Peru have trade agreements
        if any(c in country_lower for c in ["chile", "peru"]):
            score += 8.0
            risk_factors.append("trade_agreement_benefits")

    else:
        # Other International Countries
        ctx.logger.info(f"📍 {country} Supplier - Standard international trade")

        # Apply standard international tariff considerations
        score -= 5.0  # Base international tariff risk
        risk_factors.append("standard_international_tariffs")

        # Asia-Pacific (excluding China)
        if any(
            c in country_lower
            for c in ["japan", "south korea", "singapore", "australia"]
        ):
            score += 10.0
            risk_factors.append("developed_economy_partner")

        # India
        if "india" in country_lower:
            score -= 5.0
            risk_factors.append("developing_market_considerations")

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
    financial_check_node (Scrape B Corp + Apply country-based trade analysis + Run RAG)
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

    return compiled_workflow
