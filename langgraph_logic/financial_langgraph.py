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
TRADE_WAR_TRACKER_URL = "https://www.tradewartracker.com/"


class FinancialRAGSystem:
    """RAG system for financial risk analysis using Trade War Tracker tariff/inflation data"""

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

    async def scrape_trade_war_tracker(
        self, ctx: Context, country: str
    ) -> Dict[str, str]:
        """
        Scrape Trade War Tracker for tariff and inflation data for a specific country.
        
        Args:
            country: Country name to search for tariff/inflation data
            
        Returns:
            Dictionary with scraped tariff and inflation data
        """
        driver = None
        try:
            ctx.logger.info(f"Scraping Trade War Tracker for country: {country}")

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
            driver.get(TRADE_WAR_TRACKER_URL)
            time.sleep(5)

            scraped_data = {
                "country": country,
                "url": TRADE_WAR_TRACKER_URL,
                "tariff_info": "",
                "timeline_events": "",
                "trade_data": "",
                "full_content": "",
            }

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Extract main content
            main_content = soup.get_text(separator=" ", strip=True)
            scraped_data["full_content"] = main_content[:5000]  # Limit to 5000 chars

            # Extract timeline events (country-specific mentions)
            timeline_section = soup.find_all("li")
            country_mentions = []
            for item in timeline_section:
                text = item.get_text(strip=True)
                if country.lower() in text.lower():
                    country_mentions.append(text[:500])
            
            if country_mentions:
                scraped_data["timeline_events"] = " ".join(country_mentions[:5])

            # Extract tariff information from tables if available
            tables = soup.find_all("table")
            if tables:
                table_text = []
                for table in tables[:2]:  # Limit to first 2 tables
                    table_text.append(table.get_text(separator=" ", strip=True)[:1000])
                scraped_data["trade_data"] = " ".join(table_text)

            # Search for country-specific tariff mentions in the main content
            country_lower = country.lower()
            paragraphs = soup.find_all("p")
            tariff_paragraphs = []
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if country_lower in p_text.lower() and ("tariff" in p_text.lower() or "trade" in p_text.lower()):
                    tariff_paragraphs.append(p_text[:300])
            
            if tariff_paragraphs:
                scraped_data["tariff_info"] = " ".join(tariff_paragraphs[:3])

            if driver:
                driver.quit()

            ctx.logger.info(f"Successfully scraped tariff data for {country}")
            return scraped_data

        except Exception as e:
            ctx.logger.error(f"Error scraping Trade War Tracker: {e}")
            import traceback

            traceback.print_exc()

            if driver:
                try:
                    driver.quit()
                except:
                    pass

            return {
                "country": country,
                "url": TRADE_WAR_TRACKER_URL,
                "error": str(e),
            }

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
    ) -> Dict[str, Any]:
        """
        Query RAG system for country tariff/inflation information using Trade War Tracker.

        Process:
        1. Scrape Trade War Tracker for country tariff/inflation data
        2. Convert scraped data to Document and create embeddings
        3. Store embeddings in Pinecone
        4. Query RAG system for financial risk analysis
        5. Use LLM to analyze tariffs and inflation impact
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
            # Step 1: Scrape Trade War Tracker for country data
            if not country:
                ctx.logger.error("No country provided for tariff analysis")
                return self._generate_fallback_response(supplier_name, country)

            scraped_data = await self.scrape_trade_war_tracker(ctx, country)

            if "error" in scraped_data:
                ctx.logger.warning("Scraping failed, using fallback")
                return self._generate_fallback_response(supplier_name, country)

            # Step 2: Convert scraped data to LlamaIndex Document
            tariff_text = f"""
            Supplier: {supplier_name}
            Operating Country: {country}
            Industry: {industry}
            
            TARIFF INFORMATION:
            {scraped_data.get('tariff_info', 'No specific tariff data')}
            
            TIMELINE EVENTS:
            {scraped_data.get('timeline_events', 'No timeline events')}
            
            TRADE DATA:
            {scraped_data.get('trade_data', 'No trade data')}
            
            FULL CONTEXT:
            {scraped_data.get('full_content', 'No additional context')[:3000]}
            """

            # Create Document object
            doc = Document(
                text=tariff_text,
                metadata={
                    "supplier_name": supplier_name,
                    "country": country,
                    "source": "trade_war_tracker",
                    "url": TRADE_WAR_TRACKER_URL,
                },
            )

            # Step 3: Index the document in Pinecone
            await self._index_scraped_document(ctx, doc)

            # Step 4: Query RAG system
            query = finance_prompt(supplier_name, industry, country)

            # Query using the indexed data
            rag_response = self.query_engine.query(query)
            response_text = str(rag_response)

            # Step 5: Parse response and extract financial data
            result = await self._parse_rag_response(ctx, response_text, supplier_name, country)
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
                "financial_details": financial_details or "No financial information available",
                "risk_factors": risk_factors,
                "retrieved_context": response_text,
            }

        except Exception as e:
            ctx.logger.error(f"Error parsing RAG response: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name, country)

    def _generate_fallback_response(self, supplier_name: str, country: str = "Unknown") -> Dict[str, Any]:
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
    Node 1: Run financial risk check on country using Trade War Tracker.

    Process:
    1. Get supplier country from state
    2. Scrape Trade War Tracker for tariff/inflation data
    3. Store in Pinecone and analyze with RAG
    4. Return financial score and details

    Input: supplier_name, supplier_country, industry
    Output: financial_score, financial_info, risk_factors
    """
    try:
        supplier_name = state.get("supplier_name", "Unknown")
        supplier_country = state.get("supplier_country", "")
        
        if not supplier_country:
            ctx.logger.error("No supplier country provided for tariff analysis")
            state["current_step"] = "financial_check_failed"
            state["error_message"] = "No country information available"
            state["financial_score"] = 0.0
            return state

        # Run RAG query for financial analysis (includes scraping Trade War Tracker)
        rag_result = rag_system.query_financial_documents_sync(
            supplier_name=supplier_name,
            country=supplier_country,
            industry=state.get("industry", ""),
        )

        # Update state with financial results
        state["financial_score"] = rag_result.get("financial_score", 0.0)
        state["financial_info"] = rag_result.get("financial_details", "N/A")
        state["current_step"] = "financial_check_complete"

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
    financial_check_node (Scrape Trade War Tracker + Run RAG analysis)
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
