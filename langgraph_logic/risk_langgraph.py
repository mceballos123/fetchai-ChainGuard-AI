from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from models.risk import RiskRequest, RiskResponse
from langgraph_logic.state_schemas import SupplierWorkflowState
import os
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from dotenv import load_dotenv
from uagents import Context
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    Document,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from pinecone import Pinecone, ServerlessSpec
from ollama import Client

from langgraph.graph import StateGraph, START, END
from prompts.risk_prompt import risk_prompt
from llama_index.llms.ollama import Ollama

load_dotenv()

FILE_PATH = os.getenv("FILE_PATH_DOCUMENTS_RISK")
PINECONE_INDEX_NAME = os.getenv("PINECONE_RISK_MANGEMENT_INDEX_NAME")


def _resolve_risk_path() -> Path:
    """
    Resolve the risk management files path with fallback options.
    Ensures consistency across different environments and startups.

    Priority:
    1. Environment variable FILE_PATH_DOCUMENTS_RISK
    2. Relative path from backend directory
    3. Absolute path from current working directory
    """
    # Option 1: Use environment variable if set
    if FILE_PATH and FILE_PATH != "None":
        resolved = Path(FILE_PATH)
        if resolved.exists():
            return resolved

    # Option 2: Try relative path from root directory
    relative_path = Path(__file__).parent.parent / "risk_management_files"
    if relative_path.exists():
        return relative_path

    # Option 3: Try from current working directory
    cwd_path = Path.cwd() / "risk_management_files"
    if cwd_path.exists():
        return cwd_path

    # Fallback: Return the most likely path
    return relative_path


RISK_FILES_DIR = _resolve_risk_path()
EMBEDDING_DIMENSION = 768


class RiskRAGSystem:
    """RAG system for risk management using B Corp web scraping"""

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

        # Configure Settings for LlamaIndex (embeddings only)
        Settings.embed_model = self.embed_model
        Settings.llm = Ollama(model=self.llm_model, request_timeout=300)
        self.documents = []
        self.scraped_supplier_data = {}

    async def scrape_bcorp_supplier_page(
        self, ctx: Context, supplier_name: str, company_url: str
    ) -> Dict[str, str]:
        """
        Scrape B Corp supplier page for risk management information.
        Focuses on operational capacity, supply chain, and logistics data.
        """
        driver = None
        try:
            ctx.logger.info(f"Scraping B Corp page for risk analysis: {supplier_name}")

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
            driver.get(company_url)
            time.sleep(5)

            scraped_data = {
                "supplier_name": supplier_name,
                "url": company_url,
                "governance": "",
                "workers": "",
                "community": "",
                "environment": "",
                "customers": "",
                "overall_score": "",
            }

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Extract Overall B Impact Score
            score_element = soup.find("div", class_=re.compile(".*score.*", re.I))
            if score_element:
                score_text = score_element.get_text(strip=True)
                scraped_data["overall_score"] = score_text

            # Tab sections to scrape (same as compliance but for risk analysis)
            tabs = ["Governance", "Workers", "Community", "Environment", "Customers"]

            for tab in tabs:
                try:
                    tab_button = driver.find_element(
                        By.XPATH, f"//button[contains(text(), '{tab}')]"
                    )
                    driver.execute_script("arguments[0].click();", tab_button)
                    time.sleep(2)

                    tab_soup = BeautifulSoup(driver.page_source, "html.parser")
                    content_div = tab_soup.find("div", {"role": "tabpanel"})
                    if content_div:
                        tab_content = content_div.get_text(separator=" ", strip=True)
                        scraped_data[tab.lower()] = tab_content[:2000]

                except Exception as e:
                    continue

            if driver:
                driver.quit()

            ctx.logger.info(f"Successfully scraped risk data for {supplier_name}")
            return scraped_data

        except Exception as e:
            ctx.logger.error(f"Error scraping B Corp page: {e}")
            import traceback

            traceback.print_exc()

            if driver:
                try:
                    driver.quit()
                except:
                    pass

            return {
                "supplier_name": supplier_name,
                "url": company_url,
                "error": str(e),
            }

    async def initialize(self, ctx: Context) -> bool:
        try:
            ctx.logger.info("Initializing Risk Management RAG System...")

            if not await self._setup_pinecone(ctx):
                return False

            self.initialized = True
            ctx.logger.info("Risk Management RAG System initialized")
            return True

        except Exception as e:
            ctx.logger.error(f"Failed to initialize RAG system: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _load_risk_files(self, ctx: Context) -> bool:
        """Load only sunrise_sustainable risk management file"""
        try:
            # Load only the sunrise_sustainable file
            file_path = RISK_FILES_DIR / "sunrise_sustainable.txt"

            if not file_path.exists():
                ctx.logger.error(f"Risk management file not found: {file_path}")
                return False

            ctx.logger.info(f"Loading risk management file: {file_path.name}")

            # Use LlamaIndex to load the single document
            reader = SimpleDirectoryReader(str(RISK_FILES_DIR), required_exts=[".txt"])
            all_documents = reader.load_data()

            # Filter to only sunrise_sustainable
            self.documents = [
                doc
                for doc in all_documents
                if "sunrise_sustainable" in doc.metadata.get("file_name", "").lower()
            ]

            if not self.documents:
                ctx.logger.error(
                    "Could not load sunrise_sustainable risk management file"
                )
                return False

            ctx.logger.info(
                f"Successfully loaded risk management document for sunrise_sustainable"
            )
            return True

        except Exception as e:
            ctx.logger.error(f"Error loading risk management files: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def filter_to_supplier(self, ctx: Context, supplier_name: str) -> bool:
        """Filter documents to include only the selected supplier"""
        try:
            ctx.logger.info(f"Filtering documents to supplier: {supplier_name}")

            # Normalize supplier name for matching
            normalized_supplier = supplier_name.lower().replace("_", " ").strip()
            ctx.logger.info(f"Normalized search term: '{normalized_supplier}'")

            # Find matching document
            filtered_docs = []
            for doc in self.documents:
                doc_file_name = (
                    doc.metadata.get("file_name", "").lower().replace(".txt", "")
                )

                # Multiple matching strategies
                if (
                    normalized_supplier == doc_file_name.replace("_", " ")
                    or normalized_supplier in doc_file_name.replace("_", " ")
                    or doc_file_name.replace("_", " ") in normalized_supplier
                    or any(
                        word in doc_file_name for word in normalized_supplier.split()
                    )
                ):
                    filtered_docs.append(doc)

            if not filtered_docs:
                ctx.logger.error(
                    f"No risk documents found for supplier: '{supplier_name}'"
                )
                return False

            # Re-index with only the selected supplier
            self.documents = filtered_docs
            ctx.logger.info(
                f"Filtered to {len(filtered_docs)} document(s) for: {supplier_name}"
            )
            return True

        except Exception as e:
            ctx.logger.error(f"Error filtering documents: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _setup_pinecone(self, ctx: Context) -> bool:
        """Setup Pinecone vector store"""
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                ctx.logger.error("PINECONE_API_KEY not set")
                return False

            # Initialize Pinecone
            pc = Pinecone(api_key=pinecone_api_key)

            # Check if index exists
            existing_indexes = [idx.name for idx in pc.list_indexes()]

            if PINECONE_INDEX_NAME not in existing_indexes:
                ctx.logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                ctx.logger.info(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

            # Get Pinecone index
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)

            # Create LlamaIndex vector store
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

            # Create index from vector store
            self.index = VectorStoreIndex.from_vector_store(vector_store)

            # Create query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=1, response_mode="compact", verbose=True
            )

            ctx.logger.info("Pinecone vector store initialized")
            return True

        except Exception as e:
            ctx.logger.error(f"Error setting up Pinecone: {e}")
            return False

    async def _index_documents(self, ctx: Context) -> bool:
        """Index risk management documents in Pinecone (only if not already indexed)"""
        try:
            if not self.index:
                ctx.logger.warning("Index not initialized, skipping document indexing")
                return False

            # Check if documents are already indexed in Pinecone
            try:
                # Get the Pinecone index stats to check if it has vectors
                from pinecone import Pinecone

                pinecone_api_key = os.getenv("PINECONE_API_KEY")
                pc = Pinecone(api_key=pinecone_api_key)
                pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                stats = pinecone_index.describe_index_stats()

                total_vectors = stats.get("total_vector_count", 0)

                if total_vectors > 0:
                    ctx.logger.info(
                        f"Pinecone index already contains {total_vectors} vectors"
                    )
                    ctx.logger.info(
                        "Skipping document indexing (documents already in vector DB)"
                    )
                    return True
                else:
                    ctx.logger.info(
                        "Pinecone index is empty, proceeding with document indexing..."
                    )
            except Exception as e:
                ctx.logger.warning(
                    f"Could not check index stats: {e}, proceeding with indexing..."
                )

            # Parse documents into nodes (chunks)
            parser = SimpleNodeParser.from_defaults(
                chunk_size=512,
                chunk_overlap=20,
            )

            nodes = []
            for doc in self.documents:
                doc_nodes = parser.get_nodes_from_documents([doc])
                nodes.extend(doc_nodes)

            ctx.logger.info(f"Created {len(nodes)} document chunks")

            # Index nodes in Pinecone
            for node in nodes:
                try:
                    self.index.insert_nodes([node])
                except Exception as e:
                    ctx.logger.warning(f"Error indexing node: {e}")

            ctx.logger.info("Documents indexed in Pinecone")
            return True

        except Exception as e:
            ctx.logger.error(f"Error indexing documents: {e}")
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

            ctx.logger.info("Risk document indexed in Pinecone")
            return True

        except Exception as e:
            ctx.logger.error(f"Error indexing document: {e}")
            return False

    async def query_risk_documents(
        self,
        ctx: Context,
        supplier_name: str,
        industry: str,
        b_corp_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query RAG system for supplier risk information by scraping B Corp website.

        Process:
        1. Scrape B Corp page for supplier (all tabs for risk indicators)
        2. Convert scraped data to Document and create embeddings
        3. Store embeddings in Pinecone
        4. Query RAG system for risk analysis
        5. Use LLM to analyze risk factors and generate scores
        6. Return structured risk data

        Returns:
        {
            "risk_score": float (0-100),
            "risk_details": str,
            "risk_factors": List[str],
            "retrieved_context": str
        }
        """
        if not self.initialized:
            ctx.logger.error("RAG system not initialized")
            return self._generate_fallback_response(supplier_name)

        try:
            # Step 1: Scrape B Corp page for supplier
            if not b_corp_url:
                supplier_slug = (
                    supplier_name.lower().replace(" ", "-").replace("_", "-")
                )
                b_corp_url = f"https://www.bcorporation.net/en-us/find-a-b-corp/company/{supplier_slug}"

            ctx.logger.info(f"Scraping B Corp data for risk analysis: {supplier_name}")
            scraped_data = await self.scrape_bcorp_supplier_page(
                ctx, supplier_name, b_corp_url
            )

            if "error" in scraped_data:
                ctx.logger.warning("Scraping failed, using fallback")
                return self._generate_fallback_response(supplier_name)

            # Step 2: Convert scraped data to LlamaIndex Document
            supplier_text = f"""
            Supplier: {scraped_data['supplier_name']}
            Overall B Impact Score: {scraped_data.get('overall_score', 'N/A')}
            
            GOVERNANCE:
            {scraped_data.get('governance', 'No data')}
            
            WORKERS:
            {scraped_data.get('workers', 'No data')}
            
            COMMUNITY:
            {scraped_data.get('community', 'No data')}
            
            ENVIRONMENT:
            {scraped_data.get('environment', 'No data')}
            
            CUSTOMERS:
            {scraped_data.get('customers', 'No data')}
            """

            # Create Document object
            doc = Document(
                text=supplier_text,
                metadata={
                    "supplier_name": supplier_name,
                    "source": "bcorp_scrape",
                    "url": b_corp_url,
                },
            )

            # Step 3: Index the document in Pinecone
            await self._index_scraped_document(ctx, doc)

            # Step 4: Query RAG system
            query = risk_prompt(supplier_name, industry)

            # Query using the indexed data
            rag_response = self.query_engine.query(query)
            response_text = str(rag_response)

            # Step 5: Parse response and extract risk data
            result = await self._parse_rag_response(ctx, response_text, supplier_name)
            result["scraped_data"] = scraped_data

            return result

        except Exception as e:
            ctx.logger.error(f"Query failed: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)

    async def _parse_rag_response(
        self, ctx: Context, response_text: str, supplier_name: str
    ) -> Dict[str, Any]:
        """Parse LLM response and extract structured risk data"""
        try:

            # Initialize defaults (using 70 as neutral baseline instead of 50)
            risk_score = 50.0
            risk_details = "Insufficient risk data"
            risk_factors = []

            # Parse response
            lines = response_text.split("\n")

            for i, line in enumerate(lines):
                line_lower = line.lower()

                if "risk_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        risk_score = max(0, min(100, float(score_str)))
                    except (ValueError, IndexError):
                        pass

                elif "risk_details:" in line_lower:
                    # Extract content after "risk_details:"
                    content = line.split(":", 1)[-1].strip()

                    # If this line is minimal, look ahead for more content
                    if not content or len(content) < 30:
                        for next_line in lines[i + 1 :]:
                            next_line_lower = next_line.lower()
                            # Stop if we hit another field marker
                            if ":" in next_line and any(
                                field in next_line_lower
                                for field in ["risk_factors", "risk_score"]
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
                        risk_details = content

                elif "risk_factors:" in line_lower:
                    factors_str = line.split(":", 1)[-1].strip().lower()
                    if (
                        factors_str
                        and "minimal" not in factors_str
                        and "none" not in factors_str
                    ):
                        risk_factors = [f.strip() for f in factors_str.split(",")]

            return {
                "risk_score": float(risk_score),
                "risk_details": risk_details or "No risk information available",
                "risk_factors": risk_factors,
                "retrieved_context": response_text,
            }

        except Exception as e:
            ctx.logger.error(f"Error parsing RAG response: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)

    def _generate_fallback_response(self, supplier_name: str) -> Dict[str, Any]:
        """Generate fallback response when RAG is unavailable"""
        return {
            "risk_score": 70.0,  # Neutral baseline score
            "risk_details": f"Unable to retrieve detailed risk information for {supplier_name}",
            "risk_factors": ["Data retrieval unavailable"],
            "retrieved_context": "Fallback mode - RAG system unavailable",
        }

    def query_risk_documents_sync(
        self,
        supplier_name: str,
        industry: str,
        b_corp_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for query_risk_documents for use in LangGraph nodes"""
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
                            self.query_risk_documents(
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
                    self.query_risk_documents(
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
            return self._generate_fallback_response(supplier_name)


def risk_check_node(
    state: SupplierWorkflowState, rag_system: RiskRAGSystem, ctx: Context
) -> SupplierWorkflowState:
    """
    Node 1: Run risk management check on supplier using RAG system with web scraping.

    Process:
    1. Scrape B Corp webpage for supplier
    2. Convert to embeddings and store in Pinecone
    3. Query RAG system for risk analysis
    4. Return risk score and details

    Input: supplier_name, industry, b_corp_profile_url (optional)
    Output: risk_score, risk_details, risk_factors
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("[LangGraph Node: risk_check] SCRAPING & ANALYZING")
    ctx.logger.info("=" * 70)

    try:
        supplier_name = state.get("supplier_name", "Unknown")
        b_corp_url = state.get("b_corp_profile_url")

        ctx.logger.info(f"Analyzing risk for: {supplier_name}")

        # Run RAG query for risk management (includes scraping)
        rag_result = rag_system.query_risk_documents_sync(
            supplier_name=supplier_name,
            industry=state.get("industry", ""),
            b_corp_url=b_corp_url,
        )

        # Update state with risk results
        state["risk_score"] = rag_result.get("risk_score", 0.0)
        state["risk_details"] = rag_result.get("risk_details", "N/A")
        state["risk_factors"] = rag_result.get("risk_factors", [])
        state["current_step"] = "risk_check_complete"

        ctx.logger.info(f"Risk Score: {state['risk_score']}/100")

        return state

    except Exception as e:
        ctx.logger.error(f"Error in risk check: {e}")
        import traceback
        traceback.print_exc()

        state["current_step"] = "risk_check_failed"
        state["error_message"] = f"Risk check failed: {str(e)}"
        state["risk_score"] = 0.0
        return state


def risk_router(
    state: SupplierWorkflowState,
) -> Literal["success_node", "error_node"]:
    """
    Node 2: Conditional routing based on risk score.

    Routes to:
    - "success_node" if score >= 70 (APPROVED)
    - "error_node" if score < 70 OR error occurred (REJECTED)
    """
    risk_score = state.get("risk_score", 0.0)
    error_message = state.get("error_message")

    # Check for errors first
    if error_message:
        return "error_node"

    # Check risk threshold
    if risk_score >= 60:
        return "success_node"
    else:
        state["error_message"] = (
            f"Risk score {risk_score}/100 below threshold (60). Supplier rejected."
        )
        return "error_node"


def success_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3a: Success path - Supplier meets risk management requirements.
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("SUCCESS NODE - RISK MANAGEMENT APPROVED")
    ctx.logger.info("=" * 70)

    state["current_step"] = "risk_approved"
    state["should_continue"] = False

    ctx.logger.info(f"Supplier: {state.get('supplier_name', 'Unknown')}")
    ctx.logger.info(f"Score: {state['risk_score']}/100")
    ctx.logger.info(f"Status: APPROVED - Ready to send to orchestrator")
    ctx.logger.info("=" * 70)

    return state


def error_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3b: Error path - Supplier does not meet requirements.
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("ERROR NODE - RISK MANAGEMENT REJECTED")
    ctx.logger.info("=" * 70)

    risk_score = state.get("risk_score", 0.0)
    error_msg = state.get("error_message", "Unknown error")

    ctx.logger.info(f"Supplier: {state.get('supplier_name', 'Unknown')}")
    ctx.logger.info(f"Score: {risk_score}/100")
    ctx.logger.info(f"Error: {error_msg}")
    ctx.logger.info("=" * 70)

    state["current_step"] = "risk_rejected"
    state["should_continue"] = False

    return state


def build_risk_workflow(rag_system: RiskRAGSystem, ctx: Context) -> StateGraph:
    """
    Build the LangGraph risk management workflow.

    Flow:
    START
        ↓
    risk_check_node (Scrape B Corp + Run RAG analysis)
        ↓
    risk_router (Check score >= 60)
        ├─→ success_node (Score >= 60) → END
        └─→ error_node (Score < 60 or error) → END

    Returns:
        Compiled StateGraph workflow
    """
    ctx.logger.info("Building Risk Management LangGraph Workflow...")

    # Create state graph
    workflow = StateGraph(SupplierWorkflowState)

    # Add nodes
    workflow.add_node(
        "risk_check", lambda state: risk_check_node(state, rag_system, ctx)
    )
    workflow.add_node("success_node", lambda state: success_node(state, ctx))
    workflow.add_node("error_node", lambda state: error_node(state, ctx))

    # Add edges
    workflow.add_edge(START, "risk_check")
    workflow.add_conditional_edges(
        "risk_check",
        risk_router,
        {
            "success_node": "success_node",
            "error_node": "error_node",
        },
    )
    workflow.add_edge("success_node", END)
    workflow.add_edge("error_node", END)

    # Compile workflow
    compiled_workflow = workflow.compile()

    ctx.logger.info("Risk management workflow built successfully!")

    return compiled_workflow
