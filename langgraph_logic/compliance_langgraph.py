from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from models.compliance import ComplianceRequest, ComplianceResponse
from langgraph_logic.state_schemas import SupplierWorkflowState
import os
import json
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
from llama_index.llms.ollama import Ollama

from langgraph.graph import StateGraph, START, END
from prompts.compliance_prompt import compliance_prompt

load_dotenv()

FILE_PATH = os.getenv("FILE_PATH_DOCUMENTS_COMPLIANCE")


def _resolve_compliance_path() -> Path:
    """
    Resolve the compliance files path with fallback options.
    Ensures consistency across different environments and startups.

    Priority:
    1. Environment variable FILE_PATH_DOCUMENTS_COMPLIANCE
    2. Relative path from backend directory
    3. Absolute path from current working directory
    """
    # Option 1: Use environment variable if set
    if FILE_PATH and FILE_PATH != "None":
        resolved = Path(FILE_PATH)
        if resolved.exists():
            return resolved

    # Option 2: Try relative path from root directory
    relative_path = Path(__file__).parent.parent / "compliance_files"
    if relative_path.exists():
        return relative_path

    # Option 3: Try from current working directory
    cwd_path = Path.cwd() / "compliance_files"
    if cwd_path.exists():
        return cwd_path

    # Fallback: Return the most likely path (will error appropriately in _load_compliance_files)
    return relative_path


COMPLIANCE_FILES_DIR = _resolve_compliance_path()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # gets the pinecone index name
EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION")


class ComplianceRAGSystem:
    """RAG system for compliance document retrieval and analysis using B Corp web scraping"""

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

        self.documents = []
        self.scraped_supplier_data = {}

    async def scrape_bcorp_supplier_page(
        self, ctx: Context, supplier_name: str, company_url: str
    ) -> Dict[str, str]:
        """
        Scrape B Corp supplier page for ethics and sustainability information.
        Extracts data from multiple tabs: Governance, Workers, Community, Environment, Customers

        Args:
            supplier_name: Name of the supplier
            company_url: B Corp profile URL

        Returns:
            Dictionary with scraped data from all tabs
        """
        driver = None
        try:
            ctx.logger.info(f"Scraping B Corp page for: {supplier_name}")

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

            # Tab sections to scrape
            tabs = ["Governance", "Workers", "Community", "Environment", "Customers"]

            for tab in tabs:
                try:
                    # Try to find and click the tab button
                    tab_button = driver.find_element(
                        By.XPATH, f"//button[contains(text(), '{tab}')]"
                    )
                    driver.execute_script("arguments[0].click();", tab_button)
                    time.sleep(2)

                    # Get updated page content
                    tab_soup = BeautifulSoup(driver.page_source, "html.parser")

                    # Extract tab content
                    content_div = tab_soup.find("div", {"role": "tabpanel"})
                    if content_div:
                        tab_content = content_div.get_text(separator=" ", strip=True)
                        scraped_data[tab.lower()] = tab_content[:2000]

                except Exception as e:
                    continue

            if driver:
                driver.quit()

            ctx.logger.info(f"Successfully scraped data for {supplier_name}")
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
            ctx.logger.info("Initializing Compliance RAG System...")

            if not await self._setup_pinecone(ctx):
                return False

            self.initialized = True
            ctx.logger.info("Compliance RAG System initialized")
            return True

        except Exception as e:
            ctx.logger.error(f"Failed to initialize RAG system: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _load_compliance_files(self, ctx: Context) -> bool:
        """Load only sunrise_sustainable compliance file"""
        try:
            # Load only the sunrise_sustainable file
            file_path = COMPLIANCE_FILES_DIR / "sunrise_sustainable.txt"

            if not file_path.exists():
                ctx.logger.error(f"Compliance file not found: {file_path}")
                return False

            ctx.logger.info(f"Loading compliance file: {file_path.name}")

            # Use LlamaIndex to load the single document
            reader = SimpleDirectoryReader(
                str(COMPLIANCE_FILES_DIR), required_exts=[".txt"]
            )
            all_documents = reader.load_data()

            # Filter to only sunrise_sustainable
            self.documents = [
                doc
                for doc in all_documents
                if "sunrise_sustainable" in doc.metadata.get("file_name", "").lower()
            ]

            if not self.documents:
                ctx.logger.error("Could not load sunrise_sustainable compliance file")
                return False

            ctx.logger.info(
                f"Successfully loaded compliance document for sunrise_sustainable"
            )
            return True

        except Exception as e:
            ctx.logger.error(f"Error loading compliance files: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def select_best_supplier(
        self, ctx: Context, company_values: str, industry: str
    ) -> Optional[str]:
        """
        Select the BEST matching supplier from all loaded documents.

        Returns: Supplier name that best aligns with company values
        """
        if not self.documents:
            ctx.logger.error("No suppliers loaded")
            return None

        try:
            # Create a mapping of file names to help with matching
            supplier_files = {}
            for doc in self.documents:
                file_name = doc.metadata.get("file_name", "Unknown")
                normalized = file_name.lower().replace(".txt", "")
                supplier_files[normalized] = file_name

            # Build selection query for LLM
            supplier_list = "\n".join(
                [
                    f"- {doc.metadata.get('file_name', 'Unknown').replace('.txt', '')}"
                    for doc in self.documents
                ]
            )

            selection_query = f"""
            SUPPLIER SELECTION
            Company Values: {company_values}
            Industry: {industry}
            
            Available suppliers:
            {supplier_list}
            
            Select the ONE best supplier matching company values and industry.
            Respond with ONLY the exact supplier name from the list.
            """

            response = self.query_engine.query(selection_query)
            selected_supplier = str(response).strip()

            # Normalize the response for better matching
            normalized_response = selected_supplier.lower().replace("_", " ").strip()

            return selected_supplier

        except Exception as e:
            ctx.logger.error(f"Error selecting supplier: {e}")
            import traceback

            traceback.print_exc()

            # Fallback: return first supplier
            if self.documents:
                fallback = (
                    self.documents[0].metadata.get("file_name", "Unknown").lower()
                )
                ctx.logger.warning(f"✗ Using fallback supplier: {fallback}")
                return fallback
            return None

    async def filter_to_supplier(self, ctx: Context, supplier_name: str) -> bool:
        """
        Filter documents to include only the selected supplier.
        This ensures RAG only retrieves data about the chosen supplier.
        Handles name matching between LLM output and file names.
        """
        try:
            # Normalize supplier name for matching
            normalized_supplier = supplier_name.lower().replace("_", " ").strip()

            # Find matching document using multiple strategies
            filtered_docs = []
            for doc in self.documents:
                doc_file_name = (
                    doc.metadata.get("file_name", "").lower().replace(".txt", "")
                )

                # Strategy 1: Exact match on file name
                if normalized_supplier == doc_file_name.replace("_", " "):
                    filtered_docs.append(doc)

                # Strategy 2: Partial match (supplier name contains or is contained in file name)
                elif normalized_supplier in doc_file_name.replace("_", " "):
                    filtered_docs.append(doc)

                elif doc_file_name.replace("_", " ") in normalized_supplier:
                    filtered_docs.append(doc)

                # Strategy 3: Key word matching
                elif any(word in doc_file_name for word in normalized_supplier.split()):
                    filtered_docs.append(doc)

            if not filtered_docs:
                ctx.logger.error(f"No documents found for supplier: '{supplier_name}'")
                ctx.logger.error(f"Searched for: '{normalized_supplier}'")
                ctx.logger.error(f"  Available files:")
                for doc in self.documents:
                    ctx.logger.error(
                        f"    - {doc.metadata.get('file_name', 'Unknown')}"
                    )
                return False

            # Re-index with only the selected supplier
            self.documents = filtered_docs
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
            ctx.logger.info(f"existing_indexes: {existing_indexes}")
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

            # Create query engine with compact mode (faster than tree_summarize)
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=3, response_mode="compact", verbose=True
            )
            ctx.logger.info("Pinecone vector store initialized")
            return True

        except Exception as e:
            ctx.logger.error(f"Error setting up Pinecone: {e}")
            return False

    async def _index_documents(self, ctx: Context) -> bool:
        """Index compliance documents in Pinecone (only if not already indexed)"""
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
                        f"Pinecone compliance index already contains {total_vectors} vectors"
                    )
                    ctx.logger.info(
                        "Skipping document indexing (compliance documents already in vector DB)"
                    )
                    return True
                else:
                    ctx.logger.info(
                        "Pinecone compliance index is empty, proceeding with document indexing..."
                    )
            except Exception as e:
                ctx.logger.warning(
                    f"Could not check compliance index stats: {e}, proceeding with indexing..."
                )

            # Parse documents into nodes (chunks)
            parser = SimpleNodeParser.from_defaults(
                chunk_size=512,
                chunk_overlap=20,  # Chunks of 512 tokens with 20 token overlap
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

            ctx.logger.info("Compliance documents indexed in Pinecone")
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

            ctx.logger.info("Document indexed in Pinecone")
            return True

        except Exception as e:
            ctx.logger.error(f"Error indexing document: {e}")
            return False

    async def query_compliance_documents(
        self,
        ctx: Context,
        supplier_name: str,
        company_values: str,
        industry: str,
        b_corp_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query RAG system for supplier compliance information by scraping B Corp website.

        Process:
        1. Scrape B Corp page for supplier (Governance, Workers, Community, Environment, Customers)
        2. Convert scraped data to Document and create embeddings
        3. Store embeddings in Pinecone
        4. Query RAG system based on company values
        5. Use LLM to analyze compliance and generate scores
        6. Return structured compliance data

        Returns:
        {
            "compliance_score": float (0-100),
            "ethics_info": str,
            "sustainability_info": str,
            "violations": List[str],
            "selected_supplier": str,
            "retrieved_context": str
        }
        """
        if not self.initialized:
            ctx.logger.error("RAG system not initialized")
            return self._generate_fallback_response(supplier_name)

        try:
            # Step 1: Scrape B Corp page for supplier
            if not b_corp_url:
                # Construct B Corp URL from supplier name
                supplier_slug = (
                    supplier_name.lower().replace(" ", "-").replace("_", "-")
                )
                b_corp_url = f"https://www.bcorporation.net/en-us/find-a-b-corp/company/{supplier_slug}"

            ctx.logger.info(f"Step 1: Scraping B Corp data for {supplier_name}")
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
            query = compliance_prompt(company_values, industry, supplier_name)

            # Query using the indexed data
            rag_response = self.query_engine.query(query)
            response_text = str(rag_response)

            # Step 5: Parse response and extract compliance data
            result = await self._parse_rag_response(ctx, response_text, supplier_name)
            result["selected_supplier"] = supplier_name
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
        """Parse LLM response and extract structured compliance data"""
        try:

            # Initialize defaults
            ethics_score = 50.0
            sustainability_score = 50.0
            combined_score = 50.0
            ethics_info = "Insufficient data"
            sustainability_info = "Insufficient data"
            violations = []

            # Parse response (handle various formats)
            lines = response_text.split("\n")

            for i, line in enumerate(lines):
                line_lower = line.lower()

                if "ethics_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        ethics_score = max(0, min(100, float(score_str)))
                    except (ValueError, IndexError):
                        pass

                elif "sustainability_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        sustainability_score = max(0, min(100, float(score_str)))
                    except (ValueError, IndexError):
                        pass

                elif "combined_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        combined_score = max(0, min(100, float(score_str)))
                    except (ValueError, IndexError):
                        pass

                elif "ethics_info:" in line_lower:
                    # Extract content after "ethics_info:"
                    content = line.split(":", 1)[-1].strip()

                    # If this line is empty or minimal, try to get content from next lines
                    if not content or len(content) < 20:
                        # Look ahead for more content
                        for next_line in lines[i + 1 :]:
                            next_line_lower = next_line.lower()
                            # Stop if we hit another field marker
                            if ":" in next_line and any(
                                field in next_line_lower
                                for field in [
                                    "sustainability",
                                    "violations",
                                    "combined",
                                    "score",
                                ]
                            ):
                                break
                            if next_line.strip():
                                content += " " + next_line.strip()
                            else:
                                break

                    if content:
                        ethics_info = content

                elif "sustainability_info:" in line_lower:
                    # Extract content after "sustainability_info:"
                    content = line.split(":", 1)[-1].strip()

                    # If this line is empty or minimal, try to get content from next lines
                    if not content or len(content) < 20:
                        # Look ahead for more content
                        for next_line in lines[i + 1 :]:
                            next_line_lower = next_line.lower()
                            # Stop if we hit another field marker
                            if ":" in next_line and any(
                                field in next_line_lower
                                for field in [
                                    "violations",
                                    "combined",
                                    "score",
                                    "ethics",
                                ]
                            ):
                                break
                            if next_line.strip():
                                content += " " + next_line.strip()
                            else:
                                break

                    if content:
                        sustainability_info = content

                elif "violations:" in line_lower:
                    violations_str = line.split(":", 1)[-1].strip().lower()
                    if violations_str != "none" and violations_str:
                        violations = [v.strip() for v in violations_str.split(",")]

            # Fallback: If we still have "Insufficient data", try to extract from raw text
            if ethics_info == "Insufficient data":
                # Try to find ethics-related content in the response
                if "ethics" in response_text.lower():
                    ethics_section = self._extract_section_content(
                        response_text, "ethics", "sustainability"
                    )
                    if ethics_section:
                        ethics_info = ethics_section

            if sustainability_info == "Insufficient data":
                # Try to find sustainability-related content in the response
                if "sustainability" in response_text.lower():
                    sustainability_section = self._extract_section_content(
                        response_text, "sustainability", "violations"
                    )
                    if sustainability_section:
                        sustainability_info = sustainability_section

            # If combined_score wasn't provided, calculate it
            if combined_score == 50.0:
                combined_score = (ethics_score + sustainability_score) / 2

            return {
                "compliance_score": float(combined_score),
                "ethics_info": ethics_info or "No ethics information available",
                "sustainability_info": sustainability_info
                or "No sustainability information available",
                "violations": violations,
                "retrieved_context": response_text,
            }

        except Exception as e:
            ctx.logger.error(f"Error parsing RAG response: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)

    def _extract_section_content(
        self, text: str, start_marker: str, end_marker: str, max_length: int = 500
    ) -> Optional[str]:
        """
        Extract content from text between two markers.
        Used as fallback when standard parsing fails.
        """
        try:
            text_lower = text.lower()
            start_idx = text_lower.find(start_marker.lower())
            if start_idx == -1:
                return None

            # Find the end marker
            end_idx = text_lower.find(end_marker.lower(), start_idx + len(start_marker))
            if end_idx == -1:
                end_idx = len(text)

            content = text[start_idx:end_idx].strip()
            # Remove the start marker itself
            if content.lower().startswith(start_marker.lower()):
                content = content[len(start_marker) :].strip()

            # Clean up the content
            content = content.replace("\n", " ").strip()
            # Remove trailing colons if any
            content = content.lstrip(":").strip()

            if content and len(content) > 10:
                return content[:max_length]
            return None
        except Exception:
            return None

    def _generate_fallback_response(self, supplier_name: str) -> Dict[str, Any]:
        """Generate fallback response when RAG is unavailable"""
        return {
            "compliance_score": 50.0,  # Neutral score
            "ethics_info": f"Unable to retrieve detailed ethics information for {supplier_name}",
            "sustainability_info": f"Unable to retrieve detailed sustainability information for {supplier_name}",
            "violations": ["Data retrieval unavailable"],
            "retrieved_context": "Fallback mode - RAG system unavailable",
        }

    def query_compliance_documents_sync(
        self,
        supplier_name: str,
        company_values: str,
        industry: str,
        b_corp_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for query_compliance_documents for use in LangGraph nodes.

        This method should only be called from synchronous contexts.
        For async contexts, use query_compliance_documents() instead.
        """
        import asyncio

        # Create a mock context for sync execution
        class MockContext:
            def __init__(self):
                self.logs = []

            def logger_info(self, msg):
                self.logs.append(msg)

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
            # Use asyncio.run() to create a new event loop properly
            # This handles the case where an event loop is already running
            try:
                # Try to get current running loop (will raise if no loop)
                loop = asyncio.get_running_loop()
                # If we're here, a loop is running, we need to use a different approach
                # Create a new thread with its own event loop
                import concurrent.futures
                import threading

                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.query_compliance_documents(
                                ctx=mock_ctx,
                                supplier_name=supplier_name,
                                company_values=company_values,
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
                    self.query_compliance_documents(
                        ctx=mock_ctx,
                        supplier_name=supplier_name,
                        company_values=company_values,
                        industry=industry,
                        b_corp_url=b_corp_url,
                    )
                )
                return result
        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)


def compliance_check_node(
    state: SupplierWorkflowState, rag_system: ComplianceRAGSystem, ctx: Context
) -> SupplierWorkflowState:
    """
    Node 1: Run compliance check on supplier using RAG system with web scraping.

    Process:
    1. Scrape B Corp webpage for supplier
    2. Convert to embeddings and store in Pinecone
    3. Query RAG system for compliance analysis
    4. Return compliance score and details

    Input: supplier_name, company_values, industry, b_corp_profile_url (optional)
    Output: compliance_score, ethics_info, sustainability_info, violations
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("[LangGraph Node: compliance_check] SCRAPING & ANALYZING")
    ctx.logger.info("=" * 70)

    try:
        supplier_name = state.get("supplier_name", "Unknown")
        b_corp_url = state.get("b_corp_profile_url")

        ctx.logger.info(f"Analyzing compliance for: {supplier_name}")

        # Run RAG query for compliance (includes scraping)
        rag_result = rag_system.query_compliance_documents_sync(
            supplier_name=supplier_name,
            company_values=state.get("company_values", ""),
            industry=state.get("industry", ""),
            b_corp_url=b_corp_url,
        )

        # Update state with compliance results
        state["compliance_score"] = rag_result.get("compliance_score", 0.0)
        state["ethics_info"] = rag_result.get("ethics_info", "N/A")
        state["sustainability_info"] = rag_result.get("sustainability_info", "N/A")
        state["violations"] = rag_result.get("violations", [])
        state["rag_context"] = rag_result.get("retrieved_context", "")
        state["current_step"] = "compliance_check_complete"

        ctx.logger.info(f"Compliance Score: {state['compliance_score']}/100")

        return state

    except Exception as e:
        ctx.logger.error(f"Error in compliance check: {e}")
        import traceback

        traceback.print_exc()

        state["current_step"] = "compliance_check_failed"
        state["error_message"] = f"Compliance check failed: {str(e)}"
        state["compliance_score"] = 0.0
        return state


def compliance_router(
    state: SupplierWorkflowState,
) -> Literal["success_node", "error_node"]:
    """
    Node 2: Conditional routing based on compliance score.

    Routes to:
    - "success_node" if score >= 75 (APPROVED)
    - "error_node" if score < 75 OR error occurred (REJECTED)
    """
    compliance_score = state.get("compliance_score", 0.0)
    error_message = state.get("error_message")

    # Check for errors first
    if error_message:
        return "error_node"

    # Check compliance threshold (60+ means acceptable compliance)
    if compliance_score >= 60:
        return "success_node"
    else:
        state["error_message"] = (
            f"Compliance score {compliance_score}/100 below threshold (60). Supplier rejected."
        )
        return "error_node"


def success_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3a: Success path - Supplier meets compliance requirements.

    Prepares data to send to orchestrator agent.
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("SUCCESS NODE - COMPLIANCE APPROVED")
    ctx.logger.info("=" * 70)

    state["current_step"] = "compliance_approved"
    state["should_continue"] = False

    ctx.logger.info(f"Supplier: {state.get('supplier_name', 'Unknown')}")
    ctx.logger.info(f"Score: {state['compliance_score']}/100")
    ctx.logger.info(f"Status: APPROVED - Ready to send to orchestrator")
    ctx.logger.info("=" * 70)

    return state


def error_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3b: Error path - Supplier does not meet requirements.

    Prepares error response.
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("ERROR NODE - COMPLIANCE REJECTED")
    ctx.logger.info("=" * 70)

    compliance_score = state.get("compliance_score", 0.0)
    error_msg = state.get("error_message", "Unknown error")

    ctx.logger.info(f"Supplier: {state.get('supplier_name', 'Unknown')}")
    ctx.logger.info(f"Score: {compliance_score}/100")
    ctx.logger.info(f"Error: {error_msg}")
    ctx.logger.info("=" * 70)

    state["current_step"] = "compliance_rejected"
    state["should_continue"] = False

    return state


# ============================================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================================


def build_compliance_workflow(
    rag_system: ComplianceRAGSystem, ctx: Context
) -> StateGraph:
    """
    Build the LangGraph compliance workflow.

    Flow:
    START
        ↓
    compliance_check_node (Run RAG analysis)
        ↓
    compliance_router (Check score >= 75)
        ├─→ success_node (Score >= 75) → END
        └─→ error_node (Score < 75 or error) → END

    Returns:
        Compiled StateGraph workflow
    """
    ctx.logger.info("Building Compliance LangGraph Workflow...")

    # Create state graph
    workflow = StateGraph(SupplierWorkflowState)

    # Add nodes
    workflow.add_node(
        "compliance_check", lambda state: compliance_check_node(state, rag_system, ctx)
    )
    workflow.add_node("success_node", lambda state: success_node(state, ctx))
    workflow.add_node("error_node", lambda state: error_node(state, ctx))

    # Add edges
    workflow.add_edge(START, "compliance_check")
    workflow.add_conditional_edges(
        "compliance_check",
        compliance_router,
        {
            "success_node": "success_node",
            "error_node": "error_node",
        },
    )
    workflow.add_edge("success_node", END)
    workflow.add_edge("error_node", END)

    # Compile workflow
    compiled_workflow = workflow.compile()

    ctx.logger.info("Compliance workflow built successfully!")

    return compiled_workflow
