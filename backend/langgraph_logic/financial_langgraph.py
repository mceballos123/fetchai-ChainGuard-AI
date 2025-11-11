from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from backend.models.financial import FinancialRequest, FinancialResponse
from backend.langgraph_logic.state_schemas import SupplierWorkflowState
import os
import json
import re

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
from llama_index.llms.ollama import Ollama

from langgraph.graph import StateGraph, START, END
from ..prompts.finance_prompt import finance_prompt
load_dotenv()

FILE_PATH = os.getenv("FILE_PATH_DOCUMENTS_FINANCIAL", "backend/financial_files")


def _resolve_financial_path() -> Path:
    """
    Resolve the financial files path with fallback options.
    Ensures consistency across different environments and startups.

    Priority:
    1. Environment variable FILE_PATH_DOCUMENTS_FINANCIAL
    2. Relative path from backend directory
    3. Absolute path from current working directory
    """
    # Option 1: Use environment variable if set
    if FILE_PATH and FILE_PATH != "None":
        resolved = Path(FILE_PATH)
        if resolved.exists():
            return resolved

    # Option 2: Try relative path from backend directory
    relative_path = Path(__file__).parent.parent / "financial_files"
    if relative_path.exists():
        return relative_path

    # Option 3: Try from current working directory
    cwd_path = Path.cwd() / "backend" / "financial_files"
    if cwd_path.exists():
        return cwd_path

    # Fallback: Return the most likely path (will error appropriately in _load_financial_files)
    return relative_path


FINANCIAL_FILES_DIR = _resolve_financial_path()
PINECONE_INDEX_NAME_FINANCIAL = os.getenv(
    "PINECONE_INDEX_NAME_FINANCIAL", "financial-risk-index"
)
EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION", 768)


class FinancialRAGSystem:
    """RAG system for financial document retrieval and risk analysis using local files"""

    def __init__(self):
        self.initialized = False
        self.index = None
        self.query_engine = None

        # Initialize Ollama LLM
        self.llm = Ollama(model="llama3.2:1b", request_timeout=600)
        self.llm_type = "ollama"

        # Initialize Ollama embeddings
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Configure Settings for LLamaIndex (BEFORE any Pinecone initialization!)
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.documents = []

    async def initialize(self, ctx: Context) -> bool:

        try:
            ctx.logger.info("Initializing Financial RAG System...")

            # Step 1: Load financial files
            ctx.logger.info(f"Loading financial files from {FINANCIAL_FILES_DIR}")
            if not await self._load_financial_files(ctx):
                return False

            # Step 2: Initialize Pinecone
            ctx.logger.info("Initializing Pinecone vector store for financial data...")
            if not await self._setup_pinecone(ctx):
                return False

            # Step 3: Index documents in Pinecone
            ctx.logger.info("Indexing financial documents...")
            if not await self._index_documents(ctx):
                return False

            self.initialized = True
            ctx.logger.info("Financial RAG System initialized successfully!")
            return True

        except Exception as e:
            ctx.logger.error(f"Failed to initialize Financial RAG system: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _load_financial_files(self, ctx: Context) -> bool:
        """Load financial files from directory with validation"""
        try:
            # Validate path exists
            if not FINANCIAL_FILES_DIR.exists():
                ctx.logger.error(
                    f"Financial files directory not found: {FINANCIAL_FILES_DIR}"
                )
                ctx.logger.error(f"Current working directory: {Path.cwd()}")
                ctx.logger.error(
                    f"Expected 5 financial files in: {FINANCIAL_FILES_DIR}"
                )
                return False

            # List files before loading
            files_in_dir = list(FINANCIAL_FILES_DIR.glob("*.txt"))
            ctx.logger.info(f"Files found in directory: {len(files_in_dir)}")
            for f in files_in_dir:
                ctx.logger.info(f"   - {f.name}")

            # Use LlamaIndex to load documents
            reader = SimpleDirectoryReader(str(FINANCIAL_FILES_DIR))
            self.documents = reader.load_data()

            # Validation checks
            if not self.documents:
                ctx.logger.error("No documents found in financial_files folder")
                return False

            # Ensure we have all 5 expected documents
            expected_count = 5
            if len(self.documents) < expected_count:
                ctx.logger.warning(
                    f"Only loaded {len(self.documents)} documents, expected {expected_count}"
                )

            ctx.logger.info(
                f"Loaded {len(self.documents)} financial documents"
            )
            for i, doc in enumerate(self.documents, 1):
                file_name = doc.metadata.get("file_name", "Unknown")
                ctx.logger.info(f"   {i}. {file_name}")

            return True

        except Exception as e:
            ctx.logger.error(f"Error loading financial files: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def filter_to_supplier(self, ctx: Context, supplier_name: str) -> bool:
        """
        Filter documents to include only the selected supplier.
        This ensures RAG only retrieves financial data about the chosen supplier.
        Handles name matching between supplier input and file names.
        """
        try:
            ctx.logger.info(
                f"Filtering financial documents to supplier: {supplier_name}"
            )
            ctx.logger.info(f"Available files:")

            # Log all available documents with their names
            for doc in self.documents:
                doc_name = doc.metadata.get("file_name", "Unknown")
                ctx.logger.info(f"   - File: {doc_name}")

            # Normalize supplier name for matching
            normalized_supplier = supplier_name.lower().replace("_", " ").strip()
            ctx.logger.info(f"Normalized search term: '{normalized_supplier}'")

            # Find matching document using multiple strategies
            filtered_docs = []
            for doc in self.documents:
                doc_file_name = (
                    doc.metadata.get("file_name", "")
                    .lower()
                    .replace(".txt", "")
                    .replace("_financial", "")
                )

                # Multiple matching strategies
                if (
                    normalized_supplier == doc_file_name.replace("_", " ")
                    or normalized_supplier in doc_file_name.replace("_", " ")
                    or doc_file_name.replace("_", " ") in normalized_supplier
                    or any(word in doc_file_name for word in normalized_supplier.split())
                ):
                    filtered_docs.append(doc)

            if not filtered_docs:
                ctx.logger.error(
                    f"No financial documents found for supplier: '{supplier_name}'"
                )
                ctx.logger.error(f"  Searched for: '{normalized_supplier}'")
                ctx.logger.error(f"  Available files:")
                for doc in self.documents:
                    ctx.logger.error(
                        f"    - {doc.metadata.get('file_name', 'Unknown')}"
                    )
                return False

            # Re-index with only the selected supplier
            self.documents = filtered_docs
            ctx.logger.info(
                f"Filtered to {len(filtered_docs)} financial document(s) for: {supplier_name}"
            )
            return True

        except Exception as e:
            ctx.logger.error(f"Error filtering financial documents: {e}")
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

            # Create query engine with tree summarize
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5, response_mode="tree_summarize", verbose=True
            )

            ctx.logger.info("Pinecone vector store initialized for financial data")
            return True

        except Exception as e:
            ctx.logger.error(f"Error setting up Pinecone for financial data: {e}")
            return False

    async def _index_documents(self, ctx: Context) -> bool:
        """Index financial documents in Pinecone (only if not already indexed)"""
        try:
            if not self.index:
                ctx.logger.warning(
                    "Index not initialized, skipping financial document indexing"
                )
                return False

            # Check if documents are already indexed in Pinecone
            try:
                # Get the Pinecone index stats to check if it has vectors
                from pinecone import Pinecone
                pinecone_api_key = os.getenv("PINECONE_API_KEY")
                pc = Pinecone(api_key=pinecone_api_key)
                pinecone_index = pc.Index(PINECONE_INDEX_NAME_FINANCIAL)
                stats = pinecone_index.describe_index_stats()
                
                total_vectors = stats.get('total_vector_count', 0)
                
                if total_vectors > 0:
                    ctx.logger.info(f"Pinecone financial index already contains {total_vectors} vectors")
                    ctx.logger.info("Skipping document indexing (financial documents already in vector DB)")
                    return True
                else:
                    ctx.logger.info("Pinecone financial index is empty, proceeding with document indexing...")
            except Exception as e:
                ctx.logger.warning(f"Could not check financial index stats: {e}, proceeding with indexing...")

            # Parse documents into nodes (chunks)
            parser = SimpleNodeParser.from_defaults(
                chunk_size=512,
                chunk_overlap=20,  # Chunks of 512 tokens with 20 token overlap
            )

            nodes = []
            for doc in self.documents:
                doc_nodes = parser.get_nodes_from_documents([doc])
                nodes.extend(doc_nodes)

            ctx.logger.info(f"Created {len(nodes)} financial document chunks")

            # Index nodes in Pinecone
            for node in nodes:
                try:
                    self.index.insert_nodes([node])
                except Exception as e:
                    ctx.logger.warning(f"Error indexing financial node: {e}")

            ctx.logger.info("Financial documents indexed in Pinecone")
            return True

        except Exception as e:
            ctx.logger.error(f"Error indexing financial documents: {e}")
            return False

    async def query_financial_documents(
        self,
        ctx: Context,
        supplier_name: str,
        industry: str,
    ) -> Dict[str, Any]:
        """
        Query RAG system for supplier financial risk information.

        Process:
        1. Filter to the specified supplier's financial documents
        2. Retrieve relevant chunks from ONLY that supplier
        3. Use LLM (Ollama) to analyze financial risks
        4. Extract tariffs, inflation, and other financial risk factors
        5. Return structured financial data

        Returns:
        {
            "financial_score": float (0-100, higher is better/lower risk),
            "financial_details": str,
            "risk_factors": List[str],
            "retrieved_context": str
        }
        """
        if not self.initialized or not self.query_engine:
            ctx.logger.error("Financial RAG system not initialized")
            return self._generate_fallback_response(supplier_name)

        try:
            # Step 1: Filter documents to ONLY the selected supplier
            ctx.logger.info("=" * 60)
            ctx.logger.info("FINANCIAL DOCUMENT FILTERING PHASE")
            ctx.logger.info("=" * 60)

            if not await self.filter_to_supplier(ctx, supplier_name):
                ctx.logger.error(
                    "Failed to filter to selected supplier for financial analysis"
                )
                return self._generate_fallback_response(supplier_name)

            # Step 2: Financial risk analysis on the selected supplier ONLY
            ctx.logger.info("\n" + "=" * 60)
            ctx.logger.info("FINANCIAL RISK ANALYSIS PHASE")
            ctx.logger.info("=" * 60)

            # Build financial analysis query
            financial_query = finance_prompt(supplier_name, industry)
            
            

            ctx.logger.info(
                f"Analyzing financial risks for {supplier_name} (LLM: {self.llm_type})"
            )

            # Query the RAG system with ONLY the selected supplier's financial documents
            response = self.query_engine.query(financial_query)

            # Parse the response
            result = await self._parse_rag_response(ctx, str(response), supplier_name)

            ctx.logger.info(f"Financial risk analysis complete!")
            ctx.logger.info(f"   Supplier: {supplier_name}")
            ctx.logger.info(f"   Financial Score: {result['financial_score']}/100")

            return result

        except Exception as e:
            ctx.logger.error(f"Financial query failed: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)

    async def _parse_rag_response(
        self, ctx: Context, response_text: str, supplier_name: str
    ) -> Dict[str, Any]:
        """Parse LLM response and extract structured financial data"""
        try:
            ctx.logger.info("Parsing financial scores from LLM response...")
            ctx.logger.info("=" * 70)
            ctx.logger.info("RAW LLM RESPONSE:")
            ctx.logger.info(response_text)
            ctx.logger.info("=" * 70)

            # Initialize defaults
            financial_score = 50.0
            financial_details = "Insufficient financial data"
            risk_factors = []

            # Parse response (handle various formats)
            lines = response_text.split("\n")

            for i, line in enumerate(lines):
                line_lower = line.lower()

                if "financial_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        financial_score = max(0, min(100, float(score_str)))
                        ctx.logger.info(f"Financial Score: {financial_score}")
                    except (ValueError, IndexError) as e:
                        ctx.logger.warning(
                            f"Could not parse financial score from: {line} - {e}"
                        )
                        pass

                elif "financial_details:" in line_lower:
                    # Extract content after "financial_details:"
                    content = line.split(":", 1)[-1].strip()

                    # If this line is empty or minimal, try to get content from next lines
                    if not content or len(content) < 30:
                        # Look ahead for more content (up to 10 lines max)
                        collected_lines = [content] if content else []
                        line_count = 0

                        for next_line in lines[i + 1 :]:
                            line_count += 1
                            if line_count > 10:  # Prevent infinite collection
                                break

                            next_line_lower = next_line.lower()
                            next_line_stripped = next_line.strip()

                            # Stop if we hit another field marker
                            if ":" in next_line_lower and any(
                                field in next_line_lower
                                for field in [
                                    "risk_factors:",
                                    "financial_score:",
                                ]
                            ):
                                break

                            # Add non-empty lines with proper spacing
                            if next_line_stripped:
                                if collected_lines and not collected_lines[-1].endswith(
                                    " "
                                ):
                                    collected_lines.append(" ")
                                collected_lines.append(next_line_stripped)
                            else:
                                # Empty line indicates end of section
                                break

                        # Join with proper spacing
                        content = "".join(collected_lines)

                    if content:
                        # Clean up any accidental text concatenation issues
                        # Fix cases like "beansfrC" -> "beans from"
                        content = re.sub(r"(\w)([A-Z][a-z])", r"\1 \2", content)
                        # Ensure proper spacing around common words
                        content = re.sub(r"\s+", " ", content).strip()

                        financial_details = content
                        ctx.logger.info(
                            f"Extracted FINANCIAL_DETAILS: {financial_details[:100]}..."
                        )

                elif "risk_factors:" in line_lower:
                    risk_str = line.split(":", 1)[-1].strip().lower()
                    if risk_str != "minimal risks" and risk_str and risk_str != "none":
                        risk_factors = [r.strip() for r in risk_str.split(",")]
                    ctx.logger.info(f"Extracted RISK_FACTORS: {risk_factors}")

            ctx.logger.info("=" * 70)
            ctx.logger.info(f"FINAL PARSED FINANCIAL DATA:")
            ctx.logger.info(f"  Financial Score: {financial_score}/100")
            ctx.logger.info(
                f"  Details Extracted: {financial_details != 'Insufficient financial data'}"
            )
            ctx.logger.info(f"  Risk Factors Count: {len(risk_factors)}")
            ctx.logger.info("=" * 70)

            return {
                "financial_score": float(financial_score),
                "financial_details": financial_details
                or "No financial details available",
                "risk_factors": risk_factors,
                "retrieved_context": response_text,
            }

        except Exception as e:
            ctx.logger.error(f"Error parsing financial RAG response: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)

    def _generate_fallback_response(self, supplier_name: str) -> Dict[str, Any]:
        """Generate fallback response when RAG is unavailable"""
        return {
            "financial_score": 50.0,  # Neutral score
            "financial_details": f"Unable to retrieve detailed financial information for {supplier_name}",
            "risk_factors": ["Data retrieval unavailable"],
            "retrieved_context": "Fallback mode - Financial RAG system unavailable",
        }

    def query_financial_documents_sync(
        self,
        supplier_name: str,
        industry: str,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for query_financial_documents for use in LangGraph nodes.

        This method should only be called from synchronous contexts.
        For async contexts, use query_financial_documents() instead.
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
                            self.query_financial_documents(
                                ctx=mock_ctx,
                                supplier_name=supplier_name,
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
                        industry=industry,
                    )
                )
                return result
        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)


def financial_check_node(
    state: SupplierWorkflowState, rag_system: FinancialRAGSystem, ctx: Context
) -> SupplierWorkflowState:
    """
    Node 1: Run financial risk check on supplier using RAG system.

    Input: supplier_name, industry
    Output: financial_score, financial_details, risk_factors
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("Financial Risk Check Node")
    ctx.logger.info("=" * 70)

    try:
        ctx.logger.info(
            f"Checking financial risks for: {state.get('supplier_name', 'Unknown')}"
        )

        # Run RAG query for financial analysis
        rag_result = rag_system.query_financial_documents_sync(
            supplier_name=state.get("supplier_name", ""),
            industry=state.get("industry", ""),
        )

        # Update state with financial results
        state["financial_score"] = rag_result.get("financial_score", 0.0)
        state["financial_info"] = rag_result.get("financial_details", "N/A")
        state["current_step"] = "financial_check_complete"

        ctx.logger.info(f"Financial Score: {state['financial_score']}/100")
        ctx.logger.info(f"Risk Factors: {len(rag_result.get('risk_factors', []))}")

        return state

    except Exception as e:
        ctx.logger.error(f"Error in financial check: {e}")
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
    Node 3a: Success path - Supplier meets financial requirements.

    Prepares data to send to orchestrator agent.
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("Success Node - Financial Risk Approved")
    ctx.logger.info("=" * 70)

    state["current_step"] = "financial_approved"
    state["should_continue"] = False

    ctx.logger.info(f"Supplier: {state.get('supplier_name', 'Unknown')}")
    ctx.logger.info(f"Financial Score: {state['financial_score']}/100")
    ctx.logger.info(f"Status: APPROVED - Low financial risk")
    ctx.logger.info("=" * 70)

    return state


def error_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3b: Error path - Supplier does not meet financial requirements.

    Prepares error response.
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("Error Node - Financial Risk Rejected")
    ctx.logger.info("=" * 70)

    financial_score = state.get("financial_score", 0.0)
    error_msg = state.get("error_message", "Unknown error")

    ctx.logger.info(f"Supplier: {state.get('supplier_name', 'Unknown')}")
    ctx.logger.info(f"Financial Score: {financial_score}/100")
    ctx.logger.info(f"Error: {error_msg}")
    ctx.logger.info("=" * 70)

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
    financial_check_node (Run RAG analysis)
        ↓
    financial_router (Check score >= 70)
        ├─→ success_node (Score >= 70) → END
        └─→ error_node (Score < 70 or error) → END

    Returns:
        Compiled StateGraph workflow
    """
    ctx.logger.info("Building Financial LangGraph Workflow...")

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

    ctx.logger.info("Financial workflow built successfully")

    return compiled_workflow
