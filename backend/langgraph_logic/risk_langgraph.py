from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from backend.models.risk import RiskRequest, RiskResponse
from backend.langgraph_logic.state_schemas import SupplierWorkflowState
import os
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
            print(f"✓ Using ENV path: {resolved}")
            return resolved

    # Option 2: Try relative path from backend directory
    relative_path = Path(__file__).parent.parent / "risk_management_files"
    if relative_path.exists():
        print(f"✓ Using relative path: {relative_path}")
        return relative_path

    # Option 3: Try from current working directory
    cwd_path = Path.cwd() / "backend" / "risk_management_files"
    if cwd_path.exists():
        print(f"✓ Using CWD path: {cwd_path}")
        return cwd_path

    # Fallback: Return the most likely path
    print(f"⚠ No risk management files found in any expected location")
    print(f"  Checked: {FILE_PATH}, {relative_path}, {cwd_path}")
    return relative_path


RISK_FILES_DIR = _resolve_risk_path()
EMBEDDING_DIMENSION = 768
print(f"Resolved risk management path: {RISK_FILES_DIR}")
print(f"Path exists: {RISK_FILES_DIR.exists()}")


class RiskRAGSystem:
    """RAG system for risk management document retrieval and analysis"""

    def __init__(self):
        self.initialized = False
        self.index = None
        self.query_engine = None

        # Initialize Ollama LLM
        self.llm = Ollama(model="llama3.2:1b", request_timeout=300)
        self.llm_type = "ollama"

        # Initialize Ollama embeddings
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Configure Settings for LlamaIndex
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.documents = []

    async def initialize(self, ctx: Context) -> bool:
        try:
            ctx.logger.info("Initializing Risk Management RAG System...")

            # Step 1: Load risk management files
            ctx.logger.info(f"Loading risk management files from {RISK_FILES_DIR}")
            if not await self._load_risk_files(ctx):
                return False

            # Step 2: Initialize Pinecone
            ctx.logger.info("Initializing Pinecone vector store...")
            if not await self._setup_pinecone(ctx):
                return False

            # Step 3: Index documents in Pinecone
            ctx.logger.info("Indexing risk management documents...")
            if not await self._index_documents(ctx):
                return False

            self.initialized = True
            ctx.logger.info("Risk Management RAG System initialized successfully!")
            return True

        except Exception as e:
            ctx.logger.error(f"Failed to initialize RAG system: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _load_risk_files(self, ctx: Context) -> bool:
        """Load risk management files from directory"""
        try:
            if not RISK_FILES_DIR.exists():
                ctx.logger.error(
                    f"Risk management files directory not found: {RISK_FILES_DIR}"
                )
                ctx.logger.error(f"Current working directory: {Path.cwd()}")
                return False

            # List files before loading
            files_in_dir = list(RISK_FILES_DIR.glob("*.txt"))
            ctx.logger.info(f"Files found in directory: {len(files_in_dir)}")
            for f in files_in_dir:
                ctx.logger.info(f"   - {f.name}")

            # Use LlamaIndex to load documents
            reader = SimpleDirectoryReader(str(RISK_FILES_DIR))
            self.documents = reader.load_data()

            if not self.documents:
                ctx.logger.error("No documents found in risk_management_files folder")
                return False

            # Ensure we have all 5 expected documents
            expected_count = 5
            if len(self.documents) < expected_count:
                ctx.logger.warning(
                    f"Only loaded {len(self.documents)} documents, expected {expected_count}"
                )

            ctx.logger.info(
                f"✓ Successfully loaded {len(self.documents)} risk management documents"
            )
            for i, doc in enumerate(self.documents, 1):
                file_name = doc.metadata.get("file_name", "Unknown")
                ctx.logger.info(f"   {i}. {file_name}")

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
                    ctx.logger.info(f"✓ MATCHED: {doc_file_name}")

            if not filtered_docs:
                ctx.logger.error(
                    f"✗ No documents found for supplier: '{supplier_name}'"
                )
                return False

            # Re-index with only the selected supplier
            self.documents = filtered_docs
            ctx.logger.info(
                f"✓ Filtered to {len(filtered_docs)} document(s) for: {supplier_name}"
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
                similarity_top_k=5, response_mode="tree_summarize", verbose=True
            )

            ctx.logger.info("Pinecone vector store initialized")
            return True

        except Exception as e:
            ctx.logger.error(f"Error setting up Pinecone: {e}")
            return False

    async def _index_documents(self, ctx: Context) -> bool:
        """Index risk management documents in Pinecone"""
        try:
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
            if self.index:
                for node in nodes:
                    try:
                        self.index.insert_nodes([node])
                    except Exception as e:
                        ctx.logger.warning(f"Error indexing node: {e}")

                ctx.logger.info("Documents indexed in Pinecone")
            else:
                ctx.logger.warning("Index not initialized, skipping document indexing")

            return True

        except Exception as e:
            ctx.logger.error(f"Error indexing documents: {e}")
            return False

    async def query_risk_documents(
        self,
        ctx: Context,
        supplier_name: str,
        industry: str,
    ) -> Dict[str, Any]:
        """
        Query RAG system for supplier risk management information.

        Returns:
        {
            "risk_score": float (0-100),
            "risk_details": str,
            "risk_factors": List[str],
            "retrieved_context": str
        }
        """
        if not self.initialized or not self.query_engine:
            ctx.logger.error("RAG system not initialized")
            return self._generate_fallback_response("Unknown")

        try:
            ctx.logger.info("=" * 60)
            ctx.logger.info("RISK MANAGEMENT ANALYSIS")
            ctx.logger.info("=" * 60)

            # Filter to selected supplier
            if not await self.filter_to_supplier(ctx, supplier_name):
                ctx.logger.error("Failed to filter to selected supplier")
                return self._generate_fallback_response(supplier_name)

            # Build risk query
            risk_query = f"""
            RISK MANAGEMENT ANALYSIS: {supplier_name} ({industry})
            
            Score 0-100 (higher score = lower risk, better supplier):
            - 85-100: Excellent risk management, minimal vulnerabilities
            - 70-84: Good risk management, acceptable with minor considerations (APPROVED)
            - 60-69: Moderate risk, requires careful evaluation (NEEDS REVIEW)
            - 50-59: Significant risks requiring mitigation plans
            - Below 50: High risk, major concerns present
            
            SCORING GUIDANCE:
            - Start with a baseline of 75 points for standard operations
            - Add points (+5-15) for: strong infrastructure, disaster preparedness, diversified supply chains, excellent logistics
            - Deduct points (-5-15) for: capacity issues, high disaster exposure, poor logistics, supply chain concentration
            - Be realistic: most functional suppliers should score 70-85
            - Approval threshold is 70 points - suppliers scoring 70+ are acceptable
            - Only score below 50 if there are serious, documented concerns that could impact the supplier's ability to fulfill the order or the supplier's reputation.
            
            Evaluate these risk factors:
            - Capacity constraints (can they handle demand?)
            - Natural disaster exposure (location vulnerabilities including weather event that could impact the supplier)
            - Logistics and accessibility (transportation, infrastructure)
            - Supply chain dependencies (single points of failure?)
            
            Provide a balanced assessment:
            1. Risk score (0-100) - Be fair and realistic
            2. Risk summary (2-3 sentences): highlight both strengths and concerns
            3. Risk factors: list specific concerns, or "minimal risks identified" even if the supplier is approved.
            
            Response format:
            RISK_SCORE: [number]
            RISK_DETAILS: [balanced 2-3 sentence summary]
            RISK_FACTORS: [specific issues or "minimal risks identified even if the supplier is approved"]
            """

            ctx.logger.info("Executing risk management query...")
            response = self.query_engine.query(risk_query)
            response_text = str(response)

            # Parse the response
            result = await self._parse_rag_response(ctx, response_text, supplier_name)

            ctx.logger.info(f"Risk management analysis complete!")
            ctx.logger.info(f"   Supplier: {supplier_name}")
            ctx.logger.info(f"   Score: {result['risk_score']}/100")

            return result

        except Exception as e:
            ctx.logger.error(f"Query failed: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response("Unknown")

    async def _parse_rag_response(
        self, ctx: Context, response_text: str, supplier_name: str
    ) -> Dict[str, Any]:
        """Parse LLM response and extract structured risk data"""
        try:
            ctx.logger.info("Parsing risk scores from LLM response...")
            ctx.logger.info("=" * 50)
            ctx.logger.info("RAW LLM RESPONSE:")
            ctx.logger.info(response_text)
            ctx.logger.info("=" * 50)

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
                        ctx.logger.info(f"✓ Extracted RISK_SCORE: {risk_score}")
                    except (ValueError, IndexError) as e:
                        ctx.logger.warning(
                            f"Could not parse risk score from: {line} - {e}"
                        )

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
                        ctx.logger.info(
                            f"✓ Extracted RISK_DETAILS: {risk_details[:60]}..."
                        )

                elif "risk_factors:" in line_lower:
                    factors_str = line.split(":", 1)[-1].strip().lower()
                    if (
                        factors_str
                        and "minimal" not in factors_str
                        and "none" not in factors_str
                    ):
                        risk_factors = [f.strip() for f in factors_str.split(",")]
                    ctx.logger.info(f"✓ Extracted RISK_FACTORS: {risk_factors}")

            ctx.logger.info("=" * 70)
            ctx.logger.info(f"FINAL PARSED SCORES:")
            ctx.logger.info(f"  Risk Score: {risk_score}/100")
            ctx.logger.info(
                f"  Risk Details Extracted: {risk_details != 'Insufficient risk data'}"
            )
            ctx.logger.info(f"  Risk Factors Count: {len(risk_factors)}")
            ctx.logger.info("=" * 70)

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
                            )
                        )
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(run_async)
                    result = future.result(timeout=300)
                    return result
            except RuntimeError:
                # No event loop running, use asyncio.run()
                result = asyncio.run(
                    self.query_risk_documents(
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


def risk_check_node(
    state: SupplierWorkflowState, rag_system: RiskRAGSystem, ctx: Context
) -> SupplierWorkflowState:
    """
    Node 1: Run risk management check on supplier using RAG system.

    Input: supplier_name, industry
    Output: risk_score, risk_details, risk_factors
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("🔍 RISK MANAGEMENT CHECK NODE")
    ctx.logger.info("=" * 70)

    try:
        ctx.logger.info(
            f"Checking risk management for: {state.get('supplier_name', 'Unknown')}"
        )

        # Run RAG query for risk management
        rag_result = rag_system.query_risk_documents_sync(
            supplier_name=state.get("supplier_name", ""),
            industry=state.get("industry", ""),
        )

        # Update state with risk results
        state["risk_score"] = rag_result.get("risk_score", 0.0)
        state["risk_details"] = rag_result.get("risk_details", "N/A")
        state["risk_factors"] = rag_result.get("risk_factors", [])
        state["current_step"] = "risk_check_complete"

        ctx.logger.info(f"✓ Risk Score: {state['risk_score']}/100")
        ctx.logger.info(f"✓ Risk Factors: {len(state['risk_factors'])}")

        return state

    except Exception as e:
        ctx.logger.error(f"Error in risk check: {e}")
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
    if risk_score >= 70:
        return "success_node"
    else:
        state["error_message"] = (
            f"Risk score {risk_score}/100 below threshold (70). Supplier rejected."
        )
        return "error_node"


def success_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3a: Success path - Supplier meets risk management requirements.
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("✅ SUCCESS NODE - RISK MANAGEMENT APPROVED")
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
    risk_check_node (Run RAG analysis)
        ↓
    risk_router (Check score >= 70)
        ├─→ success_node (Score >= 70) → END
        └─→ error_node (Score < 70 or error) → END

    Returns:
        Compiled StateGraph workflow
    """
    ctx.logger.info("🏗️ Building Risk Management LangGraph Workflow...")

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

    ctx.logger.info("✓ Risk management workflow built successfully!")

    return compiled_workflow
