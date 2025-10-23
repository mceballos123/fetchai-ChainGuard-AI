from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from backend.models.compliance import ComplianceRequest, ComplianceResponse
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

load_dotenv()

FILE_PATH = os.getenv("FILE_PATH_DOCUMENTS_COMPLIANCE")


COMPLIANCE_FILES_DIR = Path(FILE_PATH)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_DIMENSION = os.getenv(
    "EMBEDDING_DIMENSION"
)

class ComplianceRAGSystem:
    """RAG system for compliance document retrieval and analysis using local files"""

    def __init__(self):
        self.initialized = False
        self.index = None
        self.query_engine = None

        # Initialize Ollama LLM
        self.llm = Ollama(model="llama3.2:1b", request_timeout=120)
        self.llm_type = "ollama"

        # Initialize Ollama embeddings
        self.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Configure Settings for LLamaIndex (BEFORE any Pinecone initialization!)
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.documents = []

    async def initialize(self, ctx: Context) -> bool:

        try:
            ctx.logger.info(" Initializing Compliance RAG System...")

            # Step 1: Load compliance files
            ctx.logger.info(f"Loading compliance files from {COMPLIANCE_FILES_DIR}")
            if not await self._load_compliance_files(ctx):
                return False

            # Step 2: Initialize Pinecone
            ctx.logger.info("Initializing Pinecone vector store...")
            if not await self._setup_pinecone(ctx):
                return False

            # Step 3: Index documents in Pinecone
            ctx.logger.info("Indexing compliance documents...")
            if not await self._index_documents(ctx):
                return False

            self.initialized = True
            ctx.logger.info("Compliance RAG System initialized successfully!")
            return True

        except Exception as e:
            ctx.logger.error(f"Failed to initialize RAG system: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _load_compliance_files(self, ctx: Context) -> bool:
        """Load compliance files from directory"""
        try:
            if not COMPLIANCE_FILES_DIR.exists():
                ctx.logger.error(
                    f" Compliance files directory not found: {COMPLIANCE_FILES_DIR}"
                )
                return False

            # Use LlamaIndex to load documents
            reader = SimpleDirectoryReader(str(COMPLIANCE_FILES_DIR))
            self.documents = reader.load_data()

            if not self.documents:
                ctx.logger.error(" No documents found in compliance_files folder")
                return False

            ctx.logger.info(f"Loaded {len(self.documents)} compliance documents")
            for doc in self.documents:
                ctx.logger.info(f"   - {doc.metadata.get('file_name', 'Unknown')}")

            return True

        except Exception as e:
            ctx.logger.error(f"Error loading compliance files: {e}")
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
            ctx.logger.info("Selecting best supplier match...")
            ctx.logger.info(f" Available suppliers: {len(self.documents)}")
            for doc in self.documents:
                supplier_name = doc.metadata.get("file_name", "Unknown")
                ctx.logger.info(f"     - {supplier_name}")

            # Build selection query for LLM
            supplier_list = "\n".join(
                [
                    f"- {doc.metadata.get('file_name', 'Unknown')}"
                    for doc in self.documents
                ]
            )

            selection_query = f"""
            Based on the provided supplier documents, select the ONE supplier 
            that BEST aligns with these company values: {company_values}
            Industry: {industry}
            
            Available suppliers:
            {supplier_list}
            
            Respond with ONLY the supplier name (file name without .txt) that best matches.
            Example response: "Horizon Coffee Collective" or "Sunrise Sustainable Imports"
            """

            ctx.logger.info("Using LLM to select best supplier...")
            response = self.query_engine.query(selection_query)
            selected_supplier = str(response).strip().lower()

            ctx.logger.info(f"Selected supplier: {selected_supplier}")

            return selected_supplier

        except Exception as e:
            ctx.logger.error(f"Error selecting supplier: {e}")
            # Fallback: return first supplier
            if self.documents:
                fallback = (
                    self.documents[0].metadata.get("file_name", "Unknown").lower()
                )
                ctx.logger.warning(f"Using fallback supplier: {fallback}")
                return fallback
            return None

    async def filter_to_supplier(self, ctx: Context, supplier_name: str) -> bool:
        """
        Filter documents to include only the selected supplier.
        This ensures RAG only retrieves data about the chosen supplier.
        """
        try:
            ctx.logger.info(f"Filtering documents to supplier: {supplier_name}")

            # Find and keep only the selected supplier document
            filtered_docs = []
            for doc in self.documents:
                doc_name = doc.metadata.get("file_name", "").lower()
                if supplier_name.lower() in doc_name.lower():
                    filtered_docs.append(doc)
                    ctx.logger.info(f"Including: {doc_name}")
                else:
                    ctx.logger.info(f"Excluding: {doc_name}")

            if not filtered_docs:
                ctx.logger.warning(f" No documents found for supplier: {supplier_name}")
                return False

            # Re-index with only the selected supplier
            self.documents = filtered_docs

            ctx.logger.info(f" Now analyzing only: {supplier_name}")
            return True

        except Exception as e:
            ctx.logger.error(f"Error filtering documents: {e}")
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
                ctx.logger.info(f" Creating new Pinecone index: {PINECONE_INDEX_NAME}")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                ctx.logger.info(
                    f" Using existing Pinecone index: {PINECONE_INDEX_NAME}"
                )

            # Get Pinecone index
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)

            # Create LlamaIndex vector store
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

            # Create index from vector store
            self.index = VectorStoreIndex.from_vector_store(vector_store)

            # Create query engine with tree summarize
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5, response_mode="tree_summarize", verbose=True
            )

            ctx.logger.info("Pinecone vector store initialized")
            return True

        except Exception as e:
            ctx.logger.error(f"Error setting up Pinecone: {e}")
            return False

    async def _index_documents(self, ctx: Context) -> bool:
        """Index compliance documents in Pinecone"""
        try:
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
            if self.index:
                # Add nodes to index (this will embed and store in Pinecone)
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

    async def query_compliance_documents(
        self,
        ctx: Context,
        supplier_name: str,
        company_values: str,
        industry: str,
    ) -> Dict[str, Any]:
        """
        Query RAG system for supplier compliance information.

        Process:
        1. Select the BEST supplier that matches company values
        2. Retrieve relevant chunks from ONLY that supplier
        3. Use LLM (Ollama/ASI:1) to analyze compliance
        4. Extract ethics and sustainability scores
        5. Return structured compliance data

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
        if not self.initialized or not self.query_engine:
            ctx.logger.error("RAG system not initialized")
            return self._generate_fallback_response("Unknown")

        try:
            # Step 1: Select the BEST supplier matching company values
            ctx.logger.info("=" * 60)
            ctx.logger.info("SUPPLIER SELECTION PHASE")
            ctx.logger.info("=" * 60)

            selected_supplier = await self.select_best_supplier(
                ctx, company_values, industry
            )

            if not selected_supplier:
                ctx.logger.error("Failed to select supplier")
                return self._generate_fallback_response("Unknown")

            # Step 2: Filter documents to ONLY the selected supplier
            ctx.logger.info("\n" + "=" * 60)
            ctx.logger.info("DOCUMENT FILTERING PHASE")
            ctx.logger.info("=" * 60)

            if not await self.filter_to_supplier(ctx, selected_supplier):
                ctx.logger.error("Failed to filter to selected supplier")
                return self._generate_fallback_response(selected_supplier)

            # Step 3: Compliance analysis on the selected supplier ONLY
            ctx.logger.info("\n" + "=" * 60)
            ctx.logger.info("COMPLIANCE ANALYSIS PHASE")
            ctx.logger.info("=" * 60)

            # Build compliance analysis query
            compliance_query = f"""
            Analyze the compliance, ethics, and sustainability of supplier: {selected_supplier}
            Industry: {industry}
            Company values to align with: {company_values}
            
            Based on the provided documents about this supplier, provide:
            1. Ethics score (0-100) - worker treatment, labor practices, fair wages, gender equity
            2. Sustainability score (0-100) - environmental impact, green practices, carbon footprint
            3. Combined compliance score (average of ethics + sustainability)
            4. Summary of ethical practices (2-3 sentences)
            5. Summary of sustainability practices (2-3 sentences)
            6. List any violations or concerns found (or empty list if none)
            
            Format your response as follows:
            ETHICS_SCORE: [number]
            SUSTAINABILITY_SCORE: [number]
            COMBINED_SCORE: [number]
            ETHICS_INFO: [text]
            SUSTAINABILITY_INFO: [text]
            VIOLATIONS: [comma-separated list or "none"]
            """

            ctx.logger.info(f"Analyzing {selected_supplier} (LLM: {self.llm_type})")

            # Query the RAG system with ONLY the selected supplier's documents
            response = self.query_engine.query(compliance_query)

            # Parse the response
            result = await self._parse_rag_response(
                ctx, str(response), selected_supplier
            )

            # Add the selected supplier to the response
            result["selected_supplier"] = selected_supplier

            ctx.logger.info(f"Compliance analysis complete!")
            ctx.logger.info(f"   Supplier: {selected_supplier}")
            ctx.logger.info(f"   Score: {result['compliance_score']}/100")

            return result

        except Exception as e:
            ctx.logger.error(f"Query failed: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response("Unknown")

    async def _parse_rag_response(
        self, ctx: Context, response_text: str, supplier_name: str
    ) -> Dict[str, Any]:
        """Parse LLM response and extract structured compliance data"""
        try:
            ctx.logger.info("Parsing compliance scores from LLM response...")
            ctx.logger.info("=" * 70)
            ctx.logger.info("RAW LLM RESPONSE:")
            ctx.logger.info(response_text)
            ctx.logger.info("=" * 70)

            # Initialize defaults
            ethics_score = 50.0
            sustainability_score = 50.0
            combined_score = 50.0
            ethics_info = "Insufficient data"
            sustainability_info = "Insufficient data"
            violations = []

            # Parse response (handle various formats)
            lines = response_text.split("\n")

            for line in lines:
                line_lower = line.lower()

                if "ethics_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        ethics_score = max(0, min(100, float(score_str)))
                        ctx.logger.info(f"✓ Extracted ETHICS_SCORE: {ethics_score}")
                    except (ValueError, IndexError) as e:
                        ctx.logger.warning(
                            f"Could not parse ethics score from: {line} - {e}"
                        )
                        pass

                elif "sustainability_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        sustainability_score = max(0, min(100, float(score_str)))
                        ctx.logger.info(
                            f"Extracted SUSTAINABILITY_SCORE: {sustainability_score}"
                        )
                    except (ValueError, IndexError) as e:
                        ctx.logger.warning(
                            f"Could not parse sustainability score from: {line} - {e}"
                        )
                        pass

                elif "combined_score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        combined_score = max(0, min(100, float(score_str)))
                        ctx.logger.info(f"✓ Extracted COMBINED_SCORE: {combined_score}")
                    except (ValueError, IndexError) as e:
                        ctx.logger.warning(
                            f"Could not parse combined score from: {line} - {e}"
                        )
                        pass

                elif "ethics_info:" in line_lower:
                    ethics_info = line.split(":", 1)[-1].strip()
                    ctx.logger.info(f"✓ Extracted ETHICS_INFO: {ethics_info[:50]}...")

                elif "sustainability_info:" in line_lower:
                    sustainability_info = line.split(":", 1)[-1].strip()
                    ctx.logger.info(
                        f"✓ Extracted SUSTAINABILITY_INFO: {sustainability_info[:50]}..."
                    )

                elif "violations:" in line_lower:
                    violations_str = line.split(":", 1)[-1].strip().lower()
                    if violations_str != "none" and violations_str:
                        violations = [v.strip() for v in violations_str.split(",")]
                    ctx.logger.info(f"✓ Extracted VIOLATIONS: {violations}")

            # If combined_score wasn't provided, calculate it
            if combined_score == 50.0:
                combined_score = (ethics_score + sustainability_score) / 2
                ctx.logger.info(
                    f"ℹ️  Combined score calculated from ethics+sustainability: {combined_score}"
                )

            ctx.logger.info("=" * 70)
            ctx.logger.info(f"FINAL PARSED SCORES:")
            ctx.logger.info(f"  Ethics: {ethics_score}/100")
            ctx.logger.info(f"  Sustainability: {sustainability_score}/100")
            ctx.logger.info(f"  Combined (Compliance): {combined_score}/100")
            ctx.logger.info("=" * 70)

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
                            )
                        )
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(run_async)
                    result = future.result(timeout=300)  # 5 minute timeout
                    return result
            except RuntimeError:
                # No event loop running, use asyncio.run()
                result = asyncio.run(
                    self.query_compliance_documents(
                        ctx=mock_ctx,
                        supplier_name=supplier_name,
                        company_values=company_values,
                        industry=industry,
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
    Node 1: Run compliance check on supplier using RAG system.

    Input: supplier_name, company_values, industry
    Output: compliance_score, ethics_info, sustainability_info, violations
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("🔍 COMPLIANCE CHECK NODE")
    ctx.logger.info("=" * 70)

    try:
        ctx.logger.info(
            f"Checking compliance for: {state.get('supplier_name', 'Unknown')}"
        )

        # Run RAG query for compliance
        rag_result = rag_system.query_compliance_documents_sync(
            supplier_name=state.get("supplier_name", ""),
            company_values=state.get("company_values", ""),
            industry=state.get("industry", ""),
        )

        # Update state with compliance results
        state["compliance_score"] = rag_result.get("compliance_score", 0.0)
        state["ethics_info"] = rag_result.get("ethics_info", "N/A")
        state["sustainability_info"] = rag_result.get("sustainability_info", "N/A")
        state["violations"] = rag_result.get("violations", [])
        state["rag_context"] = rag_result.get("retrieved_context", "")
        state["current_step"] = "compliance_check_complete"

        ctx.logger.info(f"✓ Compliance Score: {state['compliance_score']}/100")
        ctx.logger.info(f"✓ Violations: {len(state['violations'])}")

        return state

    except Exception as e:
        ctx.logger.error(f"❌ Error in compliance check: {e}")
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

    # Check compliance threshold
    if compliance_score >= 75:
        return "success_node"
    else:
        state["error_message"] = (
            f"Compliance score {compliance_score}/100 below threshold (75). Supplier rejected."
        )
        return "error_node"


def success_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """
    Node 3a: Success path - Supplier meets compliance requirements.

    Prepares data to send to orchestrator agent.
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("✅ SUCCESS NODE - COMPLIANCE APPROVED")
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
    ctx.logger.info("❌ ERROR NODE - COMPLIANCE REJECTED")
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
    ctx.logger.info("🏗️ Building Compliance LangGraph Workflow...")

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

    ctx.logger.info("✓ Compliance workflow built successfully!")

    return compiled_workflow
