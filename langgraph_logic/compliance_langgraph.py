"""
Compliance LangGraph Workflow - With Ollama LLM (No Pinecone/RAG)

This module handles compliance analysis by:
1. Scraping B Corp page for supplier ethics and sustainability data
2. Using Ollama LLM to evaluate the scraped data with the compliance prompt
3. Returning compliance score and details

Uses Ollama for LLM reasoning, but no vector stores or RAG systems.
"""

from typing import Dict, Any, List, Optional, Literal
from models.compliance import ComplianceRequest, ComplianceResponse
from langgraph_logic.state_schemas import SupplierWorkflowState
import os
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
from ollama import Client

from langgraph.graph import StateGraph, START, END
from prompts.compliance_prompt import compliance_prompt

# Commented out - RAG/Pinecone not needed
# from llama_index.core import VectorStoreIndex, Settings, Document
# from llama_index.core.node_parser import SimpleNodeParser
# from llama_index.vector_stores.pinecone import PineconeVectorStore
# from llama_index.embeddings.ollama import OllamaEmbedding
# from pinecone import Pinecone, ServerlessSpec
# from llama_index.llms.ollama import Ollama

load_dotenv()

# Commented out - Pinecone not needed
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION")


class ComplianceAnalysisSystem:
    """
    Compliance analysis system - scrapes B Corp page and uses Ollama LLM for analysis.

    Uses Ollama LLM for reasoning and detailed analysis generation.
    No Pinecone or RAG needed - uses scraped data directly with LLM.
    """

    def __init__(self):
        self.initialized = False

        # Initialize Ollama Client for LLM reasoning
        self.ollama_client = Client(host="http://127.0.0.1:11434", timeout=400)
        self.llm_model = "llama3.2:1b"

        # Commented out - RAG/Pinecone not needed
        # self.index = None
        # self.query_engine = None
        # self.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        # Settings.embed_model = self.embed_model
        # Settings.llm = Ollama(model=self.llm_model, request_timeout=400)

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

            # Use WebDriverWait for better timeout control
            wait = WebDriverWait(driver, 15)

            ctx.logger.info(f"Loading page: {company_url}")
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
                    # Check if driver is still alive before continuing
                    if not driver or driver.service.process is None:
                        ctx.logger.warning(f"Driver died, stopping tab scraping")
                        break

                    # Try to find and click the tab button with timeout
                    tab_button = wait.until(
                        EC.presence_of_element_located(
                            (By.XPATH, f"//button[contains(text(), '{tab}')]")
                        )
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
                    ctx.logger.warning(f"Could not scrape {tab} tab: {e}")
                    # Don't continue if driver is dead
                    try:
                        if driver and driver.service.process:
                            continue
                        else:
                            ctx.logger.error("Driver is dead, stopping scraping")
                            break
                    except:
                        ctx.logger.error("Driver check failed, stopping scraping")
                        break

            ctx.logger.info(f"Successfully scraped data for {supplier_name}")
            return scraped_data

        except Exception as e:
            ctx.logger.error(f"Error scraping B Corp page: {e}")
            import traceback

            traceback.print_exc()

            return {
                "supplier_name": supplier_name,
                "url": company_url,
                "error": str(e),
            }

        finally:
            # Always cleanup driver in finally block
            if driver:
                try:
                    driver.quit()
                    ctx.logger.info("WebDriver cleaned up successfully")
                except Exception as e:
                    ctx.logger.warning(f"Error closing driver: {e}")

    async def initialize(self, ctx: Context) -> bool:
        """Initialize the compliance analysis system with Ollama LLM"""
        try:
            # Test Ollama connection
            try:
                test_response = self.ollama_client.chat(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": "Hello"}],
                )
                ctx.logger.info(f"Ollama LLM connected: {self.llm_model}")
            except Exception as e:
                ctx.logger.warning(f"Ollama connection test failed: {e}")
                ctx.logger.info("Will use fallback analysis without LLM")

            # Commented out - Pinecone not needed
            # if not await self._setup_pinecone(ctx):
            #     return False

            ctx.logger.info(
                "Compliance Analysis System initialized (with Ollama LLM, no Pinecone)"
            )
            self.initialized = True
            return True

        except Exception as e:
            ctx.logger.error(f"Failed to initialize: {e}")
            import traceback

            traceback.print_exc()
            return False

    # Commented out - Pinecone not needed
    # async def _setup_pinecone(self, ctx: Context) -> bool:
    #     """Setup Pinecone vector store"""
    #     ...

    # async def _index_documents(self, ctx: Context) -> bool:
    #     """Index compliance documents in Pinecone"""
    #     ...

    # async def _index_scraped_document(self, ctx: Context, document) -> bool:
    #     """Index a single scraped document in Pinecone"""
    #     ...

    # async def _check_supplier_in_pinecone(self, ctx: Context, supplier_name: str) -> bool:
    #     """Check if supplier data already exists in Pinecone"""
    #     ...

    def _query_llm_for_analysis(
        self,
        ctx: Context,
        prompt: str,
        supplier_name: str,
    ) -> str:
        """
        Query Ollama LLM for detailed compliance analysis.

        Args:
            ctx: Context for logging
            prompt: The compliance prompt to send to LLM
            supplier_name: Name of the supplier

        Returns:
            LLM-generated analysis text
        """
        try:
            ctx.logger.info(f"Querying Ollama LLM for compliance analysis...")

            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a compliance analyst specializing in ethics, sustainability, and corporate responsibility. Analyze the provided data and provide structured scores and summaries.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            llm_response = response["message"]["content"]
            ctx.logger.info(f"LLM analysis received ({len(llm_response)} chars)")
            return llm_response

        except Exception as e:
            ctx.logger.warning(f"LLM query failed: {e}")
            return f"Unable to analyze compliance for {supplier_name}. Error: {str(e)}"

    async def query_compliance_documents(
        self,
        ctx: Context,
        supplier_name: str,
        company_values: str,
        industry: str,
        b_corp_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze supplier compliance by scraping B Corp page and using Ollama LLM.

        SIMPLIFIED FLOW (No RAG/Pinecone):
        1. Scrape B Corp page for supplier data
        2. Format scraped data as text
        3. Send to Ollama LLM with compliance prompt
        4. Parse LLM response and extract scores

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
            ctx.logger.error("System not initialized")
            return self._generate_fallback_response(supplier_name)

        try:
            # Construct B Corp URL if not provided
            if not b_corp_url:
                supplier_slug = (
                    supplier_name.lower().replace(" ", "-").replace("_", "-")
                )
                b_corp_url = f"https://www.bcorporation.net/en-us/find-a-b-corp/company/{supplier_slug}"

            ctx.logger.info(f"B Corp URL: {b_corp_url}")

            # ================================================================
            # STEP 1: SCRAPE B CORP PAGE
            # ================================================================
            ctx.logger.info(f"Scraping B Corp data for {supplier_name}...")

            scraped_data = await self.scrape_bcorp_supplier_page(
                ctx, supplier_name, b_corp_url
            )

            if "error" in scraped_data:
                ctx.logger.warning("Scraping failed, using fallback")
                return self._generate_fallback_response(supplier_name)

            # ================================================================
            # STEP 2: FORMAT SCRAPED DATA AS TEXT
            # ================================================================
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

            ctx.logger.info(f"Formatted supplier data for LLM analysis")

            # ================================================================
            # STEP 3: QUERY OLLAMA LLM WITH COMPLIANCE PROMPT
            # ================================================================
            query = compliance_prompt(company_values, industry, supplier_text)
            ctx.logger.info(f"Querying Ollama for compliance analysis...")

            # Query Ollama LLM directly (no RAG)
            llm_response = self._query_llm_for_analysis(ctx, query, supplier_name)

            # ================================================================
            # STEP 4: PARSE RESPONSE
            # ================================================================
            result = await self._parse_llm_response(ctx, llm_response, supplier_name)
            result["selected_supplier"] = supplier_name
            result["scraped_data"] = scraped_data

            return result

        except Exception as e:
            ctx.logger.error(f"Query failed: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)

    async def _parse_llm_response(
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
                    content = line.split(":", 1)[-1].strip()
                    if not content or len(content) < 20:
                        for next_line in lines[i + 1 :]:
                            next_line_lower = next_line.lower()
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
                    content = line.split(":", 1)[-1].strip()
                    if not content or len(content) < 20:
                        for next_line in lines[i + 1 :]:
                            next_line_lower = next_line.lower()
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
            ctx.logger.error(f"Error parsing LLM response: {e}")
            import traceback

            traceback.print_exc()
            return self._generate_fallback_response(supplier_name)

    def _generate_fallback_response(self, supplier_name: str) -> Dict[str, Any]:
        """Generate fallback response when analysis fails"""
        return {
            "compliance_score": 50.0,
            "ethics_info": f"Unable to retrieve detailed ethics information for {supplier_name}",
            "sustainability_info": f"Unable to retrieve detailed sustainability information for {supplier_name}",
            "violations": ["Data retrieval unavailable"],
            "retrieved_context": "Fallback mode - analysis unavailable",
        }

    # Commented out - Sync wrapper not needed without RAG
    # def query_compliance_documents_sync(self, ...):
    #     """Synchronous wrapper for query_compliance_documents"""
    #     ...


# ============================================================================
# LANGGRAPH WORKFLOW NODES
# ============================================================================


def compliance_check_node(
    state: SupplierWorkflowState,
    analysis_system: ComplianceAnalysisSystem,
    ctx: Context,
) -> SupplierWorkflowState:
    """
    Node 1: Run compliance check on supplier using Ollama LLM.

    Process:
    1. Scrape B Corp webpage for supplier
    2. Send scraped data to Ollama with compliance prompt
    3. Parse response and extract compliance score
    4. Return compliance score and details
    """
    ctx.logger.info("=" * 70)
    ctx.logger.info("[LangGraph Node: compliance_check] SCRAPING & ANALYZING")
    ctx.logger.info("=" * 70)

    try:
        supplier_name = state.get("supplier_name", "Unknown")
        b_corp_url = state.get("b_corp_profile_url")
        company_values = state.get("company_values", "")
        industry = state.get("industry", "")

        ctx.logger.info(f"Analyzing compliance for: {supplier_name}")

        # Run analysis synchronously
        import asyncio
        import concurrent.futures

        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    analysis_system.query_compliance_documents(
                        ctx=ctx,
                        supplier_name=supplier_name,
                        company_values=company_values,
                        industry=industry,
                        b_corp_url=b_corp_url,
                    )
                )
            finally:
                new_loop.close()

        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(run_async)
                result = future.result(timeout=300)
        except RuntimeError:
            result = asyncio.run(
                analysis_system.query_compliance_documents(
                    ctx=ctx,
                    supplier_name=supplier_name,
                    company_values=company_values,
                    industry=industry,
                    b_corp_url=b_corp_url,
                )
            )

        # Update state with compliance results
        state["compliance_score"] = result.get("compliance_score", 0.0)
        state["ethics_info"] = result.get("ethics_info", "N/A")
        state["sustainability_info"] = result.get("sustainability_info", "N/A")
        state["violations"] = result.get("violations", [])
        state["rag_context"] = result.get("retrieved_context", "")
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
    - "success_node" if score >= 60 (APPROVED)
    - "error_node" if score < 60 OR error occurred (REJECTED)
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
    """Node 3a: Success path - Supplier meets compliance requirements."""
    ctx.logger.info("=" * 70)
    ctx.logger.info("SUCCESS NODE - COMPLIANCE APPROVED")
    ctx.logger.info("=" * 70)

    state["current_step"] = "compliance_approved"
    state["should_continue"] = False

    ctx.logger.info(f"Supplier: {state.get('supplier_name', 'Unknown')}")
    ctx.logger.info(f"Score: {state['compliance_score']}/100")
    ctx.logger.info(f"Status: APPROVED")
    ctx.logger.info("=" * 70)

    return state


def error_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """Node 3b: Error path - Supplier does not meet requirements."""
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
    analysis_system: ComplianceAnalysisSystem, ctx: Context
) -> StateGraph:
    """
    Build the LangGraph compliance workflow.

    Flow:
    START
        |
    compliance_check_node (Scrape B Corp + Analyze with Ollama)
        |
    compliance_router (Check score >= 60)
        |--- success_node (Score >= 60) --> END
        |--- error_node (Score < 60 or error) --> END

    Returns:
        Compiled StateGraph workflow
    """
    ctx.logger.info("Building Compliance LangGraph Workflow...")

    # Create state graph
    workflow = StateGraph(SupplierWorkflowState)

    # Add nodes
    workflow.add_node(
        "compliance_check",
        lambda state: compliance_check_node(state, analysis_system, ctx),
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

    ctx.logger.info(
        "Compliance workflow built successfully (with Ollama LLM, no Pinecone)"
    )

    return compiled_workflow
