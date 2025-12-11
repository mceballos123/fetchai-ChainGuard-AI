"""
Compliance LangGraph Workflow - With Google Gemini LLM (No Pinecone/RAG)

This module handles compliance analysis by:
1. Scraping B Corp page for supplier ethics and sustainability data
2. Using Google Gemini LLM to evaluate the scraped data with the compliance prompt
3. Returning compliance score and details

Uses Gemini for LLM reasoning, but no vector stores or RAG systems.
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
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
from selenium.webdriver.common.by import By
from dotenv import load_dotenv
from uagents import Context
import google.generativeai as genai
import json
from langgraph.graph import StateGraph, START, END
from prompts.compliance_prompt import compliance_prompt

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class ComplianceAnalysisSystem:
    """
    Compliance analysis system - scrapes B Corp page and uses Google Gemini LLM for analysis.

    Uses Gemini LLM for reasoning and detailed analysis generation.
    No Pinecone or RAG needed - uses scraped data directly with LLM.
    """

    def __init__(self):
        self.initialized = False

        # Initialize Gemini model with token limits to reduce API costs
        self.gemini_model = genai.GenerativeModel(
            "models/gemini-2.0-flash",  # Alternative: try models/gemini-2.5-pro if safety filters persist
            generation_config={
                "max_output_tokens": 1200,  # Limit response length
                "temperature": 0.3,  # Lower temperature for more focused responses
            },
        )

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

            # Use WebDriverWait for better timeout control
            wait = WebDriverWait(driver, 15)

            ctx.logger.info(f"Loading page: {company_url}")
            driver.get(company_url)
            ctx.logger.info(f"Page loaded, current URL: {driver.current_url}")
            ctx.logger.info(f"Page title: {driver.title}")

            # Wait longer for JavaScript to render
            time.sleep(8)

            # Log page source length to verify content loaded
            ctx.logger.info(f"Page source length: {len(driver.page_source)} chars")

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
                    # For Remote WebDriver, we can't check service.process - use a simple check instead
                    if not driver:
                        ctx.logger.warning("Driver is None, stopping tab scraping")
                        break

                    # Verify driver is responsive (works for both local and remote)
                    try:
                        _ = driver.current_url
                    except Exception:
                        ctx.logger.warning(
                            "Driver not responsive, stopping tab scraping"
                        )
                        break

                    # Try to find and click the tab button with timeout
                    tab_button = wait.until(
                        EC.presence_of_element_located(
                            (By.XPATH, f"//button[contains(text(), '{tab}')]")
                        )
                    )
                    driver.execute_script("arguments[0].click();", tab_button)
                    time.sleep(5)

                    # Get updated page content
                    tab_content = ""

                    # Strategy 1: Get visible content directly from Selenium
                    try:
                        # Find visible tabpanel
                        content_elements = driver.find_elements(
                            By.XPATH, "//div[@role='tabpanel' and not(@hidden)]"
                        )
                        for elem in content_elements:
                            if elem.is_displayed():
                                text = elem.text.strip()
                                if len(text) > len(tab_content):
                                    tab_content = text
                    except Exception as e:
                        ctx.logger.debug(f"Strategy 1 failed: {e}")

                    # Strategy 2: Get main content area if Strategy 1 failed
                    if not tab_content or len(tab_content) < 100:
                        try:
                            main_elem = driver.find_element(By.XPATH, "//main")
                            if main_elem:
                                tab_content = main_elem.text.strip()
                        except Exception as e:
                            ctx.logger.debug(f"Strategy 2 failed: {e}")

                    # Store content if we found any - OPTIMIZED: Reduced from 2000 to 500 chars to save tokens
                    if tab_content and len(tab_content) > 50:
                        scraped_data[tab.lower()] = tab_content[:500]

                except Exception as e:
                    # Check if driver is still responsive before continuing
                    try:
                        if driver:
                            _ = driver.current_url  # Quick responsiveness check
                            continue
                        else:
                            break
                    except Exception:
                        break

            ctx.logger.info(f"Scraping complete for {supplier_name}")
            return scraped_data

        except Exception as e:
            ctx.logger.error(f"Scraping failed for {supplier_name}: {str(e)}")

            return {
                "supplier_name": supplier_name,
                "url": company_url,
                "error": f"{type(e).__name__}: {str(e)}",
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
        """Initialize the compliance analysis system with Google Gemini LLM"""
        try:
            # Skip test call to save tokens
            ctx.logger.info("Compliance Analysis System initialized")
            self.initialized = True
            return True

        except Exception as e:
            ctx.logger.error(f"Failed to initialize: {e}")
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
        Query Google Gemini LLM for detailed compliance analysis.

        Args:
            ctx: Context for logging
            prompt: The compliance prompt to send to LLM
            supplier_name: Name of the supplier

        Returns:
            LLM-generated analysis text
        """
        try:
            # Build the full prompt with system context - OPTIMIZED: Shorter system prompt
            full_prompt = f"""You are a compliance analyst. Analyze the supplier data and provide scores.

IMPORTANT: Return your analysis in this exact JSON format:
{{
    "ETHICS_SCORE": <number 0-100>,
    "SUSTAINABILITY_SCORE": <number 0-100>,
    "COMBINED_SCORE": <number 0-100>,
    "ETHICS_INFO": "<brief summary>",
    "SUSTAINABILITY_INFO": "<brief summary>",
    "VIOLATIONS": []
}}

{prompt}"""

            response = self.gemini_model.generate_content(full_prompt)

            # Check if response was blocked by safety filters
            if not response.candidates:
                return f"ETHICS_SCORE: 50\nSUSTAINABILITY_SCORE: 50\nCOMBINED_SCORE: 50\nETHICS_INFO: Analysis blocked by safety filters\nSUSTAINABILITY_INFO: Analysis blocked by safety filters\nVIOLATIONS: None"

            # Check finish reason
            finish_reason = response.candidates[0].finish_reason
            if finish_reason == 2:  # SAFETY block
                return f"ETHICS_SCORE: 50\nSUSTAINABILITY_SCORE: 50\nCOMBINED_SCORE: 50\nETHICS_INFO: Response blocked by safety filters\nSUSTAINABILITY_INFO: Response blocked by safety filters\nVIOLATIONS: None"
            elif finish_reason == 3:  # RECITATION
                return f"ETHICS_SCORE: 50\nSUSTAINABILITY_SCORE: 50\nCOMBINED_SCORE: 50\nETHICS_INFO: Response blocked due to recitation\nSUSTAINABILITY_INFO: Response blocked due to recitation\nVIOLATIONS: None"

            # Try to get the response text
            try:
                llm_response = response.text
            except Exception:
                return f"ETHICS_SCORE: 50\nSUSTAINABILITY_SCORE: 50\nCOMBINED_SCORE: 50\nETHICS_INFO: Error extracting response\nSUSTAINABILITY_INFO: Error extracting response\nVIOLATIONS: None"

            return llm_response

        except Exception as e:
            return f"ETHICS_SCORE: 50\nSUSTAINABILITY_SCORE: 50\nCOMBINED_SCORE: 50\nETHICS_INFO: LLM query failed\nSUSTAINABILITY_INFO: LLM query failed\nVIOLATIONS: None"

    async def query_compliance_documents(
        self,
        ctx: Context,
        supplier_name: str,
        company_values: str,
        industry: str,
        b_corp_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze supplier compliance by scraping B Corp page and using Gemini LLM.

        SIMPLIFIED FLOW (No RAG/Pinecone):
        1. Scrape B Corp page for supplier data
        2. Format scraped data as text
        3. Send to Gemini LLM with compliance prompt
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

            # Scrape B Corp page

            scraped_data = await self.scrape_bcorp_supplier_page(
                ctx, supplier_name, b_corp_url
            )

            if "error" in scraped_data:
                ctx.logger.error(f"Scraping failed for {supplier_name}")
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

            # Query Gemini LLM
            query = compliance_prompt(company_values, industry, supplier_text)
            llm_response = self._query_llm_for_analysis(ctx, query, supplier_name)

            # Parse response
            result = await self._parse_llm_response(ctx, llm_response, supplier_name)
            ctx.logger.info(f"Compliance score: {result.get('compliance_score')}/100")
            result["selected_supplier"] = supplier_name
            result["scraped_data"] = scraped_data

            return result

        except Exception as e:
            ctx.logger.error(f"Query failed: {e}")
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

            # Try parsing as JSON first (Gemini sometimes returns JSON format)
            try:
                # Extract JSON from markdown code blocks if present
                json_text = response_text.strip()

                # Remove markdown code blocks
                if "```json" in json_text:
                    json_start = json_text.find("```json") + 7
                    json_end = json_text.find("```", json_start)
                    if json_end != -1:
                        json_text = json_text[json_start:json_end].strip()
                elif "```" in json_text:
                    json_start = json_text.find("```") + 3
                    json_end = json_text.find("```", json_start)
                    if json_end != -1:
                        json_text = json_text[json_start:json_end].strip()

                # Parse JSON
                data = json.loads(json_text)

                # Extract values from JSON
                ethics_score = float(data.get("ETHICS_SCORE", 50))
                sustainability_score = float(data.get("SUSTAINABILITY_SCORE", 50))
                combined_score = float(
                    data.get(
                        "COMBINED_SCORE", (ethics_score + sustainability_score) / 2
                    )
                )
                ethics_info = data.get("ETHICS_INFO", "Insufficient data")
                sustainability_info = data.get(
                    "SUSTAINABILITY_INFO", "Insufficient data"
                )

                # Handle violations
                violations_data = data.get("VIOLATIONS", [])
                if isinstance(violations_data, list):
                    violations = violations_data
                elif isinstance(violations_data, str):
                    violations = [violations_data] if violations_data else []

                return {
                    "compliance_score": float(combined_score),
                    "ethics_info": ethics_info or "No ethics information available",
                    "sustainability_info": sustainability_info
                    or "No sustainability information available",
                    "violations": violations,
                    "retrieved_context": response_text,
                }

            except (json.JSONDecodeError, ValueError, KeyError):
                pass

            # Fallback: Parse response as plain text
            lines = response_text.split("\n")

            for i, line in enumerate(lines):
                line_lower = line.lower().strip()

                if "ethics_score:" in line_lower or "ethics score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        # Remove any non-numeric characters except decimal point
                        score_str = ''.join(c for c in score_str if c.isdigit() or c == '.')
                        if score_str:
                            ethics_score = max(0, min(100, float(score_str)))
                    except (ValueError, IndexError):
                        pass

                elif "sustainability_score:" in line_lower or "sustainability score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        # Remove any non-numeric characters except decimal point
                        score_str = ''.join(c for c in score_str if c.isdigit() or c == '.')
                        if score_str:
                            sustainability_score = max(0, min(100, float(score_str)))
                    except (ValueError, IndexError):
                        pass

                elif "combined_score:" in line_lower or "combined score:" in line_lower:
                    try:
                        score_str = line.split(":")[-1].strip().split()[0]
                        # Remove any non-numeric characters except decimal point
                        score_str = ''.join(c for c in score_str if c.isdigit() or c == '.')
                        if score_str:
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
    Node 1: Run compliance check on supplier using Gemini LLM.
    """
    try:
        supplier_name = state.get("supplier_name", "Unknown")
        b_corp_url = state.get("b_corp_profile_url")
        company_values = state.get("company_values", "")
        industry = state.get("industry", "")

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
                # Reduced timeout to 180 seconds (3 min) - scraping + LLM should complete within this
                result = future.result(timeout=180)
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
        state["violations"] = [
            f"Analysis failed: {str(e)}"
        ]  # Set violations to prevent NoneType error
        state["ethics_info"] = "Analysis could not be completed due to an error"
        state["sustainability_info"] = "Analysis could not be completed due to an error"
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
    state["current_step"] = "compliance_approved"
    state["should_continue"] = False
    return state


def error_node(state: SupplierWorkflowState, ctx: Context) -> SupplierWorkflowState:
    """Node 3b: Error path - Supplier does not meet requirements."""
    state["current_step"] = "compliance_rejected"
    state["should_continue"] = False
    return state


# ============================================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================================


def build_compliance_workflow(
    analysis_system: ComplianceAnalysisSystem, ctx: Context
) -> StateGraph:
    """Build the LangGraph compliance workflow"""
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
    return workflow.compile()
