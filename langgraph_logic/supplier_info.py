"""
LangGraph workflow for extracting detailed supplier information from SEC filings
"""

from typing import Optional, Literal
from langgraph.graph import StateGraph, START, END
from models.supplier_info import SupplierInfoState
import requests
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time


def determine_company_type_node(state: SupplierInfoState) -> SupplierInfoState:
    """
    Node 1: Determine if company is foreign or American based on SEC data
    """
    ctx = state.get("ctx")
    cik = state["cik"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: determine_company_type] Analyzing CIK: {cik}")

    try:
        # Get company information from SEC to determine if it's foreign
        # We'll check the company's filings to see if they file 20-F (foreign) or 10-Q (US)
        sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=40&search_text="

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        response = requests.get(sec_url, headers=headers, timeout=15)
        response.raise_for_status()

        content = response.text

        # Check if company files 20-F (foreign) or 10-Q (US domestic)
        has_20f = "20-F" in content
        has_10q = "10-Q" in content

        if has_20f:
            is_foreign = True
            filing_type = "20-F"
            if ctx:
                ctx.logger.info(f"✓ Company is FOREIGN - will look for 20-F filings")
        elif has_10q:
            is_foreign = False
            filing_type = "10-Q"
            if ctx:
                ctx.logger.info(f"✓ Company is AMERICAN - will look for 10-Q filings")
        else:
            # Default to 10-K for US companies if neither found
            is_foreign = False
            filing_type = "10-K"
            if ctx:
                ctx.logger.info(f"⚠ Could not determine filing type, defaulting to 10-K")

        return {
            **state,
            "is_foreign": is_foreign,
            "filing_type": filing_type,
            "score": 100
        }

    except Exception as e:
        if ctx:
            ctx.logger.error(f"Error determining company type: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error determining company type: {str(e)}",
            "score": 0
        }


def find_latest_filing_node(state: SupplierInfoState) -> SupplierInfoState:
    """
    Node 2: Find the latest filing URL for the company
    """
    ctx = state.get("ctx")
    cik = state["cik"]
    filing_type = state["filing_type"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: find_latest_filing] Looking for latest {filing_type} filing")

    try:
        # Navigate to SEC EDGAR browse page
        edgar_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={filing_type}&dateb=&owner=exclude&count=10"

        if ctx:
            ctx.logger.info(f"Navigating to: {edgar_url}")

        # Set up Selenium WebDriver in headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(edgar_url)

        # Wait for the page to load
        time.sleep(3)

        # Find the first (latest) filing document link
        # Look for the "Documents" button in the first row
        try:
            documents_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "documentsbutton"))
            )
            filing_url = documents_button.get_attribute("href")

            if ctx:
                ctx.logger.info(f"✓ Found latest {filing_type} filing: {filing_url}")

            driver.quit()

            return {
                **state,
                "latest_filing_url": filing_url,
                "score": 100
            }

        except Exception as e:
            driver.quit()
            if ctx:
                ctx.logger.error(f"Could not find filing link: {e}")
            return {
                **state,
                "success": False,
                "error_message": f"Could not find latest {filing_type} filing",
                "score": 0
            }

    except Exception as e:
        if ctx:
            ctx.logger.error(f"Error finding latest filing: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error finding latest filing: {str(e)}",
            "score": 0
        }


def extract_company_info_node(state: SupplierInfoState) -> SupplierInfoState:
    """
    Node 3: Extract company description and history from the filing
    """
    ctx = state.get("ctx")
    filing_url = state.get("latest_filing_url")
    filing_type = state["filing_type"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: extract_company_info] Extracting company info from filing")

    if not filing_url:
        return {
            **state,
            "success": False,
            "error_message": "No filing URL available",
            "score": 0
        }

    try:
        # Set up Selenium to navigate to the filing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(filing_url)

        time.sleep(3)

        # Look for the main filing document (usually ends with .htm or .html)
        try:
            # Find all document links
            table = driver.find_element(By.CLASS_NAME, "tableFile")
            rows = table.find_elements(By.TAG_NAME, "tr")

            main_doc_url = None
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) > 3:
                    doc_type = cells[3].text.strip()
                    if filing_type in doc_type or doc_type == filing_type:
                        link = cells[2].find_element(By.TAG_NAME, "a")
                        main_doc_url = link.get_attribute("href")
                        break

            if not main_doc_url:
                # Fallback: get the first .htm document
                for row in rows[1:]:  # Skip header
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) > 2:
                        link_cell = cells[2]
                        if link_cell.text.endswith('.htm') or link_cell.text.endswith('.html'):
                            link = link_cell.find_element(By.TAG_NAME, "a")
                            main_doc_url = link.get_attribute("href")
                            break

            if main_doc_url:
                if ctx:
                    ctx.logger.info(f"Found main document: {main_doc_url}")

                # Navigate to the main document
                driver.get(main_doc_url)
                time.sleep(3)

                # Get the page text
                page_text = driver.find_element(By.TAG_NAME, "body").text

                # Extract company information using keywords
                company_description = ""
                history = ""

                # Look for "History and Development of the Company" section
                history_patterns = [
                    r"History and Development of the Company[:\.]?\s*(.{100,2000})",
                    r"4\.A\.\s*History and Development[:\.]?\s*(.{100,2000})",
                    r"Item 4\.?\s*Information on the Company[:\.]?\s*(.{100,2000})",
                ]

                for pattern in history_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        history = match.group(1).strip()[:1000]  # Limit to 1000 chars
                        break

                # Look for business description
                business_patterns = [
                    r"(?:Business|Our Business|The Business)[:\.]?\s*(.{100,1500})",
                    r"Item 1\.?\s*Business[:\.]?\s*(.{100,1500})",
                    r"Description of Business[:\.]?\s*(.{100,1500})",
                ]

                for pattern in business_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        company_description = match.group(1).strip()[:1000]
                        break

                # If we didn't find specific sections, use the history as description
                if not company_description and history:
                    company_description = history
                elif not company_description:
                    # Fallback: get first few paragraphs
                    company_description = page_text[:1000]

                driver.quit()

                if ctx:
                    ctx.logger.info(f"✓ Extracted company information (description: {len(company_description)} chars)")

                return {
                    **state,
                    "company_description": company_description,
                    "history": history,
                    "success": True,
                    "score": 100
                }
            else:
                driver.quit()
                return {
                    **state,
                    "success": False,
                    "error_message": "Could not find main filing document",
                    "score": 0
                }

        except Exception as e:
            driver.quit()
            if ctx:
                ctx.logger.error(f"Error extracting company info: {e}")
            return {
                **state,
                "success": False,
                "error_message": f"Error extracting company info: {str(e)}",
                "score": 0
            }

    except Exception as e:
        if ctx:
            ctx.logger.error(f"Error in extract_company_info: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error extracting company info: {str(e)}",
            "score": 0
        }


def format_results_node(state: SupplierInfoState) -> SupplierInfoState:
    """
    Node 4: Format the extracted information
    """
    ctx = state.get("ctx")

    if ctx:
        ctx.logger.info(f"[LangGraph Node: format_results] Formatting supplier info results")

    if state.get("company_description"):
        if ctx:
            ctx.logger.info(f"✓ Successfully extracted company information")
            ctx.logger.info(f"  Company: {state['company_name']}")
            ctx.logger.info(f"  Type: {'Foreign' if state['is_foreign'] else 'American'}")
            ctx.logger.info(f"  Filing Type: {state['filing_type']}")

    return {
        **state,
        "success": True
    }


def error_node(state: SupplierInfoState) -> SupplierInfoState:
    """
    Node 5: Handle errors
    """
    ctx = state.get("ctx")

    if ctx:
        ctx.logger.error(f"[LangGraph Node: error] {state.get('error_message', 'Unknown error')}")

    return {
        **state,
        "success": False
    }


# ========== Routing Functions ==========

def should_continue_after_type_check(state: SupplierInfoState) -> Literal["find_filing", "error"]:
    """Route after determining company type"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("filing_type"):
        return "find_filing"

    return "error"


def should_continue_after_finding_filing(state: SupplierInfoState) -> Literal["extract_info", "error"]:
    """Route after finding latest filing"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("latest_filing_url"):
        return "extract_info"

    return "error"


def should_continue_to_format(state: SupplierInfoState) -> Literal["format_results", "error"]:
    """Route after extracting company info"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("company_description") or state.get("history"):
        return "format_results"

    return "error"


# ========== Graph Construction ==========

def build_supplier_info_graph():
    """Build the LangGraph workflow for supplier info extraction"""

    workflow = StateGraph(SupplierInfoState)

    # Add nodes
    workflow.add_node("determine_type", determine_company_type_node)
    workflow.add_node("find_filing", find_latest_filing_node)
    workflow.add_node("extract_info", extract_company_info_node)
    workflow.add_node("format_results", format_results_node)
    workflow.add_node("error", error_node)

    # Add edges
    workflow.add_edge(START, "determine_type")

    # Conditional routing after determining company type
    workflow.add_conditional_edges(
        "determine_type",
        should_continue_after_type_check,
        {
            "find_filing": "find_filing",
            "error": "error"
        }
    )

    # Conditional routing after finding filing
    workflow.add_conditional_edges(
        "find_filing",
        should_continue_after_finding_filing,
        {
            "extract_info": "extract_info",
            "error": "error"
        }
    )

    # Conditional routing after extracting info
    workflow.add_conditional_edges(
        "extract_info",
        should_continue_to_format,
        {
            "format_results": "format_results",
            "error": "error"
        }
    )

    # End after formatting or error
    workflow.add_edge("format_results", END)
    workflow.add_edge("error", END)

    return workflow.compile()


# Create the compiled graph
supplier_info_graph = build_supplier_info_graph()
