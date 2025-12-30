"""
LangGraph workflow for extracting financial information from SEC 10-Q filings
Focuses on US companies and their quarterly financial reports
"""

from typing import Optional, Literal
from langgraph.graph import StateGraph, START, END
from models.finance_info import FinanceInfoState
import requests
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time


def determine_filing_type_node(state: FinanceInfoState) -> FinanceInfoState:
    """
    Node 1: Determine the appropriate filing type for US companies
    For now, we only support US companies and will look for 10-Q filings
    """
    ctx = state.get("ctx")
    cik = state["cik"]
    country = state["country"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: determine_filing_type] Analyzing CIK: {cik}")

    # For now, only support US companies
    if country != "US":
        return {
            **state,
            "success": False,
            "error_message": "Only US companies are currently supported for financial info extraction",
            "score": 0
        }

    # For US companies, we'll look for 10-Q (quarterly reports)
    filing_type = "10-Q"

    if ctx:
        ctx.logger.info(f"✓ Company is US-based - will look for {filing_type} filings")

    return {
        **state,
        "filing_type": filing_type,
        "score": 100
    }


def find_latest_10q_filing_node(state: FinanceInfoState) -> FinanceInfoState:
    """
    Node 2: Find the latest 10-Q filing for the company using Selenium
    """
    ctx = state.get("ctx")
    cik = state["cik"]
    filing_type = state["filing_type"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: find_latest_10q] Looking for latest {filing_type} filing")

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
        try:
            documents_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "documentsbutton"))
            )
            filing_url = documents_button.get_attribute("href")

            # Also try to get the filing date
            filing_date = ""
            try:
                # Find the filing date in the table
                date_element = driver.find_element(By.XPATH, "//td[contains(@class, 'small')][1]")
                filing_date = date_element.text.strip()
            except:
                pass

            if ctx:
                ctx.logger.info(f"✓ Found latest {filing_type} filing: {filing_url}")
                if filing_date:
                    ctx.logger.info(f"  Filing date: {filing_date}")

            driver.quit()

            return {
                **state,
                "latest_filing_url": filing_url,
                "filing_date": filing_date,
                "score": 100
            }

        except Exception as e:
            driver.quit()
            if ctx:
                ctx.logger.error(f"Could not find {filing_type} filing link: {e}")
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


def extract_financial_info_node(state: FinanceInfoState) -> FinanceInfoState:
    """
    Node 3: Extract financial information from the 10-Q filing
    Uses keywords to find relevant sections
    """
    ctx = state.get("ctx")
    filing_url = state.get("latest_filing_url")
    filing_type = state["filing_type"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: extract_financial_info] Extracting financial data from {filing_type}")

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

        # Look for the main 10-Q document
        try:
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

            # Fallback: get the first .htm document
            if not main_doc_url:
                for row in rows[1:]:
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

                # Extract financial information using keywords
                overview = ""
                financial_condition = ""
                revenue_info = ""
                legal_proceedings = ""

                # 1. Extract Overview section
                overview_patterns = [
                    r"Overview[:\.]?\s*(.{200,1500})",
                    r"Our Business[:\.]?\s*(.{200,1500})",
                    r"Company Overview[:\.]?\s*(.{200,1500})",
                ]

                for pattern in overview_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        overview = match.group(1).strip()[:1000]
                        break

                # 2. Extract Financial Condition and Results of Operations
                financial_patterns = [
                    r"Management[''']s Discussion and Analysis of Financial Condition and Results of Operations[:\.]?\s*(.{200,1500})",
                    r"Financial Condition and Results of Operations[:\.]?\s*(.{200,1500})",
                    r"Results of Operations[:\.]?\s*(.{200,1000})",
                ]

                for pattern in financial_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        financial_condition = match.group(1).strip()[:1500]
                        break

                # 3. Extract Revenue Information
                revenue_patterns = [
                    r"Revenue[s]?[:\.]?\s*(.{100,800})",
                    r"Net Revenue[s]?[:\.]?\s*(.{100,800})",
                    r"Total Revenue[s]?[:\.]?\s*(.{100,800})",
                ]

                for pattern in revenue_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        revenue_info = match.group(1).strip()[:800]
                        break

                # 4. Extract Legal Proceedings
                legal_patterns = [
                    r"Legal Proceedings[:\.]?\s*(.{100,1000})",
                    r"Litigation[:\.]?\s*(.{100,1000})",
                    r"Governmental and Regulatory Inquiries[:\.]?\s*(.{100,1000})",
                ]

                for pattern in legal_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        legal_proceedings = match.group(1).strip()[:1000]
                        break

                driver.quit()

                # Log what we extracted
                if ctx:
                    ctx.logger.info(f"✓ Extracted financial information:")
                    ctx.logger.info(f"  Overview: {len(overview)} chars")
                    ctx.logger.info(f"  Financial Condition: {len(financial_condition)} chars")
                    ctx.logger.info(f"  Revenue Info: {len(revenue_info)} chars")
                    ctx.logger.info(f"  Legal Proceedings: {len(legal_proceedings)} chars")

                return {
                    **state,
                    "overview": overview,
                    "financial_condition": financial_condition,
                    "revenue_info": revenue_info,
                    "legal_proceedings": legal_proceedings,
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
                ctx.logger.error(f"Error extracting financial info: {e}")
            return {
                **state,
                "success": False,
                "error_message": f"Error extracting financial info: {str(e)}",
                "score": 0
            }

    except Exception as e:
        if ctx:
            ctx.logger.error(f"Error in extract_financial_info: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error extracting financial info: {str(e)}",
            "score": 0
        }


def format_results_node(state: FinanceInfoState) -> FinanceInfoState:
    """
    Node 4: Format the extracted financial information
    """
    ctx = state.get("ctx")

    if ctx:
        ctx.logger.info(f"[LangGraph Node: format_results] Formatting financial info results")

    if state.get("overview") or state.get("financial_condition"):
        if ctx:
            ctx.logger.info(f"✓ Successfully extracted financial information")
            ctx.logger.info(f"  Company: {state['company_name']}")
            ctx.logger.info(f"  Filing Type: {state['filing_type']}")

    return {
        **state,
        "success": True
    }


def error_node(state: FinanceInfoState) -> FinanceInfoState:
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

def should_continue_after_filing_type(state: FinanceInfoState) -> Literal["find_filing", "error"]:
    """Route after determining filing type"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("filing_type"):
        return "find_filing"

    return "error"


def should_continue_after_finding_filing(state: FinanceInfoState) -> Literal["extract_info", "error"]:
    """Route after finding latest filing"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("latest_filing_url"):
        return "extract_info"

    return "error"


def should_continue_to_format(state: FinanceInfoState) -> Literal["format_results", "error"]:
    """Route after extracting financial info"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("overview") or state.get("financial_condition"):
        return "format_results"

    return "error"


# ========== Graph Construction ==========

def build_finance_info_graph():
    """Build the LangGraph workflow for financial info extraction"""

    workflow = StateGraph(FinanceInfoState)

    # Add nodes
    workflow.add_node("determine_filing_type", determine_filing_type_node)
    workflow.add_node("find_filing", find_latest_10q_filing_node)
    workflow.add_node("extract_info", extract_financial_info_node)
    workflow.add_node("format_results", format_results_node)
    workflow.add_node("error", error_node)

    # Add edges
    workflow.add_edge(START, "determine_filing_type")

    # Conditional routing after determining filing type
    workflow.add_conditional_edges(
        "determine_filing_type",
        should_continue_after_filing_type,
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
finance_info_graph = build_finance_info_graph()
