"""
LangGraph workflow for extracting risk factors from SEC filings (10-Q and 20-F)
Extracts risk information for both US and foreign companies
"""

from typing import Optional, Literal, List
from langgraph.graph import StateGraph, START, END
from models.risk_info import RiskInfoState
import requests
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time


def determine_company_type_node(state: RiskInfoState) -> RiskInfoState:
    """
    Node 1: Determine if company is foreign or American based on SEC data
    """
    ctx = state.get("ctx")
    cik = state["cik"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: determine_company_type] Analyzing CIK: {cik}")

    # Add 5-second delay to avoid rate limiting
    if ctx:
        ctx.logger.info("Waiting 5 seconds to avoid rate limiting...")
    time.sleep(5)

    try:
        # Get company information from SEC to determine if it's foreign
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
            # Default to 10-Q for US companies if neither found
            is_foreign = False
            filing_type = "10-Q"
            if ctx:
                ctx.logger.info(f"⚠ Could not determine filing type, defaulting to 10-Q")

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


def find_latest_filing_node(state: RiskInfoState) -> RiskInfoState:
    """
    Node 2: Find the latest filing URL for the company
    """
    ctx = state.get("ctx")
    cik = state["cik"]
    filing_type = state["filing_type"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: find_latest_filing] Looking for latest {filing_type} filing")

    # Add 5-second delay to avoid rate limiting
    if ctx:
        ctx.logger.info("Waiting 5 seconds to avoid rate limiting...")
    time.sleep(5)

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

            # Try to get the filing date
            filing_date = ""
            try:
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


def extract_risk_factors_node(state: RiskInfoState) -> RiskInfoState:
    """
    Node 3: Extract risk factors from the filing
    """
    ctx = state.get("ctx")
    filing_url = state.get("latest_filing_url")
    filing_type = state["filing_type"]

    if ctx:
        ctx.logger.info(f"[LangGraph Node: extract_risk_factors] Extracting risk data from filing")

    if not filing_url:
        return {
            **state,
            "success": False,
            "error_message": "No filing URL available",
            "score": 0
        }

    # Add 5-second delay to avoid rate limiting
    if ctx:
        ctx.logger.info("Waiting 5 seconds to avoid rate limiting...")
    time.sleep(5)

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

        # Look for the main filing document
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

                # Extract risk factors using keywords
                summary_risk_factors = ""
                operational_risks = ""
                financial_risks = ""
                legal_regulatory_risks = ""
                all_risk_factors = []

                # 1. Extract Summary Risk Factors (common in 10-Q)
                summary_patterns = [
                    r"SUMMARY RISK FACTORS[:\.]?\s*(.{200,2500})",
                    r"Summary of Risk Factors[:\.]?\s*(.{200,2500})",
                ]

                for pattern in summary_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        summary_risk_factors = match.group(1).strip()[:2500]
                        break

                # 2. Extract general Risk Factors section
                risk_factors_patterns = [
                    r"Risk Factors[:\.]?\s*(.{200,3000})",
                    r"Item 1A\.\s*Risk Factors[:\.]?\s*(.{200,3000})",
                    r"Risks Relating to[:\.]?\s*(.{200,2000})",
                ]

                for pattern in risk_factors_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        risk_section = match.group(1).strip()[:3000]

                        # If we didn't find summary, use this as summary
                        if not summary_risk_factors:
                            summary_risk_factors = risk_section
                        break

                # 3. Extract individual risk factors (bullet points or sections)
                # Look for lines starting with bullet points or risk indicators
                bullet_pattern = r"[•·\-\*]\s*(.{50,500})"
                bullets = re.findall(bullet_pattern, page_text[:10000], re.MULTILINE)

                if bullets:
                    all_risk_factors = [bullet.strip() for bullet in bullets[:15]]  # Limit to 15 risk factors

                # 4. Extract specific risk categories if available

                # Operational risks
                operational_patterns = [
                    r"(?:business|operational|operation)\s+risk[s]?[:\.]?\s*(.{100,1000})",
                    r"risks?\s+(?:relating to|related to)\s+(?:our|the)\s+business[:\.]?\s*(.{100,1000})",
                ]

                for pattern in operational_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        operational_risks = match.group(1).strip()[:1000]
                        break

                # Financial risks
                financial_patterns = [
                    r"financial\s+risk[s]?[:\.]?\s*(.{100,1000})",
                    r"market\s+risk[s]?[:\.]?\s*(.{100,1000})",
                ]

                for pattern in financial_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        financial_risks = match.group(1).strip()[:1000]
                        break

                # Legal and regulatory risks
                legal_patterns = [
                    r"legal\s+(?:and\s+)?regulatory\s+risk[s]?[:\.]?\s*(.{100,1000})",
                    r"litigation\s+risk[s]?[:\.]?\s*(.{100,1000})",
                    r"compliance\s+risk[s]?[:\.]?\s*(.{100,1000})",
                ]

                for pattern in legal_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        legal_regulatory_risks = match.group(1).strip()[:1000]
                        break

                driver.quit()

                # Log what we extracted
                if ctx:
                    ctx.logger.info(f"✓ Extracted risk information:")
                    ctx.logger.info(f"  Summary Risk Factors: {len(summary_risk_factors)} chars")
                    ctx.logger.info(f"  Operational Risks: {len(operational_risks)} chars")
                    ctx.logger.info(f"  Financial Risks: {len(financial_risks)} chars")
                    ctx.logger.info(f"  Legal/Regulatory Risks: {len(legal_regulatory_risks)} chars")
                    ctx.logger.info(f"  Individual Risk Factors: {len(all_risk_factors)} items")

                return {
                    **state,
                    "summary_risk_factors": summary_risk_factors,
                    "operational_risks": operational_risks,
                    "financial_risks": financial_risks,
                    "legal_regulatory_risks": legal_regulatory_risks,
                    "all_risk_factors": all_risk_factors,
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
                ctx.logger.error(f"Error extracting risk factors: {e}")
            return {
                **state,
                "success": False,
                "error_message": f"Error extracting risk factors: {str(e)}",
                "score": 0
            }

    except Exception as e:
        if ctx:
            ctx.logger.error(f"Error in extract_risk_factors: {e}")
        return {
            **state,
            "success": False,
            "error_message": f"Error extracting risk factors: {str(e)}",
            "score": 0
        }


def format_results_node(state: RiskInfoState) -> RiskInfoState:
    """
    Node 4: Format the extracted risk information
    """
    ctx = state.get("ctx")

    if ctx:
        ctx.logger.info(f"[LangGraph Node: format_results] Formatting risk info results")

    if state.get("summary_risk_factors") or state.get("all_risk_factors"):
        if ctx:
            ctx.logger.info(f"✓ Successfully extracted risk information")
            ctx.logger.info(f"  Company: {state['company_name']}")
            ctx.logger.info(f"  Type: {'Foreign' if state['is_foreign'] else 'American'}")
            ctx.logger.info(f"  Filing Type: {state['filing_type']}")

    return {
        **state,
        "success": True
    }


def error_node(state: RiskInfoState) -> RiskInfoState:
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

def should_continue_after_type_check(state: RiskInfoState) -> Literal["find_filing", "error"]:
    """Route after determining company type"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("filing_type"):
        return "find_filing"

    return "error"


def should_continue_after_finding_filing(state: RiskInfoState) -> Literal["extract_info", "error"]:
    """Route after finding latest filing"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("latest_filing_url"):
        return "extract_info"

    return "error"


def should_continue_to_format(state: RiskInfoState) -> Literal["format_results", "error"]:
    """Route after extracting risk info"""
    if state.get("error_message") or state.get("score", 0) < 75:
        return "error"

    if state.get("summary_risk_factors") or state.get("all_risk_factors"):
        return "format_results"

    return "error"


# ========== Graph Construction ==========

def build_risk_info_graph():
    """Build the LangGraph workflow for risk info extraction"""

    workflow = StateGraph(RiskInfoState)

    # Add nodes
    workflow.add_node("determine_type", determine_company_type_node)
    workflow.add_node("find_filing", find_latest_filing_node)
    workflow.add_node("extract_info", extract_risk_factors_node)
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
risk_info_graph = build_risk_info_graph()
