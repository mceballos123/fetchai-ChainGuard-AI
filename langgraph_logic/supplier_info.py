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
                # iXBRL viewer doesn't work in headless mode - use raw HTML instead
                actual_doc_url = main_doc_url
                if "/ix?doc=" in main_doc_url:
                    # Extract the raw document URL (bypass iXBRL viewer)
                    actual_doc_url = "https://www.sec.gov" + main_doc_url.split("/ix?doc=")[1]
                    if ctx:
                        ctx.logger.info(f"📄 Using raw HTML (iXBRL viewer doesn't work in headless mode)")
                        ctx.logger.info(f"📄 Raw document URL: {actual_doc_url}")

                driver.get(actual_doc_url)
                time.sleep(5)

                # Get page content
                page_text = driver.find_element(By.TAG_NAME, "body").text

                if ctx:
                    ctx.logger.info(f"📄 Page text length: {len(page_text)} chars")

                    if len(page_text) > 0:
                        ctx.logger.info(f"📄 First 500 chars: {page_text[:500]}")
                    else:
                        ctx.logger.warning(f"⚠ Page text is empty, will try HTML parsing")

                # Extract company information from Item 4 section
                # Focus on getting concise, relevant company description (~150 words)
                company_description = ""
                history = ""

                if ctx:
                    ctx.logger.info(f"🔍 Looking specifically for 'Item 4. Information on the Company' section (NOT Key Information)...")

                # Step 1: Skip Table of Contents completely
                # The TOC appears early in the document with page numbers
                # Actual content starts much later, after all the TOC entries

                content_start = 0

                # Find the SECOND occurrence of "PART I" (first is in TOC, second is actual content)
                part_i_matches = list(re.finditer(r"PART\s+I\b", page_text, re.IGNORECASE))

                if len(part_i_matches) >= 2:
                    # Use the second "PART I" which is the actual content start
                    content_start = part_i_matches[1].start()
                    if ctx:
                        ctx.logger.info(f"📍 Found actual content start (2nd PART I occurrence) at position {content_start}")

                elif len(part_i_matches) == 1:
                    # If only one PART I, look for first actual Item with substantial text after it
                    # Pattern: "Item 1." followed by actual paragraph (not just page numbers)
                    item1_pattern = r"Item\s+1\.\s+Identity of Directors[^\d]{100,}"
                    match = re.search(item1_pattern, page_text, re.IGNORECASE)
                    if match:
                        content_start = match.start()
                        if ctx:
                            ctx.logger.info(f"📍 Found content start via Item 1 with content at position {content_start}")

                if content_start == 0:
                    # Fallback: Skip first 20% of document (usually TOC is in first portion)
                    content_start = len(page_text) // 5
                    if ctx:
                        ctx.logger.warning(f"⚠ Using fallback: skipping first 20% of document")
                        ctx.logger.info(f"📍 Content start at position {content_start}")

                # Use content after TOC
                main_content = page_text[content_start:] if content_start > 0 else page_text

                if ctx:
                    ctx.logger.info(f"📄 Main content length: {len(main_content)} chars")
                    ctx.logger.info(f"📄 First 300 chars of main content: {main_content[:300]}...")

                # Step 2: Find actual "Item 4. Information on the Company" section content
                # NOT "Item 4A" - we want the intro paragraph before subsections

                # Look for "Item 4." header followed by actual text paragraphs
                # Actual content will have sentences, not just page numbers
                item4_patterns = [
                    # Pattern 1: Item 4 followed by text until Item 4.A or Item 4A
                    r"Item\s+4\.\s+Information on the Company\s*\n+(.+?)(?=\s+(?:Item\s+4\.A|Item\s+4A|4\.A\.|ITEM\s+4A))",
                    # Pattern 2: More flexible - until any subsection
                    r"Item\s+4\.\s+Information on the Company\s*\n+(.+?)(?=\s+Item\s+\d+\.?[A-Z])",
                    # Pattern 3: Get first paragraph after Item 4
                    r"Item\s+4\.\s+Information on the Company\s*\n+([A-Z][^\n]+(?:\n[A-Z][^\n]+){1,3})",
                ]

                if ctx:
                    ctx.logger.info(f"  Trying {len(item4_patterns)} patterns to find Item 4 actual content...")

                for i, pattern in enumerate(item4_patterns):
                    match = re.search(pattern, main_content, re.IGNORECASE | re.DOTALL)
                    if match:
                        company_description = match.group(1).strip()

                        if ctx:
                            ctx.logger.info(f"📝 Pattern {i+1} matched! Checking if valid...")
                            ctx.logger.info(f"  First 200 chars: {company_description[:200]}...")

                        # Strong validation: reject if it looks like TOC
                        # TOC has: numbers followed by "Item"
                        if re.search(r'^\d+\s*$', company_description, re.MULTILINE):
                            if ctx:
                                ctx.logger.warning(f"⚠ Contains standalone numbers (TOC pattern), skipping...")
                            continue

                        if re.search(r'\d+\s+Item\s+\d+', company_description[:300]):
                            if ctx:
                                ctx.logger.warning(f"⚠ Contains 'number Item number' (TOC pattern), skipping...")
                            continue

                        # Must have actual text (words), not just numbers and item references
                        word_count = len(re.findall(r'\b[A-Za-z]{3,}\b', company_description[:500]))
                        if word_count < 20:
                            if ctx:
                                ctx.logger.warning(f"⚠ Too few words ({word_count}), likely not actual content...")
                            continue

                        if ctx:
                            ctx.logger.info(f"✓ VALID content found with pattern {i+1}!")

                        # Clean up: remove excessive whitespace
                        company_description = re.sub(r'\s+', ' ', company_description)
                        # Limit to approximately 150 words
                        words = company_description.split()
                        company_description = ' '.join(words[:150])

                        if ctx:
                            ctx.logger.info(f"  Final: {len(company_description)} chars (~{len(words[:150])} words)")
                            ctx.logger.info(f"  Preview: {company_description[:200]}...")
                        break
                else:
                    if ctx:
                        ctx.logger.warning(f"⚠ Could not find Item 4 actual content after trying all patterns")

                # Step 2: If Item 4 not found, look for "4.A. History and Development"
                if not company_description:
                    if ctx:
                        ctx.logger.info(f"🔍 Item 4 intro not found, looking for 4.A History section...")

                    history_patterns = [
                        r"4\.A\.\s*History and Development of the Company\s+(In [A-Z][^\.]+\.[^\n]{200,1000})",
                        r"History and Development of the Company[:\.]?\s+(In [A-Z][^\.]+\.[^\n]{200,1000})",
                    ]

                    for i, pattern in enumerate(history_patterns):
                        match = re.search(pattern, main_content, re.IGNORECASE | re.DOTALL)
                        if match:
                            history = match.group(1).strip()
                            history = re.sub(r'\s+', ' ', history)
                            words = history.split()
                            history = ' '.join(words[:150])

                            if ctx:
                                ctx.logger.info(f"✓ Found History section with pattern {i+1}")
                                ctx.logger.info(f"  Length: {len(history)} chars (~{len(words[:150])} words)")

                            # Use history as description
                            company_description = history
                            break

                # Step 3: Fallback - look for company name mentions
                if not company_description:
                    if ctx:
                        ctx.logger.info(f"📝 Using fallback - searching for company description...")

                    # Look for sentences that describe what the company does
                    company_name = state.get("company_name", "")
                    if company_name:
                        # Search for the company name followed by description
                        fallback_pattern = rf"{re.escape(company_name)}[^\n]{{0,50}}(?:is|operates|provides|engages|specializes)([^\n]{{200,800}})"
                        match = re.search(fallback_pattern, main_content, re.IGNORECASE)
                        if match:
                            company_description = f"{company_name} {match.group(0)}"
                            company_description = re.sub(r'\s+', ' ', company_description)
                            words = company_description.split()
                            company_description = ' '.join(words[:150])

                            if ctx:
                                ctx.logger.info(f"✓ Found company description via fallback")
                                ctx.logger.info(f"  Length: {len(company_description)} chars (~{len(words[:150])} words)")

                driver.quit()

                if ctx:
                    ctx.logger.info(f"✓ Extracted company information:")
                    ctx.logger.info(f"  - Description: {len(company_description)} chars")
                    ctx.logger.info(f"  - History: {len(history)} chars")
                    if company_description:
                        ctx.logger.info(f"  - Description preview: {company_description[:200]}...")
                    else:
                        ctx.logger.error(f"  - ❌ DESCRIPTION IS EMPTY!")

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

    error_msg = state.get('error_message', 'Unknown error')

    if ctx:
        ctx.logger.error(f"[LangGraph Node: error] Entering error handler")
        ctx.logger.error(f"  - Error message: {error_msg}")
        ctx.logger.error(f"  - Company: {state.get('company_name')}")
        ctx.logger.error(f"  - Has description: {bool(state.get('company_description'))}")
        ctx.logger.error(f"  - Has history: {bool(state.get('history'))}")

        # If we got here without an error message but also without data, set a default error
        if not error_msg and not state.get('company_description') and not state.get('history'):
            error_msg = "Failed to extract company description or history from SEC filing"
            ctx.logger.error(f"  - Setting default error message: {error_msg}")

    return {
        **state,
        "success": False,
        "error_message": error_msg
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
    ctx = state.get("ctx")

    if ctx:
        ctx.logger.info(f"🔀 Routing decision after extraction:")
        ctx.logger.info(f"  - error_message: {state.get('error_message')}")
        ctx.logger.info(f"  - score: {state.get('score', 0)}")
        ctx.logger.info(f"  - has company_description: {bool(state.get('company_description'))}")
        ctx.logger.info(f"  - has history: {bool(state.get('history'))}")

    if state.get("error_message") or state.get("score", 0) < 75:
        if ctx:
            ctx.logger.warning(f"❌ Routing to ERROR (error_message or low score)")
        return "error"

    if state.get("company_description") or state.get("history"):
        if ctx:
            ctx.logger.info(f"✓ Routing to FORMAT_RESULTS")
        return "format_results"

    if ctx:
        ctx.logger.warning(f"❌ Routing to ERROR (no description or history)")
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
