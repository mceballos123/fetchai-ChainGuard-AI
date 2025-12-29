#!/usr/bin/env python3
"""
Multi-Country Supplier Finder with LangGraph
Searches for suppliers using US SEC API or UK Companies House API based on user location
"""

import os
import re
import requests
import json
from dotenv import load_dotenv
from typing import TypedDict, Literal, List, Dict, Optional
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()


# ========== State Definitions ==========

class InputState(TypedDict):
    """Input state for the graph"""
    user_message: str

class OverallState(TypedDict):
    """Overall state that tracks the entire workflow"""
    user_message: str
    country: Optional[str]
    establishment_type: Optional[str]
    search_query: Optional[str]
    suppliers: List[Dict]
    error: Optional[str]
    score: int

class OutputState(TypedDict):
    """Output state for the graph"""
    suppliers: List[Dict]
    error: Optional[str]
    message: str


# ========== Helper Functions ==========

def detect_country_and_query(message: str) -> tuple[Optional[str], Optional[str], str]:
    """
    Detect country and establishment type from user message

    Returns:
        tuple: (country, establishment_type, search_query)
    """
    message_lower = message.lower()

    # Detect country
    country = None
    if any(word in message_lower for word in ["united states", "usa", "us", "america", "american"]):
        country = "US"
    elif any(word in message_lower for word in ["uk", "united kingdom", "britain", "british", "england"]):
        country = "UK"

    # Extract establishment type - look for common business types
    establishment_patterns = [
        r"(\w+)\s+supplier",
        r"(\w+)\s+distributor",
        r"(\w+)\s+manufacturer",
        r"(\w+)\s+wholesaler",
        r"find\s+(\w+)\s+(?:for|companies)",
    ]

    establishment_type = None
    for pattern in establishment_patterns:
        match = re.search(pattern, message_lower)
        if match:
            establishment_type = match.group(1)
            break

    # If no specific type found, try to extract any business keyword
    if not establishment_type:
        business_keywords = ["coffee", "restaurant", "automobile", "food", "retail", "tech", "software"]
        for keyword in business_keywords:
            if keyword in message_lower:
                establishment_type = keyword
                break

    # Build search query
    if establishment_type:
        search_query = f"{establishment_type} supplier"
    else:
        search_query = "supplier"

    return country, establishment_type, search_query


# ========== Node Functions ==========

def parse_user_input(state: InputState) -> OverallState:
    """
    Parse user input to extract country and establishment type
    """
    message = state["user_message"]
    country, establishment_type, search_query = detect_country_and_query(message)

    return {
        "user_message": message,
        "country": country,
        "establishment_type": establishment_type,
        "search_query": search_query,
        "suppliers": [],
        "error": None,
        "score": 100  # Start with perfect score
    }


def search_uk_suppliers(state: OverallState) -> OverallState:
    """
    Search for suppliers using UK Companies House API
    """
    api_key = os.getenv('GOV_UK_API_KEY', '').strip()

    if not api_key or api_key == "your_api_key_here":
        return {
            **state,
            "error": "GOV_UK_API_KEY not configured in .env file",
            "score": 0
        }

    base_url = "https://api.company-information.service.gov.uk/search/companies"
    params = {
        "q": state["search_query"],
        "items_per_page": 4  # Top 4 suppliers
    }

    try:
        response = requests.get(
            base_url,
            params=params,
            auth=(api_key, ''),
            headers={'Accept': 'application/json'},
            timeout=10
        )

        if response.status_code == 401:
            return {
                **state,
                "error": "Invalid UK API key. Please get a valid key from https://developer.company-information.service.gov.uk/",
                "score": 0
            }

        response.raise_for_status()
        data = response.json()
        companies = data.get("items", [])

        suppliers = []
        for company in companies[:4]:
            address = company.get("address", {})
            address_parts = [
                address.get("address_line_1", ""),
                address.get("address_line_2", ""),
                address.get("locality", ""),
                address.get("postal_code", "")
            ]
            formatted_address = ", ".join(filter(None, address_parts))

            suppliers.append({
                "name": company.get("title"),
                "company_number": company.get("company_number"),
                "status": company.get("company_status"),
                "type": company.get("company_type"),
                "address": formatted_address,
                "country": "UK"
            })

        score = 100 if len(suppliers) >= 4 else 75

        return {
            **state,
            "suppliers": suppliers,
            "score": score
        }

    except Exception as e:
        return {
            **state,
            "error": f"Error searching UK suppliers: {str(e)}",
            "score": 0
        }


def search_us_suppliers(state: OverallState) -> OverallState:
    """
    Search for suppliers using US SEC API
    """
    # SEC API doesn't require authentication but needs User-Agent header
    # We'll search the company tickers file

    try:
        # First, get the company tickers JSON
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        response = requests.get(tickers_url, headers=headers, timeout=10)
        response.raise_for_status()

        companies_data = response.json()

        # Search for companies matching the query
        search_term = state.get("establishment_type", "").lower()
        if not search_term:
            search_term = "supplier"

        matching_companies = []
        for _, company in companies_data.items():
            company_name = company.get("title", "").lower()
            if search_term in company_name or "supplier" in company_name:
                matching_companies.append({
                    "name": company.get("title"),
                    "ticker": company.get("ticker"),
                    "cik": str(company.get("cik_str")).zfill(10),
                    "country": "US"
                })

                if len(matching_companies) >= 4:
                    break

        score = 100 if len(matching_companies) >= 4 else 75

        return {
            **state,
            "suppliers": matching_companies,
            "score": score
        }

    except Exception as e:
        return {
            **state,
            "error": f"Error searching US suppliers: {str(e)}",
            "score": 0
        }


def format_output(state: OverallState) -> OutputState:
    """
    Format the final output for display
    """
    if state.get("error"):
        return {
            "suppliers": [],
            "error": state["error"],
            "message": f"Error: {state['error']}"
        }

    suppliers = state.get("suppliers", [])
    country = state.get("country", "Unknown")
    establishment = state.get("establishment_type", "supplier")

    message = f"Found {len(suppliers)} {establishment} suppliers in {country}"

    return {
        "suppliers": suppliers,
        "error": None,
        "message": message
    }


# ========== Routing Functions ==========

def route_by_country(state: OverallState) -> Literal["search_us", "search_uk", "end"]:
    """
    Route to appropriate search node based on detected country
    """
    # Check if there's already an error
    if state.get("error"):
        return "end"

    # Check score
    if state.get("score", 0) < 75:
        return "end"

    # Route based on country
    country = state.get("country")

    if country == "US":
        return "search_us"
    elif country == "UK":
        return "search_uk"
    else:
        # If no country detected, default to error
        state["error"] = "Could not detect country. Please specify 'US' or 'UK' in your message."
        state["score"] = 0
        return "end"


def should_continue_to_output(state: OverallState) -> Literal["format_output", "end"]:
    """
    Determine if we should format output or end due to errors
    """
    # If error or low score, end
    if state.get("error") or state.get("score", 0) < 75:
        return "end"

    # If we have suppliers, format output
    if state.get("suppliers"):
        return "format_output"

    return "end"


# ========== Graph Construction ==========

def create_supplier_finder_graph():
    """
    Create the LangGraph workflow for finding suppliers
    """
    # Create the graph
    builder = StateGraph(
        OverallState,
        input_schema=InputState,
        output_schema=OutputState
    )

    # Add nodes
    builder.add_node("parse_input", parse_user_input)
    builder.add_node("search_us", search_us_suppliers)
    builder.add_node("search_uk", search_uk_suppliers)
    builder.add_node("format_output", format_output)

    # Add edges
    builder.add_edge(START, "parse_input")

    # Conditional routing based on country
    builder.add_conditional_edges(
        "parse_input",
        route_by_country,
        {
            "search_us": "search_us",
            "search_uk": "search_uk",
            "end": END
        }
    )

    # After searching, check if we should continue or end
    builder.add_conditional_edges(
        "search_us",
        should_continue_to_output,
        {
            "format_output": "format_output",
            "end": END
        }
    )

    builder.add_conditional_edges(
        "search_uk",
        should_continue_to_output,
        {
            "format_output": "format_output",
            "end": END
        }
    )

    # End after formatting output
    builder.add_edge("format_output", END)

    # Compile the graph
    return builder.compile()


# ========== Display Functions ==========

def display_results(result: dict):
    """
    Display the search results to the user
    """
    print("\n" + "=" * 80)

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        print("=" * 80)
        return

    suppliers = result.get("suppliers", [])

    if not suppliers:
        print("No suppliers found.")
        print("=" * 80)
        return

    print(result.get("message", "Search Results"))
    print("=" * 80)
    print()

    for idx, supplier in enumerate(suppliers, 1):
        print(f"{idx}. {supplier.get('name', 'Unknown')}")

        if supplier.get("country") == "UK":
            print(f"   Company Number: {supplier.get('company_number', 'N/A')}")
            print(f"   Status: {supplier.get('status', 'N/A')}")
            print(f"   Type: {supplier.get('type', 'N/A')}")
            if supplier.get('address'):
                print(f"   Address: {supplier.get('address')}")
        elif supplier.get("country") == "US":
            print(f"   Ticker: {supplier.get('ticker', 'N/A')}")
            print(f"   CIK: {supplier.get('cik', 'N/A')}")

        print(f"   Country: {supplier.get('country', 'N/A')}")
        print()

    # Save to JSON
    output_file = "supplier_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Results saved to {output_file}")
    print("=" * 80)


# ========== Main Function ==========

def main():
    """
    Main function to run the supplier finder
    """
    print("=" * 80)
    print("Multi-Country Supplier Finder")
    print("Powered by LangGraph")
    print("=" * 80)
    print()
    print("Supported countries: US (SEC API) and UK (Companies House API)")
    print("Example queries:")
    print("  - 'Find coffee suppliers in the UK'")
    print("  - 'I need restaurant suppliers in the United States'")
    print("  - 'Looking for automobile suppliers in America'")
    print()

    # Get user input
    user_message = input("Enter your request: ").strip()

    if not user_message:
        print("No input provided. Exiting.")
        return

    try:
        # Create the graph
        graph = create_supplier_finder_graph()

        # Run the graph
        print("\n🔍 Searching...")
        result = graph.invoke({"user_message": user_message})

        # Display results
        display_results(result)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


# ========== Test Functions ==========

def test_us_api():
    """
    Test the US SEC API search functionality
    """
    print("\n" + "="*80)
    print("Testing US SEC API")
    print("="*80)

    test_state = {
        "user_message": "Find coffee suppliers in the US",
        "country": "US",
        "establishment_type": "coffee",
        "search_query": "coffee supplier",
        "suppliers": [],
        "error": None,
        "score": 100
    }

    print(f"Test query: {test_state['search_query']}")
    print("Searching US SEC database...")

    result = search_us_suppliers(test_state)

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return False

    suppliers = result.get("suppliers", [])
    print(f"✓ Found {len(suppliers)} US suppliers")

    for idx, supplier in enumerate(suppliers, 1):
        print(f"\n{idx}. {supplier.get('name')}")
        print(f"   Ticker: {supplier.get('ticker', 'N/A')}")
        print(f"   CIK: {supplier.get('cik', 'N/A')}")
        print(f"   Country: {supplier.get('country')}")

    print("\n" + "="*80)
    return len(suppliers) > 0


def test_uk_api():
    """
    Test the UK Companies House API search functionality
    """
    print("\n" + "="*80)
    print("Testing UK Companies House API")
    print("="*80)

    test_state = {
        "user_message": "Find coffee suppliers in the UK",
        "country": "UK",
        "establishment_type": "coffee",
        "search_query": "coffee supplier",
        "suppliers": [],
        "error": None,
        "score": 100
    }

    print(f"Test query: {test_state['search_query']}")
    print("Searching UK Companies House database...")

    result = search_uk_suppliers(test_state)

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return False

    suppliers = result.get("suppliers", [])
    print(f"✓ Found {len(suppliers)} UK suppliers")

    for idx, supplier in enumerate(suppliers, 1):
        print(f"\n{idx}. {supplier.get('name')}")
        print(f"   Company Number: {supplier.get('company_number', 'N/A')}")
        print(f"   Status: {supplier.get('status', 'N/A')}")
        print(f"   Address: {supplier.get('address', 'N/A')}")
        print(f"   Country: {supplier.get('country')}")

    print("\n" + "="*80)
    return len(suppliers) > 0


def test_country_detection():
    """
    Test the country detection logic
    """
    print("\n" + "="*80)
    print("Testing Country Detection")
    print("="*80)

    test_cases = [
        ("Find coffee suppliers in the US", "US", "coffee"),
        ("I need coffee suppliers in the UK", "UK", "coffee"),
        ("Looking for restaurant suppliers in America", "US", "restaurant"),
        ("Find automobile suppliers in Britain", "UK", "automobile"),
    ]

    for message, expected_country, expected_type in test_cases:
        country, est_type, query = detect_country_and_query(message)
        status = "✓" if country == expected_country else "❌"
        print(f"{status} '{message}'")
        print(f"   Detected: Country={country}, Type={est_type}, Query={query}")
        print(f"   Expected: Country={expected_country}, Type={expected_type}")
        print()

    print("="*80)


def run_tests():
    """
    Run all test cases
    """
    print("\n" + "="*80)
    print("Running Test Suite")
    print("="*80)

    # Test country detection
    test_country_detection()

    # Test US API
    us_success = test_us_api()

    # Test UK API
    uk_success = test_uk_api()

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"US API Test: {'✓ PASSED' if us_success else '❌ FAILED'}")
    print(f"UK API Test: {'✓ PASSED' if uk_success else '❌ FAILED'}")
    print("="*80)


if __name__ == "__main__":
    import sys

    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        main()
