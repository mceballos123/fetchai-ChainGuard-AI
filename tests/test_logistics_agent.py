"""
Test file for Logistics Agent - Uses Gemini to analyze CSV data with the prompt
Run: python -m tests.test_logistics_agent
"""

import os
import sys
import csv
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai
from prompts.logistics_prompt import LOGISTICS_PROMPT
from dotenv import load_dotenv

load_dotenv()

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Path to logistics monitoring CSV data
LOGISTICS_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "../data_for_monitor/logistics_monitoring_data.csv"
)


def load_logistics_csv_data(product_category: str) -> List[Dict[str, Any]]:
    """
    Load logistics monitoring data from CSV and filter by product category.
    """
    filtered_data = []

    try:
        with open(LOGISTICS_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Product type", "").lower() == product_category.lower():
                    filtered_data.append(row)
    except Exception as e:
        print(f"Error loading logistics CSV: {e}")

    return filtered_data


def format_csv_data_for_llm(data: List[Dict[str, Any]], max_rows: int = 15) -> str:
    """
    Format CSV data into a readable string for the LLM to analyze.
    """
    if not data:
        return "No data available"

    # Get column headers
    headers = list(data[0].keys())

    # Build formatted string
    formatted = "CSV Data Summary:\n"
    formatted += "-" * 80 + "\n"
    formatted += f"Total rows: {len(data)}\n"
    formatted += f"Columns: {', '.join(headers)}\n"
    formatted += "-" * 80 + "\n\n"

    # Add sample rows
    formatted += f"Sample Data (first {min(max_rows, len(data))} rows):\n\n"

    for i, row in enumerate(data[:max_rows]):
        formatted += f"Row {i+1}:\n"
        for key, value in row.items():
            formatted += f"  {key}: {value}\n"
        formatted += "\n"

    # Add statistics
    formatted += "\nData Statistics:\n"
    formatted += "-" * 40 + "\n"

    try:
        lead_times = [int(row.get("Lead times", 0)) for row in data]
        shipping_times = [int(row.get("Shipping times", 0)) for row in data]
        shipping_costs = [float(row.get("Shipping costs", 0)) for row in data]
        costs = [float(row.get("Costs", 0)) for row in data]

        formatted += f"Average Lead Time: {sum(lead_times)/len(lead_times):.1f} days\n"
        formatted += f"Average Shipping Time: {sum(shipping_times)/len(shipping_times):.1f} days\n"
        formatted += (
            f"Average Shipping Cost: ${sum(shipping_costs)/len(shipping_costs):.2f}\n"
        )
        formatted += f"Average Total Cost: ${sum(costs)/len(costs):.2f}\n"
        formatted += f"Long Lead Time (>20 days): {len([lt for lt in lead_times if lt > 20])} items\n"
        formatted += f"Short Lead Time (<7 days): {len([lt for lt in lead_times if lt < 7])} items\n"

        # Count carriers
        carriers = {}
        for row in data:
            carrier = row.get("Shipping carriers", "Unknown")
            carriers[carrier] = carriers.get(carrier, 0) + 1
        formatted += f"Carriers: {carriers}\n"

        # Count transport modes
        modes = {}
        for row in data:
            mode = row.get("Transportation modes", "Unknown")
            modes[mode] = modes.get(mode, 0) + 1
        formatted += f"Transport Modes: {modes}\n"

        # Count locations
        locations = {}
        for row in data:
            loc = row.get("Location", "Unknown")
            locations[loc] = locations.get(loc, 0) + 1
        formatted += f"Supplier Locations: {locations}\n"

    except Exception as e:
        formatted += f"Could not calculate statistics: {e}\n"

    return formatted


def analyze_with_gemini(
    data: List[Dict[str, Any]], product_category: str, supplier_name: str
) -> str:
    """
    Use Gemini to analyze the logistics data using the LOGISTICS_PROMPT.
    """
    print("Connecting to Gemini...")

    # Format the data for the LLM
    data_summary = format_csv_data_for_llm(data)

    # Build the full prompt
    full_prompt = f"""
{LOGISTICS_PROMPT}

---

SUPPLIER: {supplier_name}
PRODUCT CATEGORY: {product_category.upper()}

{data_summary}

---

Based on the above data and your role as a LOGISTICS MONITORING AGENT, provide your analysis in the following format:

CURRENT INVENTORY LEVEL: [LOW / MEDIUM / HIGH / CRITICAL]
CURRENT INVENTORY DETAILS: [Your analysis based on lead times and product availability]

PREDICTED INVENTORY LEVEL: [LOW / MEDIUM / HIGH / STABLE]
INVENTORY FORECAST: [Your forecast based on shipping patterns and lead times]

RESTOCKING STATUS: [ON_TIME / MONITOR / DELAYED]
RESTOCKING DETAILS: [Your analysis of supplier lead times and shipping schedules]

OVERALL LOGISTICS STATUS: [HEALTHY / WARNING / CRITICAL]

ALERTS:
- [List any actionable alerts for inventory management]

REASONING:
[Explain your analysis and reasoning based on the data]
"""

    print(f"Sending data to Gemini...")
    print("This may take a moment...\n")

    try:
        response = gemini_model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return f"Error calling Gemini: {e}"


def test_logistics_agent():
    print("=" * 80)
    print("                    LOGISTICS AGENT TEST (GEMINI-POWERED)")
    print("=" * 80)
    print()

    # Display the agent's prompt/role
    print("AGENT PROMPT/ROLE:")
    print("-" * 80)
    print(LOGISTICS_PROMPT.strip())
    print("-" * 80)
    print()

    # Test with "coffee" supplier - maps to "food" category
    product_category = "food"
    supplier_name = "Coffee Supplier Co."
    print(f"Testing with product category: {product_category}")
    print(f"Supplier: {supplier_name}")
    print()

    # Step 1: Verify CSV file exists
    print(f"CSV Path: {LOGISTICS_CSV_PATH}")
    if os.path.exists(LOGISTICS_CSV_PATH):
        print("   CSV file exists")
    else:
        print("   CSV file NOT found!")
        return False
    print()

    # Step 2: Load and filter CSV data
    print("Loading CSV data...")
    data = load_logistics_csv_data(product_category)
    print(f"   Found {len(data)} {product_category} products")
    print()

    if not data:
        print("   No data found for this category!")
        return False

    # Step 3: Show sample data from CSV
    print("Sample CSV data (first 3 rows):")
    print("-" * 80)
    for i, row in enumerate(data[:3]):
        print(f"Row {i+1}:")
        print(f"  SKU: {row.get('SKU')}")
        print(f"  Lead Time: {row.get('Lead times')} days")
        print(f"  Shipping Time: {row.get('Shipping times')} days")
        print(f"  Carrier: {row.get('Shipping carriers')}")
        print(f"  Shipping Cost: ${float(row.get('Shipping costs', 0)):.2f}")
        print(f"  Transport Mode: {row.get('Transportation modes')}")
        print(f"  Route: {row.get('Routes')}")
        print(f"  Supplier: {row.get('Supplier name')}")
        print(f"  Location: {row.get('Location')}")
        print()
    print("-" * 80)

    # Step 4: Analyze with Gemini
    print("\n" + "=" * 80)
    print("                    GEMINI ANALYSIS")
    print("=" * 80)

    llm_response = analyze_with_gemini(data, product_category, supplier_name)

    # Step 5: Display the LLM's analysis
    print("\n" + "=" * 80)
    print("                    LOGISTICS MONITORING AGENT REPORT")
    print("=" * 80)
    print(f"\nSupplier: {supplier_name}")
    print(f"Product Category: {product_category.upper()}")
    print(f"Data Points Analyzed: {len(data)}")
    print("-" * 80)
    print("\nLLM ANALYSIS:\n")
    print(llm_response)
    print("\n" + "=" * 80)
    print("                    END OF REPORT")
    print("=" * 80)

    return True


def test_all_categories():
    """Test all product categories with Gemini"""
    print("\n" + "=" * 80)
    print("                    TESTING ALL PRODUCT CATEGORIES")
    print("=" * 80 + "\n")

    categories = ["food", "clothing", "electronics"]
    supplier_names = {
        "food": "Fresh Foods Inc.",
        "clothing": "Fashion Forward Ltd.",
        "electronics": "TechGear Solutions",
    }

    for category in categories:
        print(f"\n{'='*80}")
        print(f"Testing {category.upper()}")
        print("=" * 80)

        data = load_logistics_csv_data(category)
        print(f"Found {len(data)} products")

        if data:
            llm_response = analyze_with_gemini(data, category, supplier_names[category])
            print("\nLLM Analysis:")
            print("-" * 40)
            print(llm_response)
            print("-" * 40)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("                    LOGISTICS AGENT CSV TEST (GEMINI)")
    print("=" * 80 + "\n")

    # Run main test
    success = test_logistics_agent()

    # Optionally test all categories (uncomment to run)
    # test_all_categories()

    print("\n" + "=" * 80)
    if success:
        print("TEST COMPLETED SUCCESSFULLY")
    else:
        print("TEST FAILED")
    print("=" * 80 + "\n")
