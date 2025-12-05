"""
Test file for Demand Agent - Tests CSV loading and analysis
Run: python -m tests.test_demand_agent
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.monitoring.demand_agent import (
    load_demand_csv_data,
    analyze_demand_data,
    DEMAND_CSV_PATH,
)
from prompts.demand_prompt import DEMAND_FORECASE_PROMPT


def test_demand_agent():
    print("=" * 70)
    print("🧪 TESTING DEMAND AGENT")
    print("=" * 70)
    print()

    # Display the agent's prompt/role
    print("📜 AGENT PROMPT/ROLE:")
    print("-" * 70)
    print(DEMAND_FORECASE_PROMPT.strip())
    print("-" * 70)
    print()

    # Test with "coffee" supplier - maps to "food" category
    product_category = "food"
    print(f"📋 Testing with product category: {product_category}")
    print(f"   (Coffee supplier maps to 'food' category)")
    print()

    # Step 1: Verify CSV file exists
    print(f"📁 CSV Path: {DEMAND_CSV_PATH}")
    if os.path.exists(DEMAND_CSV_PATH):
        print("   ✅ CSV file exists")
    else:
        print("   ❌ CSV file NOT found!")
        return False
    print()

    # Step 2: Load and filter CSV data
    print("📊 Loading CSV data...")
    data = load_demand_csv_data(product_category)
    print(f"   ✅ Found {len(data)} {product_category} products")
    print()

    if not data:
        print("   ❌ No data found for this category!")
        return False

    # Step 3: Show sample data
    print("📝 Sample data (first 3 rows):")
    for i, row in enumerate(data[:3]):
        print(
            f"   Row {i+1}: SKU={row.get('SKU')}, Price=${float(row.get('Price', 0)):.2f}, "
            f"Availability={row.get('Availability')}%, Sold={row.get('Number of products sold')}"
        )
    print()

    # Step 4: Run analysis
    print("🔍 Analyzing demand data...")
    insights = analyze_demand_data(data, product_category)
    print()

    # Step 5: Display results
    print("=" * 70)
    print("📊 DEMAND ANALYSIS RESULTS")
    print("=" * 70)
    print()

    print("⏱️  DELAY STATUS:")
    print(f"   Status: {insights['delay_status']}")
    print(f"   Details: {insights['delay_details']}")
    print()

    print("🚢 SHIPPING STATUS:")
    print(f"   Status: {insights['shipping_status']}")
    print(f"   Details: {insights['shipping_details']}")
    print()

    print("✨ QUALITY STATUS:")
    print(f"   Status: {insights['quality_status']}")
    print(f"   Details: {insights['quality_details']}")
    print()

    print("⚠️  ALERTS:")
    if insights["alerts"]:
        for alert in insights["alerts"]:
            print(f"   • {alert}")
    else:
        print("   • No alerts")
    print()

    print(f"📈 OVERALL PERFORMANCE: {insights['overall_performance']}")
    print()
    print("=" * 70)
    print("✅ DEMAND AGENT TEST COMPLETE")
    print("=" * 70)

    return True


def test_all_categories():
    """Test all product categories"""
    print("\n" + "=" * 70)
    print("🧪 TESTING ALL PRODUCT CATEGORIES")
    print("=" * 70 + "\n")

    categories = ["food", "clothing", "electronics"]

    for category in categories:
        print(f"\n--- Testing {category.upper()} ---")
        data = load_demand_csv_data(category)
        print(f"Found {len(data)} products")

        if data:
            insights = analyze_demand_data(data, category)
            print(f"Delay Status: {insights['delay_status']}")
            print(f"Shipping Status: {insights['shipping_status']}")
            print(f"Quality Status: {insights['quality_status']}")
            print(f"Overall Performance: {insights['overall_performance']}")
            print(f"Alerts: {len(insights['alerts'])}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 DEMAND AGENT CSV TEST")
    print("=" * 70 + "\n")

    # Run main test
    success = test_demand_agent()

    # Also test all categories
    test_all_categories()

    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ TESTS FAILED")
    print("=" * 70 + "\n")
