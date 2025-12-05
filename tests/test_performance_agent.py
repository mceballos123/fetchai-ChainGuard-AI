"""
Test file for Performance Agent - Tests CSV loading and analysis
Run: python -m tests.test_performance_agent
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.monitoring.performance_agent import (
    load_performance_csv_data,
    analyze_performance_data,
    PERFORMANCE_CSV_PATH,
)
from prompts.performance_prompt import PROMPT as PERFORMANCE_PROMPT


def test_performance_agent():
    print("=" * 70)
    print("🧪 TESTING PERFORMANCE AGENT")
    print("=" * 70)
    print()

    # Display the agent's prompt/role
    print("📜 AGENT PROMPT/ROLE:")
    print("-" * 70)
    print(PERFORMANCE_PROMPT.strip())
    print("-" * 70)
    print()

    # Test with "coffee" supplier - maps to "food" category
    product_category = "food"
    print(f"📋 Testing with product category: {product_category}")
    print(f"   (Coffee supplier maps to 'food' category)")
    print()

    # Step 1: Verify CSV file exists
    print(f"📁 CSV Path: {PERFORMANCE_CSV_PATH}")
    if os.path.exists(PERFORMANCE_CSV_PATH):
        print("   ✅ CSV file exists")
    else:
        print("   ❌ CSV file NOT found!")
        return False
    print()

    # Step 2: Load and filter CSV data
    print("📊 Loading CSV data...")
    data = load_performance_csv_data(product_category)
    print(f"   ✅ Found {len(data)} {product_category} products")
    print()

    if not data:
        print("   ❌ No data found for this category!")
        return False

    # Step 3: Show sample data
    print("📝 Sample data (first 3 rows):")
    for i, row in enumerate(data[:3]):
        print(
            f"   Row {i+1}: SKU={row.get('SKU')}, Stock={row.get('Stock levels')}, "
            f"Defect Rate={row.get('Defect rates')}, Inspection={row.get('Inspection results')}"
        )
    print()

    # Step 4: Run analysis
    print("🔍 Analyzing performance data...")
    insights = analyze_performance_data(data, product_category)
    print()

    # Step 5: Display results
    print("=" * 70)
    print("📊 PERFORMANCE ANALYSIS RESULTS")
    print("=" * 70)
    print()

    print("🌤️  WEATHER STATUS:")
    print(f"   Status: {insights['weather_status']}")
    print(f"   Details: {insights['weather_details']}")
    print()

    print("👷 STRIKE STATUS:")
    print(f"   Status: {insights['strike_status']}")
    print(f"   Details: {insights['strike_details']}")
    print()

    print("🏛️  POLITICAL STATUS:")
    print(f"   Status: {insights['political_status']}")
    print(f"   Details: {insights['political_details']}")
    print()

    print("⚖️  LEGAL STATUS:")
    print(f"   Status: {insights['legal_status']}")
    print(f"   Details: {insights['legal_details']}")
    print()

    print("⚠️  ALERTS:")
    if insights["alerts"]:
        for alert in insights["alerts"]:
            print(f"   • {alert}")
    else:
        print("   • No alerts")
    print()

    print(f"📈 OVERALL RISK LEVEL: {insights['overall_risk']}")
    print()
    print("=" * 70)
    print("✅ PERFORMANCE AGENT TEST COMPLETE")
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
        data = load_performance_csv_data(category)
        print(f"Found {len(data)} products")

        if data:
            insights = analyze_performance_data(data, category)
            print(f"Overall Risk: {insights['overall_risk']}")
            print(f"Alerts: {len(insights['alerts'])}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 PERFORMANCE AGENT CSV TEST")
    print("=" * 70 + "\n")

    # Run main test
    success = test_performance_agent()

    # Also test all categories
    test_all_categories()

    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ TESTS FAILED")
    print("=" * 70 + "\n")
