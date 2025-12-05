"""
Test file for Logistics Agent - Tests CSV loading and analysis
Run: python -m tests.test_logistics_agent
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.monitoring.logistics_agent import (
    load_logistics_csv_data,
    analyze_logistics_data,
    LOGISTICS_CSV_PATH,
)
from prompts.logistics_prompt import LOGISTICS_PROMPT


def test_logistics_agent():
    print("=" * 70)
    print("🧪 TESTING LOGISTICS AGENT")
    print("=" * 70)
    print()

    # Display the agent's prompt/role
    print("📜 AGENT PROMPT/ROLE:")
    print("-" * 70)
    print(LOGISTICS_PROMPT.strip())
    print("-" * 70)
    print()

    # Test with "coffee" supplier - maps to "food" category
    product_category = "food"
    print(f"📋 Testing with product category: {product_category}")
    print(f"   (Coffee supplier maps to 'food' category)")
    print()

    # Step 1: Verify CSV file exists
    print(f"📁 CSV Path: {LOGISTICS_CSV_PATH}")
    if os.path.exists(LOGISTICS_CSV_PATH):
        print("   ✅ CSV file exists")
    else:
        print("   ❌ CSV file NOT found!")
        return False
    print()

    # Step 2: Load and filter CSV data
    print("📊 Loading CSV data...")
    data = load_logistics_csv_data(product_category)
    print(f"   ✅ Found {len(data)} {product_category} products")
    print()

    if not data:
        print("   ❌ No data found for this category!")
        return False

    # Step 3: Show sample data
    print("📝 Sample data (first 3 rows):")
    for i, row in enumerate(data[:3]):
        print(
            f"   Row {i+1}: SKU={row.get('SKU')}, Lead Time={row.get('Lead times')} days, "
            f"Carrier={row.get('Shipping carriers')}, Location={row.get('Location')}"
        )
    print()

    # Step 4: Run analysis
    print("🔍 Analyzing logistics data...")
    insights = analyze_logistics_data(data, product_category)
    print()

    # Step 5: Display results
    print("=" * 70)
    print("📊 LOGISTICS ANALYSIS RESULTS")
    print("=" * 70)
    print()

    print("📦 CURRENT INVENTORY:")
    print(f"   Level: {insights['current_inventory_level']}")
    print(f"   Units: {insights['current_inventory_units']:,}")
    print(f"   Details: {insights['inventory_details']}")
    print()

    print("🔮 PREDICTED INVENTORY:")
    print(f"   Level: {insights['predicted_inventory_level']}")
    print(f"   Units: {insights['predicted_inventory_units']:,}")
    print(f"   Forecast: {insights['inventory_forecast']}")
    print()

    print("🚚 RESTOCKING STATUS:")
    print(f"   Status: {insights['restocking_status']}")
    print(f"   Details: {insights['restocking_details']}")
    print()

    print("⚠️  ALERTS:")
    if insights["alerts"]:
        for alert in insights["alerts"]:
            print(f"   • {alert}")
    else:
        print("   • No alerts")
    print()

    print(f"📈 OVERALL LOGISTICS STATUS: {insights['overall_logistics_status']}")
    print()
    print("=" * 70)
    print("✅ LOGISTICS AGENT TEST COMPLETE")
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
        data = load_logistics_csv_data(category)
        print(f"Found {len(data)} products")

        if data:
            insights = analyze_logistics_data(data, category)
            print(
                f"Current Inventory: {insights['current_inventory_level']} ({insights['current_inventory_units']:,} units)"
            )
            print(f"Predicted: {insights['predicted_inventory_level']}")
            print(f"Restocking: {insights['restocking_status']}")
            print(f"Overall Status: {insights['overall_logistics_status']}")
            print(f"Alerts: {len(insights['alerts'])}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 LOGISTICS AGENT CSV TEST")
    print("=" * 70 + "\n")

    # Run main test
    success = test_logistics_agent()

    # Also test all categories
    test_all_categories()

    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ TESTS FAILED")
    print("=" * 70 + "\n")
