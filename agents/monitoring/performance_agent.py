from uagents import Agent, Context, Protocol
import os
import csv
from typing import Dict, Any, List

# Import models
from models.performance import PerformanceRequest, PerformanceResponse
from prompts.performance_prompt import PROMPT
from dotenv import load_dotenv

load_dotenv()

performance_agent = Agent(
    name="performance_agent",
    seed=os.getenv("PERFORMANCE_AGENT_SEED"),
    port=8004,
    mailbox=True,
)

performance_protocol = Protocol(name="performance_protocol", version="1.0")

# Path to performance monitoring CSV data
PERFORMANCE_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "../../data_for_monitor/performance_monitoring_data.csv"
)


def load_performance_csv_data(product_category: str) -> List[Dict[str, Any]]:
    """
    Load performance monitoring data from CSV and filter by product category.

    Args:
        product_category: The product type to filter by (food, clothing, electronics)

    Returns:
        List of dictionaries containing filtered performance data
    """
    filtered_data = []

    try:
        with open(PERFORMANCE_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Product type", "").lower() == product_category.lower():
                    filtered_data.append(row)
    except Exception as e:
        print(f"Error loading performance CSV: {e}")

    return filtered_data


def analyze_performance_data(
    data: List[Dict[str, Any]], product_category: str
) -> Dict[str, Any]:
    """
    Analyze performance data for weather, strikes, politics, and legal insights.

    CSV columns: Product type, SKU, Stock levels, Lead times, Order quantities,
                 Production volumes, Manufacturing lead time, Manufacturing costs,
                 Inspection results, Defect rates

    We use this production/manufacturing data to simulate external factors:
    - Weather: Derived from production volumes and lead times
    - Strikes: Derived from manufacturing lead times and costs
    - Political: Derived from stock levels and order quantities
    - Legal: Derived from inspection results and defect rates

    Args:
        data: List of filtered performance data rows
        product_category: The product category being analyzed

    Returns:
        Dictionary with performance monitoring insights
    """
    insights = {
        "weather_status": "NORMAL",
        "weather_details": "",
        "strike_status": "NO STRIKES",
        "strike_details": "",
        "political_status": "STABLE",
        "political_details": "",
        "legal_status": "COMPLIANT",
        "legal_details": "",
        "alerts": [],
        "overall_risk": "LOW",
    }

    if not data:
        insights["weather_status"] = "UNKNOWN"
        insights["weather_details"] = (
            f"No data available for {product_category} products"
        )
        insights["overall_risk"] = "UNKNOWN"
        insights["alerts"].append(
            f"No {product_category} products found in performance data"
        )
        return insights

    # Analyze the data
    total_products = len(data)
    total_stock = 0
    total_lead_time = 0
    total_order_qty = 0
    total_production = 0
    total_mfg_lead_time = 0
    total_mfg_cost = 0
    total_defect_rate = 0

    # Track inspection results
    pass_count = 0
    fail_count = 0
    pending_count = 0

    low_stock_count = 0
    high_defect_count = 0
    long_mfg_lead_count = 0

    for row in data:
        try:
            stock_level = int(row.get("Stock levels", 0))
            lead_time = int(row.get("Lead times", 0))
            order_qty = int(row.get("Order quantities", 0))
            production = int(row.get("Production volumes", 0))
            mfg_lead_time = int(row.get("Manufacturing lead time", 0))
            mfg_cost = float(row.get("Manufacturing costs", 0))
            defect_rate = float(row.get("Defect rates", 0))
            inspection = row.get("Inspection results", "Pending")

            total_stock += stock_level
            total_lead_time += lead_time
            total_order_qty += order_qty
            total_production += production
            total_mfg_lead_time += mfg_lead_time
            total_mfg_cost += mfg_cost
            total_defect_rate += defect_rate

            # Track inspection results
            if inspection.lower() == "pass":
                pass_count += 1
            elif inspection.lower() == "fail":
                fail_count += 1
            else:
                pending_count += 1

            # Track problem areas
            if stock_level < 20:
                low_stock_count += 1
            if defect_rate > 3.0:
                high_defect_count += 1
            if mfg_lead_time > 20:
                long_mfg_lead_count += 1

        except (ValueError, TypeError):
            continue

    avg_stock = total_stock / total_products if total_products > 0 else 0
    avg_lead_time = total_lead_time / total_products if total_products > 0 else 0
    avg_production = total_production / total_products if total_products > 0 else 0
    avg_mfg_lead_time = (
        total_mfg_lead_time / total_products if total_products > 0 else 0
    )
    avg_mfg_cost = total_mfg_cost / total_products if total_products > 0 else 0
    avg_defect_rate = total_defect_rate / total_products if total_products > 0 else 0

    # WEATHER STATUS: Based on production volumes and lead times
    # (simulating weather impact on supply chain)
    if avg_production < 400:
        insights["weather_status"] = "RISK: PRODUCTION SLOWDOWN"
        insights["weather_details"] = (
            f"Low production volumes ({avg_production:.0f} avg) detected for {product_category}. "
            f"This may indicate weather-related supply chain disruptions or seasonal factors. "
            f"Monitor weather conditions in supplier regions."
        )
        insights["alerts"].append("Production slowdown - possible weather impact")
    elif avg_lead_time > 18:
        insights["weather_status"] = "MONITOR: DELAYS"
            insights["weather_details"] = (
            f"Extended lead times ({avg_lead_time:.0f} days) for {product_category} products. "
            f"Possible weather-related transportation delays. Track shipping conditions."
            )
        insights["alerts"].append("Extended lead times - monitor weather conditions")
    else:
        insights["weather_status"] = "NORMAL"
            insights["weather_details"] = (
            f"Production and logistics for {product_category} operating normally. "
            f"Avg production: {avg_production:.0f} units. Avg lead time: {avg_lead_time:.0f} days. "
            f"No weather-related disruptions detected."
            )

    # STRIKE STATUS: Based on manufacturing lead times and costs
    # (simulating labor issues)
    long_mfg_ratio = long_mfg_lead_count / total_products if total_products > 0 else 0
    if long_mfg_ratio > 0.4:
        insights["strike_status"] = "RISK: LABOR CONSTRAINTS"
        insights["strike_details"] = (
            f"{long_mfg_lead_count}/{total_products} {product_category} items have extended manufacturing times. "
            f"Avg manufacturing lead time: {avg_mfg_lead_time:.0f} days. "
            f"Possible labor shortages or workforce constraints at supplier facilities."
        )
        insights["alerts"].append(
            "Extended manufacturing times - monitor labor conditions"
        )
    elif avg_mfg_cost > 60:
        insights["strike_status"] = "MONITOR: COST PRESSURE"
        insights["strike_details"] = (
            f"High manufacturing costs (${avg_mfg_cost:.2f} avg) for {product_category}. "
            f"May indicate labor cost increases or wage negotiations. Monitor supplier labor relations."
        )
    else:
        insights["strike_status"] = "NO STRIKES"
            insights["strike_details"] = (
            f"Manufacturing operations stable for {product_category}. "
            f"Avg manufacturing lead time: {avg_mfg_lead_time:.0f} days. "
            f"Avg manufacturing cost: ${avg_mfg_cost:.2f}. No labor disruptions reported."
        )

    # POLITICAL STATUS: Based on stock levels and order quantities
    # (simulating supply stability)
    low_stock_ratio = low_stock_count / total_products if total_products > 0 else 0
    if low_stock_ratio > 0.4:
        insights["political_status"] = "UNSTABLE"
        insights["political_details"] = (
            f"Low stock levels across {low_stock_count}/{total_products} {product_category} products. "
            f"May indicate supply chain disruptions due to trade policies or regional instability. "
            f"Review supplier country situations."
        )
        insights["alerts"].append("Low stock levels - review supply chain stability")
    elif avg_stock < 40:
        insights["political_status"] = "MONITOR"
        insights["political_details"] = (
            f"Below-average stock levels ({avg_stock:.0f}) for {product_category}. "
            f"Monitor political and trade developments in supplier regions."
        )
    else:
        insights["political_status"] = "STABLE"
            insights["political_details"] = (
            f"Stock levels healthy ({avg_stock:.0f} avg) for {product_category}. "
            f"Avg order quantity: {total_order_qty / total_products:.0f} units. "
            f"Supply chain operating without political disruptions."
        )

    # LEGAL STATUS: Based on inspection results and defect rates
    # (simulating regulatory compliance)
    fail_ratio = fail_count / total_products if total_products > 0 else 0
    if fail_ratio > 0.3 or avg_defect_rate > 3.0:
        insights["legal_status"] = "COMPLIANCE RISK"
        insights["legal_details"] = (
            f"Quality concerns: {fail_count}/{total_products} {product_category} items failed inspection. "
            f"Avg defect rate: {avg_defect_rate:.2f}%. High defects on {high_defect_count} items. "
            f"Review quality standards and regulatory compliance."
        )
        insights["alerts"].append("High failure rate - review regulatory compliance")
    elif pending_count > pass_count:
        insights["legal_status"] = "PENDING REVIEW"
        insights["legal_details"] = (
            f"Many {product_category} items pending inspection ({pending_count}/{total_products}). "
            f"Pass rate: {pass_count}, Fail rate: {fail_count}. "
            f"Expedite quality reviews to ensure compliance."
        )
        insights["alerts"].append("Multiple items pending inspection review")
    else:
        insights["legal_status"] = "COMPLIANT"
            insights["legal_details"] = (
            f"{product_category.capitalize()} products meeting quality standards. "
            f"Pass: {pass_count}, Fail: {fail_count}, Pending: {pending_count}. "
            f"Avg defect rate: {avg_defect_rate:.2f}%. Regulatory compliance maintained."
            )

    # Determine overall risk level
    alert_count = len(insights["alerts"])
    if alert_count == 0:
        insights["overall_risk"] = "LOW"
    elif alert_count == 1:
        insights["overall_risk"] = "MEDIUM"
    elif alert_count <= 3:
        insights["overall_risk"] = "HIGH"
    else:
        insights["overall_risk"] = "CRITICAL"

    return insights


@performance_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent on startup"""
    ctx.logger.info("Performance Agent starting up...")
    ctx.logger.info(f"Agent address: {performance_agent.address}")

    # Load performance prompt
    ctx.storage.set("performance_prompt", PROMPT)

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info("Performance Agent ready to receive requests!")


@performance_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Performance Agent shutting down...")


@performance_protocol.on_message(model=PerformanceRequest, replies=PerformanceResponse)
async def handle_performance_request(
    ctx: Context, sender: str, msg: PerformanceRequest
):
    """
    Handle performance monitoring request from Orchestrator Agent.

    Process:
    1. Receive supplier name and product category from orchestrator
    2. Load performance CSV data and filter by product category
    3. Analyze filtered data for weather, strikes, politics, and legal insights
    4. Return performance update to orchestrator
    """
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 PERFORMANCE AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier Name: {msg.supplier_name}")
    ctx.logger.info(f"Product Category: {msg.product_category}")
    ctx.logger.info("=" * 70)

    try:
        # Load and filter CSV data by product category
        product_category = msg.product_category or "food"
        ctx.logger.info(f"Loading performance data for category: {product_category}")

        performance_data = load_performance_csv_data(product_category)
        ctx.logger.info(f"Found {len(performance_data)} {product_category} products")

        # Analyze the filtered data
        insights = analyze_performance_data(performance_data, product_category)

        # Build response from analyzed insights
            response = PerformanceResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                weather_status=insights["weather_status"],
                weather_details=insights["weather_details"],
                strike_status=insights["strike_status"],
                strike_details=insights["strike_details"],
                political_status=insights["political_status"],
                political_details=insights["political_details"],
                legal_status=insights["legal_status"],
                legal_details=insights["legal_details"],
                overall_risk_level=insights["overall_risk"],
                alerts=insights.get("alerts", []),
                timestamp="",
            )

        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("📤 PERFORMANCE AGENT: SENDING RESPONSE")
        ctx.logger.info("=" * 70)
        ctx.logger.info(f"To: {sender}")
        ctx.logger.info(f"Overall Risk: {response.overall_risk_level}")
        ctx.logger.info(f"Weather: {response.weather_status}")
        ctx.logger.info(f"Strikes: {response.strike_status}")
        ctx.logger.info(f"Political: {response.political_status}")
        ctx.logger.info(f"Legal: {response.legal_status}")
        ctx.logger.info(f"Alerts: {len(response.alerts)}")
        ctx.logger.info("=" * 70)

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

    except Exception as e:
        ctx.logger.error(f"Error processing performance request: {e}")
        import traceback

        traceback.print_exc()

        # Send error response
        error_response = PerformanceResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            weather_status="ERROR",
            weather_details=f"Error retrieving monitoring data: {str(e)}",
            strike_status="ERROR",
            strike_details="Error retrieving data",
            political_status="ERROR",
            political_details="Error retrieving data",
            legal_status="ERROR",
            legal_details="Error retrieving data",
            overall_risk_level="UNKNOWN",
            alerts=[f"Error processing request: {str(e)}"],
            timestamp="",
        )

        await ctx.send(sender, error_response)


performance_agent.include(performance_protocol, publish_manifest=True)

if __name__ == "__main__":
    performance_agent.run()
