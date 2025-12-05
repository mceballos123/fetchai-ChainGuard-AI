from uagents import Agent, Context, Protocol
import os
import csv
from typing import Dict, Any, List

# Import models
from models.demand import DemandRequest, DemandResponse
from prompts.demand_prompt import DEMAND_FORECASE_PROMPT
from dotenv import load_dotenv

load_dotenv()

demand_agent = Agent(
    name="demand_agent",
    seed=os.getenv("DEMAND_AGENT_SEED"),
    port=8005,
    mailbox=True,
)

demand_protocol = Protocol(name="demand_protocol", version="1.0")

# Path to demand monitoring CSV data
DEMAND_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "../../data_for_monitor/demand_monitoring_data.csv"
)


def load_demand_csv_data(product_category: str) -> List[Dict[str, Any]]:
    """
    Load demand monitoring data from CSV and filter by product category.

    Args:
        product_category: The product type to filter by (food, clothing, electronics)

    Returns:
        List of dictionaries containing filtered demand data
    """
    filtered_data = []

    try:
        with open(DEMAND_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Product type", "").lower() == product_category.lower():
                    filtered_data.append(row)
    except Exception as e:
        print(f"Error loading demand CSV: {e}")

    return filtered_data


def analyze_demand_data(
    data: List[Dict[str, Any]], product_category: str
) -> Dict[str, Any]:
    """
    Analyze demand data for delays, quality, and shipping insights.

    CSV columns: Product type, SKU, Price, Availability, Number of products sold,
                 Revenue generated, Customer demographics

    Args:
        data: List of filtered demand data rows
        product_category: The product category being analyzed

    Returns:
        Dictionary with demand insights
    """
    insights = {
        "delay_status": "ON TIME",
        "delay_details": "",
        "shipping_status": "NORMAL",
        "shipping_details": "",
        "quality_status": "EXCELLENT",
        "quality_details": "",
        "alerts": [],
        "overall_performance": "EXCELLENT",
    }

    if not data:
        insights["delay_status"] = "UNKNOWN"
        insights["delay_details"] = f"No data available for {product_category} products"
        insights["overall_performance"] = "UNKNOWN"
        insights["alerts"].append(
            f"No {product_category} products found in monitoring data"
        )
        return insights

    # Analyze the data
    total_products = len(data)
    total_sold = 0
    total_revenue = 0
    low_availability_count = 0
    high_price_count = 0

    for row in data:
        try:
            availability = int(row.get("Availability", 0))
            products_sold = int(row.get("Number of products sold", 0))
            revenue = float(row.get("Revenue generated", 0))
            price = float(row.get("Price", 0))

            total_sold += products_sold
            total_revenue += revenue

            # Check for low availability (potential delays)
            if availability < 30:
                low_availability_count += 1

            # Check for high-priced items (quality indicator)
            if price > 70:
                high_price_count += 1

        except (ValueError, TypeError):
            continue

    avg_sold = total_sold / total_products if total_products > 0 else 0
    avg_revenue = total_revenue / total_products if total_products > 0 else 0

    # Determine delay status based on availability
    low_availability_ratio = (
        low_availability_count / total_products if total_products > 0 else 0
    )
    if low_availability_ratio > 0.5:
        insights["delay_status"] = "DELAYS DETECTED"
        insights["delay_details"] = (
            f"High demand detected: {low_availability_count}/{total_products} {product_category} products "
            f"have low availability (<30%). Average products sold: {avg_sold:.0f}. "
            f"Consider restocking to meet demand."
        )
        insights["alerts"].append(
            f"Low availability on {low_availability_count} {product_category} items"
        )
    elif low_availability_ratio > 0.3:
        insights["delay_status"] = "MONITOR: AVAILABILITY"
        insights["delay_details"] = (
            f"Moderate demand pressure: {low_availability_count}/{total_products} {product_category} products "
            f"have limited availability. Monitor closely for potential delays."
        )
        insights["alerts"].append("Moderate availability constraints detected")
    else:
        insights["delay_status"] = "ON TIME"
        insights["delay_details"] = (
            f"Good availability across {product_category} products. {total_products} SKUs monitored "
            f"with average revenue of ${avg_revenue:.2f} per product."
        )

    # Determine shipping status based on sales volume
    if avg_sold > 500:
        insights["shipping_status"] = "HIGH VOLUME"
        insights["shipping_details"] = (
            f"High shipping volume for {product_category}: Average {avg_sold:.0f} units sold per SKU. "
            f"Ensure logistics capacity can handle demand."
        )
        insights["alerts"].append("High shipping volume - monitor logistics capacity")
    elif avg_sold > 300:
        insights["shipping_status"] = "MODERATE VOLUME"
        insights["shipping_details"] = (
            f"Moderate shipping activity for {product_category}: Average {avg_sold:.0f} units per SKU. "
            f"Shipping operations within normal parameters."
        )
    else:
        insights["shipping_status"] = "NORMAL"
        insights["shipping_details"] = (
            f"Standard shipping volume for {product_category}: Average {avg_sold:.0f} units per SKU. "
            f"No shipping concerns detected."
        )

    # Determine quality status based on price distribution (premium products = quality focus)
    quality_ratio = high_price_count / total_products if total_products > 0 else 0
    if quality_ratio > 0.3:
        insights["quality_status"] = "PREMIUM"
        insights["quality_details"] = (
            f"Premium {product_category} portfolio: {high_price_count}/{total_products} products are high-value items. "
            f"Maintain strict quality control for premium products."
        )
    elif quality_ratio > 0.1:
        insights["quality_status"] = "GOOD"
        insights["quality_details"] = (
            f"Balanced {product_category} portfolio with mix of price points. "
            f"Quality standards being maintained across product range."
        )
    else:
        insights["quality_status"] = "STANDARD"
        insights["quality_details"] = (
            f"Standard {product_category} products in portfolio. "
            f"Regular quality monitoring recommended."
        )

    # Determine overall performance
    alert_count = len(insights["alerts"])
    if alert_count == 0:
        insights["overall_performance"] = "EXCELLENT"
    elif alert_count == 1:
        insights["overall_performance"] = "GOOD"
    elif alert_count == 2:
        insights["overall_performance"] = "FAIR"
    else:
        insights["overall_performance"] = "POOR"

    return insights


@demand_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent on startup"""
    ctx.logger.info("Demand Agent starting up...")
    ctx.logger.info(f"Agent address: {demand_agent.address}")

    # Load demand forecast prompt
    ctx.storage.set("demand_prompt", DEMAND_FORECASE_PROMPT)

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info("Demand Agent ready to receive requests!")


@demand_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Demand Agent shutting down...")


@demand_protocol.on_message(model=DemandRequest, replies=DemandResponse)
async def handle_demand_request(ctx: Context, sender: str, msg: DemandRequest):
    """
    Handle demand forecast request from Orchestrator Agent.

    Process:
    1. Receive supplier name and product category from orchestrator
    2. Load demand CSV data and filter by product category
    3. Analyze filtered data for delays, quality, and shipping insights
    4. Return demand forecast update to orchestrator
    """
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 DEMAND AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier Name: {msg.supplier_name}")
    ctx.logger.info(f"Product Category: {msg.product_category}")
    ctx.logger.info("=" * 70)

    try:
        # Load and filter CSV data by product category
        product_category = msg.product_category or "food"
        ctx.logger.info(f"Loading demand data for category: {product_category}")

        demand_data = load_demand_csv_data(product_category)
        ctx.logger.info(f"Found {len(demand_data)} {product_category} products")

        # Analyze the filtered data
        insights = analyze_demand_data(demand_data, product_category)

        # Build response from analyzed insights
        response = DemandResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            delay_status=insights["delay_status"],
            delay_details=insights["delay_details"],
            shipping_status=insights["shipping_status"],
            shipping_details=insights["shipping_details"],
            quality_status=insights["quality_status"],
            quality_details=insights["quality_details"],
            overall_performance=insights["overall_performance"],
            alerts=insights.get("alerts", []),
            timestamp="",
        )

        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("📤 DEMAND AGENT: SENDING RESPONSE")
        ctx.logger.info("=" * 70)
        ctx.logger.info(f"To: {sender}")
        ctx.logger.info(f"Overall Performance: {response.overall_performance}")
        ctx.logger.info(f"Delay Status: {response.delay_status}")
        ctx.logger.info(f"Alerts: {len(response.alerts)}")
        ctx.logger.info("=" * 70)

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

    except Exception as e:
        ctx.logger.error(f"Error processing demand request: {e}")
        import traceback

        traceback.print_exc()

        # Send error response
        error_response = DemandResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            delay_status="ERROR",
            delay_details=f"Error retrieving demand forecast data: {str(e)}",
            shipping_status="ERROR",
            shipping_details="Error retrieving data",
            quality_status="ERROR",
            quality_details="Error retrieving data",
            overall_performance="UNKNOWN",
            alerts=[f"Error processing request: {str(e)}"],
            timestamp="",
        )

        await ctx.send(sender, error_response)


demand_agent.include(demand_protocol, publish_manifest=True)

if __name__ == "__main__":
    demand_agent.run()
