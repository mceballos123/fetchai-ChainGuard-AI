from uagents import Agent, Context, Protocol
import os
import csv
from typing import Dict, Any, List

# Import models
from models.logistics import LogisticsRequest, LogisticsResponse
from prompts.logistics_prompt import LOGISTICS_PROMPT
from dotenv import load_dotenv

load_dotenv()

logistics_agent = Agent(
    name="logistics_agent",
    seed=os.getenv("LOGISTICS_AGENT_SEED"),
    port=8006,
    mailbox=True,
)

logistics_protocol = Protocol(name="logistics_protocol", version="1.0")

# Path to logistics monitoring CSV data
LOGISTICS_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "../../data_for_monitor/logistics_monitoring_data.csv"
)


def load_logistics_csv_data(product_category: str) -> List[Dict[str, Any]]:
    """
    Load logistics monitoring data from CSV and filter by product category.

    Args:
        product_category: The product type to filter by (food, clothing, electronics)

    Returns:
        List of dictionaries containing filtered logistics data
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


def analyze_logistics_data(
    data: List[Dict[str, Any]], product_category: str
) -> Dict[str, Any]:
    """
    Analyze logistics data for inventory levels and predictions.

    CSV columns: Product type, SKU, Lead times, Shipping times, Shipping carriers,
                 Shipping costs, Transportation modes, Routes, Costs, Supplier name, Location

    Args:
        data: List of filtered logistics data rows
        product_category: The product category being analyzed

    Returns:
        Dictionary with logistics and inventory insights
    """
    insights = {
        "current_inventory_level": "MEDIUM",
        "current_inventory_units": 0,
        "inventory_details": "",
        "predicted_inventory_level": "MEDIUM",
        "predicted_inventory_units": 0,
        "inventory_forecast": "",
        "restocking_status": "ON_TIME",
        "restocking_details": "",
        "overall_logistics_status": "HEALTHY",
        "alerts": [],
    }

    if not data:
        insights["current_inventory_level"] = "UNKNOWN"
        insights["inventory_details"] = (
            f"No data available for {product_category} products"
        )
        insights["overall_logistics_status"] = "UNKNOWN"
        insights["alerts"].append(
            f"No {product_category} products found in logistics data"
        )
        return insights

    # Analyze the data
    total_products = len(data)
    total_lead_time = 0
    total_shipping_time = 0
    total_shipping_cost = 0
    total_transport_cost = 0
    long_lead_time_count = 0
    high_cost_count = 0

    # Track transportation modes and routes
    transport_modes = {}
    carriers = {}
    locations = {}

    for row in data:
        try:
            lead_time = int(row.get("Lead times", 0))
            shipping_time = int(row.get("Shipping times", 0))
            shipping_cost = float(row.get("Shipping costs", 0))
            transport_cost = float(row.get("Costs", 0))
            transport_mode = row.get("Transportation modes", "Unknown")
            carrier = row.get("Shipping carriers", "Unknown")
            location = row.get("Location", "Unknown")

            total_lead_time += lead_time
            total_shipping_time += shipping_time
            total_shipping_cost += shipping_cost
            total_transport_cost += transport_cost

            # Track long lead times (>20 days)
            if lead_time > 20:
                long_lead_time_count += 1

            # Track high cost shipments
            if transport_cost > 700:
                high_cost_count += 1

            # Count transport modes
            transport_modes[transport_mode] = transport_modes.get(transport_mode, 0) + 1

            # Count carriers
            carriers[carrier] = carriers.get(carrier, 0) + 1

            # Count locations
            locations[location] = locations.get(location, 0) + 1

        except (ValueError, TypeError):
            continue

    avg_lead_time = total_lead_time / total_products if total_products > 0 else 0
    avg_shipping_time = (
        total_shipping_time / total_products if total_products > 0 else 0
    )
    avg_shipping_cost = (
        total_shipping_cost / total_products if total_products > 0 else 0
    )
    avg_transport_cost = (
        total_transport_cost / total_products if total_products > 0 else 0
    )

    # Estimate inventory based on lead times and product count
    # Lower lead times = higher inventory availability
    estimated_inventory = int(total_products * (30 - avg_lead_time) * 100)
    if estimated_inventory < 0:
        estimated_inventory = total_products * 500

    insights["current_inventory_units"] = estimated_inventory

    # Determine inventory level
    if estimated_inventory > 50000:
        insights["current_inventory_level"] = "HIGH"
    elif estimated_inventory > 20000:
        insights["current_inventory_level"] = "MEDIUM"
    elif estimated_inventory > 5000:
        insights["current_inventory_level"] = "LOW"
    else:
        insights["current_inventory_level"] = "CRITICAL"

    # Build inventory details
    most_common_mode = (
        max(transport_modes, key=transport_modes.get) if transport_modes else "Unknown"
    )
    most_common_carrier = max(carriers, key=carriers.get) if carriers else "Unknown"
    most_common_location = max(locations, key=locations.get) if locations else "Unknown"

    insights["inventory_details"] = (
        f"Monitoring {total_products} {product_category} SKUs. "
        f"Primary transport: {most_common_mode}. Main carrier: {most_common_carrier}. "
        f"Primary source: {most_common_location}. Avg shipping cost: ${avg_shipping_cost:.2f}."
    )

    # Predict future inventory based on lead times
    if avg_lead_time > 20:
        insights["predicted_inventory_level"] = "LOW"
        predicted_units = int(estimated_inventory * 0.6)
        insights["predicted_inventory_units"] = predicted_units
        insights["inventory_forecast"] = (
            f"Long average lead time ({avg_lead_time:.0f} days) for {product_category}. "
            f"Inventory expected to decrease to {predicted_units:,} units. "
            f"Recommend accelerating restocking orders."
        )
        insights["alerts"].append(
            f"High lead times ({avg_lead_time:.0f} days) - plan restocking early"
        )
    elif avg_lead_time > 15:
        insights["predicted_inventory_level"] = "MEDIUM"
        predicted_units = int(estimated_inventory * 0.8)
        insights["predicted_inventory_units"] = predicted_units
        insights["inventory_forecast"] = (
            f"Moderate lead times ({avg_lead_time:.0f} days) for {product_category}. "
            f"Inventory forecast: {predicted_units:,} units. Monitor closely."
        )
    else:
        insights["predicted_inventory_level"] = "STABLE"
        predicted_units = int(estimated_inventory * 0.95)
        insights["predicted_inventory_units"] = predicted_units
        insights["inventory_forecast"] = (
            f"Good lead times ({avg_lead_time:.0f} days) for {product_category}. "
            f"Inventory expected to remain stable at ~{predicted_units:,} units."
        )

    # Determine restocking status
    long_lead_ratio = long_lead_time_count / total_products if total_products > 0 else 0
    if long_lead_ratio > 0.4:
        insights["restocking_status"] = "DELAYED"
        insights["restocking_details"] = (
            f"{long_lead_time_count}/{total_products} {product_category} items have extended lead times (>20 days). "
            f"Average shipping time: {avg_shipping_time:.0f} days. Consider alternative suppliers or expedited shipping."
        )
        insights["alerts"].append("Multiple items with extended lead times")
    elif long_lead_ratio > 0.2:
        insights["restocking_status"] = "MONITOR"
        insights["restocking_details"] = (
            f"Some {product_category} items have extended lead times. "
            f"Average lead time: {avg_lead_time:.0f} days, shipping: {avg_shipping_time:.0f} days."
        )
    else:
        insights["restocking_status"] = "ON_TIME"
        insights["restocking_details"] = (
            f"Restocking on schedule for {product_category}. "
            f"Average lead time: {avg_lead_time:.0f} days, shipping: {avg_shipping_time:.0f} days. "
            f"Avg transport cost: ${avg_transport_cost:.2f}."
        )

    # Check for high cost alerts
    if high_cost_count > total_products * 0.3:
        insights["alerts"].append(
            f"High transport costs on {high_cost_count} items - review logistics routes"
        )

    # Determine overall logistics status
    alert_count = len(insights["alerts"])
    if alert_count == 0:
        insights["overall_logistics_status"] = "HEALTHY"
    elif alert_count == 1:
        insights["overall_logistics_status"] = "HEALTHY"
    elif alert_count == 2:
        insights["overall_logistics_status"] = "WARNING"
    else:
        insights["overall_logistics_status"] = "CRITICAL"

    return insights


@logistics_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent on startup"""
    ctx.logger.info("Logistics Agent starting up...")
    ctx.logger.info(f"Agent address: {logistics_agent.address}")

    # Load logistics prompt
    ctx.storage.set("logistics_prompt", LOGISTICS_PROMPT)

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info("Logistics Agent ready to receive requests!")


@logistics_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Logistics Agent shutting down...")


@logistics_protocol.on_message(model=LogisticsRequest, replies=LogisticsResponse)
async def handle_logistics_request(ctx: Context, sender: str, msg: LogisticsRequest):
    """
    Handle logistics/inventory request from Orchestrator Agent.

    Process:
    1. Receive supplier name and product category from orchestrator
    2. Load logistics CSV data and filter by product category
    3. Analyze filtered data for inventory levels and predictions
    4. Return logistics monitoring update to orchestrator
    """
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 LOGISTICS AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier Name: {msg.supplier_name}")
    ctx.logger.info(f"Product Category: {msg.product_category}")
    ctx.logger.info("=" * 70)

    try:
        # Load and filter CSV data by product category
        product_category = msg.product_category or "food"
        ctx.logger.info(f"Loading logistics data for category: {product_category}")

        logistics_data = load_logistics_csv_data(product_category)
        ctx.logger.info(f"Found {len(logistics_data)} {product_category} products")

        # Analyze the filtered data
        insights = analyze_logistics_data(logistics_data, product_category)

        # Build response from analyzed insights
        response = LogisticsResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            current_inventory_level=insights["current_inventory_level"],
            current_inventory_units=insights["current_inventory_units"],
            inventory_details=insights["inventory_details"],
            predicted_inventory_level=insights["predicted_inventory_level"],
            predicted_inventory_units=insights["predicted_inventory_units"],
            inventory_forecast=insights["inventory_forecast"],
            restocking_status=insights["restocking_status"],
            restocking_details=insights["restocking_details"],
            overall_logistics_status=insights["overall_logistics_status"],
            alerts=insights.get("alerts", []),
            timestamp="",
        )

        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("📤 LOGISTICS AGENT: SENDING RESPONSE")
        ctx.logger.info("=" * 70)
        ctx.logger.info(f"To: {sender}")
        ctx.logger.info(f"Overall Status: {response.overall_logistics_status}")
        ctx.logger.info(f"Current Inventory: {response.current_inventory_level}")
        ctx.logger.info(f"Predicted Inventory: {response.predicted_inventory_level}")
        ctx.logger.info(f"Alerts: {len(response.alerts)}")
        ctx.logger.info("=" * 70)

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

    except Exception as e:
        ctx.logger.error(f"Error processing logistics request: {e}")
        import traceback

        traceback.print_exc()

        # Send error response
        error_response = LogisticsResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            current_inventory_level="ERROR",
            current_inventory_units=0,
            inventory_details=f"Error retrieving logistics data: {str(e)}",
            predicted_inventory_level="ERROR",
            predicted_inventory_units=0,
            inventory_forecast="Error retrieving data",
            restocking_status="ERROR",
            restocking_details="Error retrieving data",
            overall_logistics_status="UNKNOWN",
            alerts=[f"Error processing request: {str(e)}"],
            timestamp="",
        )

        await ctx.send(sender, error_response)


logistics_agent.include(logistics_protocol, publish_manifest=True)

if __name__ == "__main__":
    logistics_agent.run()
