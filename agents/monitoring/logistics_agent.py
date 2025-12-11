from uagents import Agent, Context, Protocol
import os
import csv
import re
from typing import Dict, Any, List
import google.generativeai as genai

# Import models
from models.logistics import LogisticsRequest, LogisticsResponse
from prompts.logistics_prompt import LOGISTICS_PROMPT
from prompts.logistics_data_prompt import analyze_logistics_data_prompt
from dotenv import load_dotenv

load_dotenv()

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

logistics_agent = Agent(
    name="logistics_agent",
    seed=os.getenv("LOGISTICS_AGENT_SEED"),
    port=8008,
    endpoint=[os.getenv("LOGISTICS_AGENT_ENDPOINT")],  # Docker network endpoint
    mailbox=False,  # Local communication without Agentverse mailbox
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


def format_csv_data_for_llm(data: List[Dict[str, Any]], max_rows: int = 15) -> str:
    """
    Format CSV data into a readable string for the LLM to analyze.
    """
    if not data:
        return "No data available"

    headers = list(data[0].keys())

    formatted = "CSV Data Summary:\n"
    formatted += "-" * 60 + "\n"
    formatted += f"Total rows: {len(data)}\n"
    formatted += f"Columns: {', '.join(headers)}\n"
    formatted += "-" * 60 + "\n\n"

    formatted += f"Sample Data (first {min(max_rows, len(data))} rows):\n\n"

    for i, row in enumerate(data[:max_rows]):
        formatted += f"Row {i+1}:\n"
        for key, value in row.items():
            formatted += f"  {key}: {value}\n"
        formatted += "\n"

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


def analyze_logistics_with_gemini(
    data: List[Dict[str, Any]], product_category: str, supplier_name: str
) -> Dict[str, Any]:
    """
    Use Google Gemini LLM to analyze logistics data using the LOGISTICS_PROMPT.

    Args:
        data: List of filtered logistics data rows
        product_category: The product category being analyzed
        supplier_name: Name of the supplier

    Returns:
        Dictionary with logistics and inventory insights from LLM analysis
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

    # Calculate estimated inventory for the response
    total_products = len(data)
    try:
        lead_times = [int(row.get("Lead times", 0)) for row in data]
        avg_lead_time = sum(lead_times) / len(lead_times)
        estimated_inventory = int(total_products * (30 - avg_lead_time) * 100)
        if estimated_inventory < 0:
            estimated_inventory = total_products * 500
    except Exception:
        estimated_inventory = total_products * 1000
        avg_lead_time = 15

    insights["current_inventory_units"] = estimated_inventory
    insights["predicted_inventory_units"] = int(estimated_inventory * 0.85)

    try:
        data_summary = format_csv_data_for_llm(data)

        full_prompt = analyze_logistics_data_prompt(LOGISTICS_PROMPT, supplier_name, product_category, estimated_inventory, data_summary)

        response = gemini_model.generate_content(full_prompt)
        llm_response = response.text

        # Parse LLM response
        insights["current_inventory_level"] = extract_field(
            llm_response, "CURRENT_INVENTORY_LEVEL", "MEDIUM"
        )
        insights["inventory_details"] = (
            f"ChainGuard AI Analysis: {extract_field(llm_response, 'INVENTORY_DETAILS', 'Analysis pending')}"
        )
        insights["predicted_inventory_level"] = extract_field(
            llm_response, "PREDICTED_INVENTORY_LEVEL", "MEDIUM"
        )
        insights["inventory_forecast"] = (
            f"ChainGuard AI Analysis: {extract_field(llm_response, 'INVENTORY_FORECAST', 'Analysis pending')}"
        )
        insights["restocking_status"] = extract_field(
            llm_response, "RESTOCKING_STATUS", "ON_TIME"
        )
        insights["restocking_details"] = (
            f"ChainGuard AI Analysis: {extract_field(llm_response, 'RESTOCKING_DETAILS', 'Analysis pending')}"
        )
        insights["overall_logistics_status"] = extract_field(
            llm_response, "OVERALL_LOGISTICS_STATUS", "HEALTHY"
        )

        # Parse alerts
        alerts_text = extract_field(llm_response, "ALERTS", "None")
        if alerts_text.lower() != "none" and alerts_text:
            insights["alerts"] = [
                a.strip() for a in alerts_text.split(";") if a.strip()
            ]

    except Exception as e:
        print(f"Gemini error: {e}")
        # Fallback to basic analysis
        insights["inventory_details"] = (
            f"ChainGuard AI Analysis: Monitoring {total_products} {product_category} products. "
            f"Estimated inventory: {estimated_inventory:,} units. LLM analysis unavailable."
        )
        insights["inventory_forecast"] = (
            f"ChainGuard AI Analysis: Based on avg lead time of {avg_lead_time:.0f} days."
        )
        insights["restocking_details"] = (
            f"ChainGuard AI Analysis: Restocking monitoring active for {product_category}."
        )

    return insights


def extract_field(text: str, field_name: str, default: str = "") -> str:
    """Extract a field value from LLM response."""
    patterns = [
        rf"{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)",
        rf"\*\*{field_name}\*\*:\s*(.+?)(?=\n|$)",
        rf"{field_name}\s*[:=]\s*(.+?)(?=\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            # Clean up the value
            value = re.sub(r"^\[|\]$", "", value)
            value = value.strip()
            if value:
                return value

    return default


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
    3. Use Gemini LLM to analyze the data with the prompt
    4. Return logistics monitoring update to orchestrator
    """
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("LOGISTICS AGENT: RECEIVED REQUEST")
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

        # Analyze with Gemini LLM
        ctx.logger.info("Sending data to Gemini for analysis...")
        insights = analyze_logistics_with_gemini(
            logistics_data, product_category, msg.supplier_name
        )
        ctx.logger.info("Gemini analysis complete")

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
        ctx.logger.info("LOGISTICS AGENT: SENDING RESPONSE")
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
