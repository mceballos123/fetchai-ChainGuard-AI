from uagents import Agent, Context, Protocol
import os
import csv
import re
from typing import Dict, Any, List
import google.generativeai as genai

# Import models
from models.demand import DemandRequest, DemandResponse
from prompts.demand_prompt import DEMAND_FORECASE_PROMPT
from prompts.demand_data_prompt import analyze_demand_data_prompt
from dotenv import load_dotenv

load_dotenv()

# Gemini configuration - OPTIMIZED: Using lite model to reduce costs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

demand_agent = Agent(
    name="demand_agent",
    seed=os.getenv("DEMAND_AGENT_SEED"),
    port=8005,
    endpoint=[os.getenv("DEMAND_AGENT_ENDPOINT")],  # Docker network endpoint
    mailbox=False,  # Local communication without Agentverse mailbox
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


def format_csv_data_for_llm(data: List[Dict[str, Any]], max_rows: int = 15) -> str:
    """
    Format CSV data into a readable string for the LLM to analyze.
    OPTIMIZED: Removed detailed row data, keeping only statistics to save tokens.
    """
    if not data:
        return "No data available"

    headers = list(data[0].keys())

    formatted = "CSV Data Summary:\n"
    formatted += "-" * 60 + "\n"
    formatted += f"Total rows: {len(data)}\n"
    formatted += f"Columns: {', '.join(headers)}\n"
    formatted += "-" * 60 + "\n\n"

    formatted += "Data Statistics:\n"
    formatted += "-" * 40 + "\n"

    try:
        prices = [float(row.get("Price", 0)) for row in data]
        availability = [int(row.get("Availability", 0)) for row in data]
        sold = [int(row.get("Number of products sold", 0)) for row in data]
        revenue = [float(row.get("Revenue generated", 0)) for row in data]

        formatted += f"Average Price: ${sum(prices)/len(prices):.2f}\n"
        formatted += (
            f"Average Availability: {sum(availability)/len(availability):.1f}%\n"
        )
        formatted += f"Total Products Sold: {sum(sold):,}\n"
        formatted += f"Total Revenue: ${sum(revenue):,.2f}\n"
        formatted += f"Low Availability (<30%): {len([a for a in availability if a < 30])} items\n"
        formatted += f"High Availability (>70%): {len([a for a in availability if a > 70])} items\n"
    except Exception as e:
        formatted += f"Could not calculate statistics: {e}\n"

    return formatted


def analyze_demand_with_gemini(
    data: List[Dict[str, Any]], product_category: str, supplier_name: str
) -> Dict[str, Any]:
    """
    Use Google Gemini LLM to analyze demand data using the DEMAND_FORECASE_PROMPT.

    Args:
        data: List of filtered demand data rows
        product_category: The product category being analyzed
        supplier_name: Name of the supplier

    Returns:
        Dictionary with demand insights from LLM analysis
    """
    insights = {
        "delay_status": "ON TIME",
        "delay_details": "",
        "shipping_status": "NORMAL",
        "shipping_details": "",
        "quality_status": "GOOD",
        "quality_details": "",
        "alerts": [],
        "overall_performance": "GOOD",
    }

    if not data:
        insights["delay_status"] = "UNKNOWN"
        insights["delay_details"] = f"No data available for {product_category} products"
        insights["overall_performance"] = "UNKNOWN"
        insights["alerts"].append(
            f"No {product_category} products found in monitoring data"
        )
        return insights

    try:
        data_summary = format_csv_data_for_llm(data)

        full_prompt = analyze_demand_data_prompt(DEMAND_FORECASE_PROMPT, supplier_name, product_category, data_summary)

        response = gemini_model.generate_content(full_prompt)
        llm_response = response.text

        # Parse LLM response
        insights["delay_status"] = extract_field(
            llm_response, "DELAY_STATUS", "ON TIME"
        )
        insights["delay_details"] = (
            f"ChainGuard AI Analysis: {extract_field(llm_response, 'DELAY_DETAILS', 'Analysis pending')}"
        )
        insights["shipping_status"] = extract_field(
            llm_response, "SHIPPING_STATUS", "NORMAL"
        )
        insights["shipping_details"] = (
            f"ChainGuard AI Analysis: {extract_field(llm_response, 'SHIPPING_DETAILS', 'Analysis pending')}"
        )
        insights["quality_status"] = extract_field(
            llm_response, "QUALITY_STATUS", "GOOD"
        )
        insights["quality_details"] = (
            f"ChainGuard AI Analysis: {extract_field(llm_response, 'QUALITY_DETAILS', 'Analysis pending')}"
        )
        insights["overall_performance"] = extract_field(
            llm_response, "OVERALL_PERFORMANCE", "GOOD"
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
        insights["delay_details"] = (
            f"ChainGuard AI Analysis: Monitoring {len(data)} {product_category} products. "
            f"LLM analysis unavailable - using default assessment."
        )
        insights["shipping_details"] = (
            f"ChainGuard AI Analysis: Standard shipping volume for {product_category}."
        )
        insights["quality_details"] = (
            f"ChainGuard AI Analysis: Quality monitoring active for {product_category}."
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
    3. Use Gemini LLM to analyze the data with the prompt
    4. Return demand forecast update to orchestrator
    """
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("DEMAND AGENT: RECEIVED REQUEST")
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

        # Analyze with Gemini LLM
        ctx.logger.info("Sending data to Gemini for analysis...")
        insights = analyze_demand_with_gemini(
            demand_data, product_category, msg.supplier_name
        )
        ctx.logger.info("Gemini analysis complete")

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
        ctx.logger.info("DEMAND AGENT: SENDING RESPONSE")
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
