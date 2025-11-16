from uagents import Agent, Context, Protocol
import os
import json
from typing import Dict, Any, Optional

# Import models
from backend.models.logistics import LogisticsRequest, LogisticsResponse
from backend.prompts.logistics_prompt import LOGISTICS_PROMPT
from dotenv import load_dotenv

load_dotenv()

logistics_agent = Agent(
    name="logistics_agent",
    seed=os.getenv("LOGISTICS_AGENT_SEED"),
    port=8006,
    mailbox=True,
)

logistics_protocol = Protocol(name="logistics_protocol", version="1.0")

# Paths to supplier data files
COMPLIANCE_FILES_PATH = os.path.join(
    os.path.dirname(__file__), "../../compliance_files"
)
FINANCIAL_FILES_PATH = os.path.join(os.path.dirname(__file__), "../../financial_files")
RISK_MANAGEMENT_FILES_PATH = os.path.join(
    os.path.dirname(__file__), "../../risk_management_files"
)


def load_supplier_file(supplier_name: str, file_path: str) -> Optional[str]:
    """
    Load supplier data from a file.

    Args:
        supplier_name: The name of the supplier (converted to lowercase with underscores)
        file_path: The directory path where the file is located

    Returns:
        The file contents as a string, or None if file not found
    """
    # Convert supplier name to file format (e.g., "Green Valley" -> "green_valley")
    file_name = f"{supplier_name.lower().replace(' ', '_')}.txt"
    full_path = os.path.join(file_path, file_name)

    try:
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read()
        else:
            return None
    except Exception as e:
        print(f"Error loading file {full_path}: {e}")
        return None


def extract_logistics_insights(
    compliance_data: str, risk_data: str, financial_data: str
) -> Dict[str, Any]:
    """
    Extract logistics/inventory insights from supplier files using the logistics prompt guidelines.

    The logistics_prompt directs the agent to monitor:
    - CURRENT INVENTORY LEVELS
    - PREDICTOR FOR FUTURE INVENTORY LEVELS

    Args:
        compliance_data: Contents of compliance file
        risk_data: Contents of risk management file
        financial_data: Contents of financial file

    Returns:
        Dictionary with logistics and inventory insights
    """
    insights = {
        "current_inventory_level": "MEDIUM",
        "current_inventory_units": 5000,
        "inventory_details": "",
        "predicted_inventory_level": "MEDIUM",
        "predicted_inventory_units": 4800,
        "inventory_forecast": "",
        "restocking_status": "ON_TIME",
        "restocking_details": "",
        "overall_logistics_status": "HEALTHY",
        "alerts": [],
    }

    # Extract inventory information from financial data
    if financial_data:
        # Check for inventory mentions
        if "inventory" in financial_data.lower():
            insights["inventory_details"] = (
                "Inventory management system is active. Current stock levels are being monitored."
            )

            # Check for warehouse/capacity info
            if "warehouse" in financial_data.lower() or "capacity" in financial_data.lower():
                insights["current_inventory_level"] = "HIGH"
                insights["current_inventory_units"] = 8500
                insights["inventory_details"] += " Warehouse capacity is optimal."
                insights["alerts"].append("High inventory levels detected - plan sales activities")

            # Check for storage/stockpile info
            if "stockpile" in financial_data.lower():
                insights["current_inventory_level"] = "CRITICAL"
                insights["current_inventory_units"] = 12000
                insights["inventory_details"] += " Stockpiling detected - urgently reduce inventory."
                insights["alerts"].append("CRITICAL: Excess inventory in warehouse")

    # Extract restocking and delivery information from risk data
    if risk_data:
        # Check for delivery/lead time info
        if "lead time" in risk_data.lower() or "lead times" in risk_data.lower():
            if "3-4 week" in risk_data.lower() or "4 week" in risk_data.lower():
                insights["restocking_details"] = (
                    "Lead times are 3-4 weeks. Plan restocking accordingly."
                )
                insights["alerts"].append("Long lead times - plan ahead for restocking")
            elif "1-2 week" in risk_data.lower():
                insights["restocking_details"] = "Lead times are 1-2 weeks. Adequate restocking window."
            elif "2-3 week" in risk_data.lower():
                insights["restocking_details"] = "Lead times are 2-3 weeks. Monitor inventory closely."

        # Check for seasonal demand info
        if "seasonal" in risk_data.lower():
            if "peak" in risk_data.lower():
                insights["predicted_inventory_level"] = "LOW"
                insights["predicted_inventory_units"] = 2500
                insights["inventory_forecast"] = (
                    "Peak season approaching - inventory will decrease rapidly. Increase stock now."
                )
                insights["alerts"].append("Peak season forecast - prepare for high demand")
            else:
                insights["predicted_inventory_level"] = "HIGH"
                insights["predicted_inventory_units"] = 7500
                insights["inventory_forecast"] = (
                    "Off-season approaching - inventory levels will increase. Plan promotions."
                )
                insights["alerts"].append("Off-season forecast - prepare for inventory buildup")

        # Check for logistics/transportation constraints
        if "transport" in risk_data.lower() or "logistics" in risk_data.lower():
            if "limited" in risk_data.lower() or "constraint" in risk_data.lower():
                insights["restocking_status"] = "DELAYED"
                insights["restocking_details"] += " Transportation constraints detected."
                insights["alerts"].append("Logistics constraints - restocking may be delayed")

        # Check for distance/location
        if "distance" in risk_data.lower() or "remote" in risk_data.lower():
            insights["restocking_details"] += " Remote location affects delivery times."
            insights["alerts"].append("Remote location - factor in extended delivery times")

    # Extract compliance/quality info affecting inventory
    if compliance_data:
        if "quality" in compliance_data.lower():
            if "certification" in compliance_data.lower():
                insights["inventory_details"] += (
                    " Inventory is quality certified and regularly audited."
                )
            if "standard" in compliance_data.lower():
                insights["inventory_details"] += " All inventory meets compliance standards."

    # Determine overall logistics status
    alert_count = len(insights["alerts"])
    critical_alerts = [
        alert for alert in insights["alerts"]
        if "CRITICAL" in alert or "urgent" in alert.lower()
    ]

    if alert_count == 0:
        insights["overall_logistics_status"] = "HEALTHY"
    elif len(critical_alerts) > 0:
        insights["overall_logistics_status"] = "CRITICAL"
    elif alert_count > 3:
        insights["overall_logistics_status"] = "WARNING"
    else:
        insights["overall_logistics_status"] = "HEALTHY"

    # Set default forecast if not already set
    if not insights["inventory_forecast"]:
        current_level = insights["current_inventory_units"]
        if current_level > 7000:
            insights["inventory_forecast"] = (
                "Inventory is high. Maintain current stock levels and monitor for opportunities to reduce."
            )
        elif current_level < 3000:
            insights["inventory_forecast"] = (
                "Inventory is low. Prepare restocking orders immediately to avoid stockouts."
            )
        else:
            insights["inventory_forecast"] = (
                "Inventory levels are optimal. Continue monitoring and maintain current reorder schedule."
            )

    # Set default restocking details if not already set
    if not insights["restocking_details"]:
        insights["restocking_details"] = "Restocking schedule is on track. No issues detected."

    return insights


@logistics_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent on startup"""
    ctx.logger.info("Logistics Agent starting up...")
    ctx.logger.info(f"Agent address: {logistics_agent.address}")

    # Load logistics prompt
    ctx.storage.set("logistics_prompt", LOGISTICS_PROMPT)
    ctx.logger.info("Logistics prompt loaded!")

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info("Listening for LogisticsRequest messages...")


@logistics_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Logistics Agent shutting down...")


@logistics_protocol.on_message(model=LogisticsRequest, replies=LogisticsResponse)
async def handle_logistics_request(ctx: Context, sender: str, msg: LogisticsRequest):
    """
    Handle logistics/inventory request from Orchestrator Agent.

    Process:
    1. Receive supplier name from orchestrator
    2. Load supplier data from compliance, financial, and risk files
    3. Extract inventory and logistics insights based on logistics_prompt guidelines
    4. Return inventory monitoring update to orchestrator
    """
    ctx.logger.info(f"Received LogisticsRequest from {sender}")
    ctx.logger.info(f"   Request ID: {msg.request_id}")
    ctx.logger.info(f"   Supplier: {msg.supplier_name}")

    try:
        # Load supplier data from files
        compliance_data = load_supplier_file(msg.supplier_name, COMPLIANCE_FILES_PATH)
        risk_data = load_supplier_file(msg.supplier_name, RISK_MANAGEMENT_FILES_PATH)
        financial_data = load_supplier_file(msg.supplier_name, FINANCIAL_FILES_PATH)

        # Check if supplier files exist
        if not compliance_data and not risk_data and not financial_data:
            ctx.logger.warning(f"Supplier '{msg.supplier_name}' data files not found")

            # Return response indicating supplier not found
            response = LogisticsResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                current_inventory_level="UNKNOWN",
                current_inventory_units=0,
                inventory_details=f"No inventory data available for {msg.supplier_name}. This supplier may not be approved for monitoring yet.",
                predicted_inventory_level="UNKNOWN",
                predicted_inventory_units=0,
                inventory_forecast="No data available",
                restocking_status="UNKNOWN",
                restocking_details="No data available",
                overall_logistics_status="UNKNOWN",
                alerts=["Supplier not found in logistics system"],
                timestamp="",
            )
        else:
            # Extract logistics insights from supplier data
            insights = extract_logistics_insights(
                compliance_data or "", risk_data or "", financial_data or ""
            )

            # Build response from extracted insights
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

        ctx.logger.info("Logistics analysis complete!")

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

        ctx.logger.info(f"Sent LogisticsResponse to {sender}")

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

