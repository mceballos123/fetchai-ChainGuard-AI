from uagents import Agent, Context, Protocol
import os
import json
from typing import Dict, Any, Optional

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


def extract_demand_insights(
    compliance_data: str, risk_data: str, financial_data: str
) -> Dict[str, Any]:
    """
    Extract demand forecast insights from supplier files using the demand prompt guidelines.

    The demand_prompt directs the agent to monitor:
    - DELAYS in delivery and operations
    - QUALITY tracking and control
    - SHIPPING tracking and logistics

    Args:
        compliance_data: Contents of compliance file
        risk_data: Contents of risk management file
        financial_data: Contents of financial file

    Returns:
        Dictionary with demand forecast insights
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

    # Extract delay-related information from all data sources
    if risk_data:
        if "delay" in risk_data.lower() or "late" in risk_data.lower():
            insights["delay_status"] = "DELAYS DETECTED"
            insights["delay_details"] = (
                "Historical delays detected in risk assessment. Monitor delivery schedules closely for potential timing issues."
            )
            insights["alerts"].append("Delivery delay risk identified")

        if "seasonal" in risk_data.lower():
            if insights["delay_details"]:
                insights[
                    "delay_details"
                ] += " Seasonal fluctuations may impact delivery timelines."
            else:
                insights["delay_details"] = (
                    "Seasonal variations may affect delivery schedules. Plan ahead for peak periods."
                )
            insights["alerts"].append("Seasonal delivery variations expected")

    if financial_data:
        if "cost" in financial_data.lower() and "increase" in financial_data.lower():
            insights[
                "delay_details"
            ] += " Cost increases may impact delivery schedules and priorities."
            if "Delivery delay risk identified" not in insights["alerts"]:
                insights["delay_status"] = "MONITOR: COST IMPACT"

    # Extract shipping and logistics information
    if risk_data:
        if "transport" in risk_data.lower() or "logistics" in risk_data.lower():
            insights["shipping_status"] = "MONITOR: LOGISTICS"
            insights["shipping_details"] = (
                "Logistics factors detected in risk profile. Monitor transportation routes and carrier performance."
            )
            insights["alerts"].append("Logistics monitoring required")

        if "distance" in risk_data.lower() or "location" in risk_data.lower():
            if insights["shipping_details"]:
                insights[
                    "shipping_details"
                ] += " Geographic distance may affect shipping times and costs."
            else:
                insights["shipping_details"] = (
                    "Geographic factors may impact shipping schedules. Track transit times carefully."
                )
            insights["alerts"].append("Geographic shipping considerations")

    # Extract quality tracking information
    if compliance_data:
        if "quality" in compliance_data.lower():
            if "poor" in compliance_data.lower() or "low" in compliance_data.lower():
                insights["quality_status"] = "CONCERNS IDENTIFIED"
                insights["quality_details"] = (
                    "Quality concerns noted in compliance records. Implement enhanced quality control measures."
                )
                insights["alerts"].append("Quality control enhancement needed")
            else:
                insights["quality_status"] = "GOOD"
                insights["quality_details"] = (
                    "Quality standards maintained per compliance records. Continue regular monitoring."
                )

        if "certification" in compliance_data.lower():
            if insights["quality_details"]:
                insights[
                    "quality_details"
                ] += " Quality certifications are active and maintained."
            else:
                insights["quality_details"] = (
                    "Quality certifications verified. Standards are being met."
                )

    if risk_data:
        if "quality" in risk_data.lower() or "defect" in risk_data.lower():
            if "CONCERNS IDENTIFIED" not in insights["quality_status"]:
                insights["quality_status"] = "MONITOR: QUALITY VARIANCE"
            insights[
                "quality_details"
            ] += (
                " Historical quality variance detected. Maintain strict quality checks."
            )
            insights["alerts"].append("Quality variance monitoring required")

    # Determine overall performance based on extracted information
    alert_count = len(insights["alerts"])
    critical_issues = [
        alert
        for alert in insights["alerts"]
        if "delay" in alert.lower() or "quality control enhancement" in alert.lower()
    ]

    if alert_count == 0:
        insights["overall_performance"] = "EXCELLENT"
    elif len(critical_issues) > 0 or alert_count > 3:
        insights["overall_performance"] = "POOR"
    elif alert_count > 2:
        insights["overall_performance"] = "FAIR"
    else:
        insights["overall_performance"] = "GOOD"

    return insights


@demand_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent on startup"""
    ctx.logger.info("Demand Forecast Agent starting up...")
    ctx.logger.info(f"Agent address: {demand_agent.address}")

    # Load demand forecast prompt
    ctx.storage.set("demand_prompt", DEMAND_FORECASE_PROMPT)
    ctx.logger.info("Demand forecast prompt loaded!")

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info("Listening for DemandRequest messages...")


@demand_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Demand Forecast Agent shutting down...")


@demand_protocol.on_message(model=DemandRequest, replies=DemandResponse)
async def handle_demand_request(ctx: Context, sender: str, msg: DemandRequest):
    """
    Handle demand forecast request from Orchestrator Agent.

    Process:
    1. Receive supplier name from orchestrator
    2. Load supplier data from compliance, financial, and risk files
    3. Extract demand insights based on demand_prompt guidelines
    4. Return demand forecast update to orchestrator
    """
    ctx.logger.info(f"Received DemandRequest from {sender}")
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
            response = DemandResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                delay_status="UNKNOWN",
                delay_details=f"No demand forecast data available for {msg.supplier_name}. This supplier may not be approved for monitoring yet.",
                shipping_status="UNKNOWN",
                shipping_details="No data available",
                quality_status="UNKNOWN",
                quality_details="No data available",
                overall_performance="UNKNOWN",
                alerts=["Supplier not found in monitoring system"],
                timestamp="",
            )
        else:
            # Extract demand insights from supplier data
            insights = extract_demand_insights(
                compliance_data or "", risk_data or "", financial_data or ""
            )

            # Build response from extracted insights
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

        ctx.logger.info("Demand forecast analysis complete!")

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

        ctx.logger.info(f"Sent DemandResponse to {sender}")

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
