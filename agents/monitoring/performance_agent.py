from uagents import Agent, Context, Protocol
import os
import json
from typing import Dict, Any, Optional

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
        return None


def extract_monitoring_insights(
    compliance_data: str, risk_data: str, financial_data: str
) -> Dict[str, Any]:
    """
    Extract monitoring insights from supplier files using the performance prompt guidelines.

    The performance_prompt directs the agent to monitor:
    - WEATHER conditions
    - Strikes and labor issues
    - Political environment
    - New laws and regulations

    Args:
        compliance_data: Contents of compliance file
        risk_data: Contents of risk management file
        financial_data: Contents of financial file

    Returns:
        Dictionary with monitoring insights
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

    # Extract weather-related information from risk data
    if risk_data:
        if "flood" in risk_data.lower():
            insights["weather_status"] = "RISK: FLOOD"
            insights["weather_details"] = (
                "Supplier location has flood risk history. Monitor weather alerts and seasonal conditions."
            )
            insights["alerts"].append("Flood risk detected in supplier location")
        if "drought" in risk_data.lower() or "water restriction" in risk_data.lower():
            insights["weather_status"] = "RISK: DROUGHT"
            insights["weather_details"] = (
                "Drought conditions or water restrictions possible. Monitor water availability and irrigation status."
            )
            if "RISK: FLOOD" not in insights["weather_status"]:
                insights["alerts"].append("Drought/water restriction risk identified")
        if (
            "temperature" in risk_data.lower()
            or "freeze" in risk_data.lower()
            or "heat wave" in risk_data.lower()
        ):
            insights[
                "weather_details"
            ] += " Temperature extremes possible. Monitor seasonal forecasts."
            insights["alerts"].append("Temperature extreme risk identified")

    # Extract labor/strike information
    if risk_data:
        if "labor" in risk_data.lower() or "workforce" in risk_data.lower():
            insights["strike_status"] = "MONITOR: LABOR CHANGES"
            insights["strike_details"] = (
                "Workforce fluctuations detected. Monitor seasonal staffing levels and labor availability."
            )
            insights["alerts"].append("Labor availability constraints noted")

    # Extract political/regulatory information
    if compliance_data:
        if (
            "certification" in compliance_data.lower()
            or "regulation" in compliance_data.lower()
        ):
            insights["political_details"] = (
                "Regulatory compliance requirements active. Monitor certification status and regulatory changes."
            )
            if "RISK" in compliance_data:
                insights["alerts"].append("Regulatory monitoring required")

    # Extract legal/regulatory from risk data
    if risk_data:
        if (
            "permit" in risk_data.lower()
            or "compliance" in risk_data.lower()
            or "fsma" in risk_data.lower()
        ):
            insights["legal_status"] = "MONITOR: PERMITS/COMPLIANCE"
            insights["legal_details"] = (
                "Active permit reviews and compliance audits in progress. Monitor regulatory changes and permit renewals."
            )
            insights["alerts"].append("Permit and regulatory compliance review ongoing")

    # Determine overall risk level based on extracted information
    alert_count = len(insights["alerts"])
    if alert_count == 0:
        insights["overall_risk"] = "LOW"
    elif alert_count <= 2:
        insights["overall_risk"] = "MEDIUM"
    elif alert_count <= 4:
        insights["overall_risk"] = "HIGH"
    else:
        insights["overall_risk"] = "CRITICAL"

    return insights


@performance_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent on startup"""
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


@performance_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    pass


@performance_protocol.on_message(model=PerformanceRequest, replies=PerformanceResponse)
async def handle_performance_request(
    ctx: Context, sender: str, msg: PerformanceRequest
):
    """
    Handle performance monitoring request from Orchestrator Agent.

    Process:
    1. Receive supplier name from orchestrator
    2. Load supplier data from compliance, financial, and risk files
    3. Extract monitoring insights based on performance_prompt guidelines
    4. Return performance update to orchestrator
    """
    try:
        # Load supplier data from files
        compliance_data = load_supplier_file(msg.supplier_name, COMPLIANCE_FILES_PATH)
        risk_data = load_supplier_file(msg.supplier_name, RISK_MANAGEMENT_FILES_PATH)
        financial_data = load_supplier_file(msg.supplier_name, FINANCIAL_FILES_PATH)

        # Check if supplier files exist
        if not compliance_data and not risk_data and not financial_data:

            # Return response indicating supplier not found
            response = PerformanceResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                weather_status="UNKNOWN",
                weather_details=f"No monitoring data available for {msg.supplier_name}. This supplier may not be approved for monitoring yet.",
                strike_status="UNKNOWN",
                strike_details="No data available",
                political_status="UNKNOWN",
                political_details="No data available",
                legal_status="UNKNOWN",
                legal_details="No data available",
                overall_risk_level="UNKNOWN",
                alerts=["Supplier not found in monitoring system"],
                timestamp="",
            )
        else:
            # Extract monitoring insights from supplier data
            insights = extract_monitoring_insights(
                compliance_data or "", risk_data or "", financial_data or ""
            )

            # Build response from extracted insights
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
