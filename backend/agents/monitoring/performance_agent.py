from uagents import Agent, Context, Protocol
import os
import json

# Import models
from backend.models.performance import PerformanceRequest, PerformanceResponse
from dotenv import load_dotenv

load_dotenv()

performance_agent = Agent(
    name="performance_agent",
    seed=os.getenv("PERFORMANCE_AGENT_SEED"),
    port=8004,
    mailbox=True,
)

performance_protocol = Protocol(name="performance_protocol", version="1.0")

# Load mock monitoring data
MONITORING_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "../../data/supplier_monitoring_data.json"
)


@performance_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent on startup"""
    ctx.logger.info("Performance Monitoring Agent starting up...")
    ctx.logger.info(f"Agent address: {performance_agent.address}")

    # Load monitoring data
    try:
        with open(MONITORING_DATA_PATH, "r") as f:
            monitoring_data = json.load(f)
            ctx.storage.set("monitoring_data", monitoring_data)
            ctx.logger.info("Monitoring data loaded successfully!")
    except Exception as e:
        ctx.logger.error(f"Error loading monitoring data: {e}")
        ctx.storage.set("monitoring_data", {"suppliers": {}})

    # Initialize update counter for each supplier
    ctx.storage.set("update_counters", {})

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info("Listening for PerformanceRequest messages...")


@performance_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Performance Monitoring Agent shutting down...")


@performance_protocol.on_message(model=PerformanceRequest, replies=PerformanceResponse)
async def handle_performance_request(
    ctx: Context, sender: str, msg: PerformanceRequest
):
    """
    Handle performance monitoring request from Orchestrator Agent.

    Process:
    1. Receive supplier name
    2. Look up monitoring data from mock database
    3. Rotate through different updates each time
    4. Return performance update to orchestrator
    """
    ctx.logger.info(f"Received PerformanceRequest from {sender}")
    ctx.logger.info(f"   Request ID: {msg.request_id}")
    ctx.logger.info(f"   Supplier: {msg.supplier_name}")

    try:
        # Get monitoring data
        monitoring_data = ctx.storage.get("monitoring_data") or {"suppliers": {}}
        suppliers = monitoring_data.get("suppliers", {})

        # Check if supplier exists in our monitoring database
        if msg.supplier_name not in suppliers:
            ctx.logger.warning(
                f"Supplier '{msg.supplier_name}' not found in monitoring database"
            )

            # Return response indicating supplier not being monitored
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
            # Get supplier monitoring data
            supplier_data = suppliers[msg.supplier_name]
            updates = supplier_data.get("monitoring_updates", [])

            if not updates:
                ctx.logger.warning(
                    f"No monitoring updates available for {msg.supplier_name}"
                )
                response = PerformanceResponse(
                    request_id=msg.request_id,
                    supplier_name=msg.supplier_name,
                    weather_status="UNKNOWN",
                    weather_details="No monitoring updates available",
                    strike_status="UNKNOWN",
                    strike_details="No data available",
                    political_status="UNKNOWN",
                    political_details="No data available",
                    legal_status="UNKNOWN",
                    legal_details="No data available",
                    overall_risk_level="UNKNOWN",
                    alerts=["No updates available"],
                    timestamp="",
                )
            else:
                # Get update counter for this supplier
                update_counters = ctx.storage.get("update_counters") or {}
                current_index = update_counters.get(msg.supplier_name, 0)

                # Get the current update (rotate through available updates)
                update = updates[current_index % len(updates)]

                # Increment counter for next request
                update_counters[msg.supplier_name] = (current_index + 1) % len(updates)
                ctx.storage.set("update_counters", update_counters)

                ctx.logger.info(
                    f"   Location: {supplier_data.get('location', 'Unknown')}"
                )
                ctx.logger.info(f"   Using update #{update['update_number']}")

                # Build response from update data
                response = PerformanceResponse(
                    request_id=msg.request_id,
                    supplier_name=msg.supplier_name,
                    weather_status=update["weather"]["status"],
                    weather_details=update["weather"]["details"],
                    strike_status=update["strikes"]["status"],
                    strike_details=update["strikes"]["details"],
                    political_status=update["political"]["status"],
                    political_details=update["political"]["details"],
                    legal_status=update["legal"]["status"],
                    legal_details=update["legal"]["details"],
                    overall_risk_level=update["overall_risk"],
                    alerts=update.get("alerts", []),
                    timestamp="",
                )

        ctx.logger.info("Performance monitoring analysis complete!")
        ctx.logger.info(f"Overall Risk Level: {response.overall_risk_level}")
        ctx.logger.info(f"Active Alerts: {len(response.alerts)}")

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

        ctx.logger.info(f"Sent PerformanceResponse to {sender}")

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
