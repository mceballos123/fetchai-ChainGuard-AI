from uuid import uuid4
from uagents import Agent, Context, Protocol
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Dict, List, Any

# Import chat protocol components
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
    chat_protocol_spec,
)

# Import compliance, financial, risk, performance, and demand models
from backend.models.compliance import ComplianceRequest, ComplianceResponse
from backend.models.financial import FinancialRequest, FinancialResponse
from backend.models.risk import RiskRequest, RiskResponse
from backend.models.performance import PerformanceRequest, PerformanceResponse
from backend.models.demand import DemandRequest, DemandResponse

# Import test utilities
from backend.test_func import (
    verify_orchestrator_connection,
    log_message_transmission,
    validate_request_before_sending,
)

load_dotenv()

SUPPLIER_ORCHESTRATOR_SEED = os.getenv("SUPPLIER_ORCHESTRATOR_SEED")

supplier_orchestrator = Agent(
    name="supplier_orchestrator",
    seed=SUPPLIER_ORCHESTRATOR_SEED,
    port=8000,
    mailbox=True,
)

chat_proto = Protocol(name="chat_protocol", spec=chat_protocol_spec)

COMPLIANCE_AGENT_ADDRESS = os.getenv(
    "COMPLIANCE_AGENT_ADDRESS",
)
FINANCIAL_AGENT_ADDRESS = os.getenv(
    "FINANCIAL_AGENT_ADDRESS",
)
RISK_AGENT_ADDRESS = os.getenv(
    "RISK_AGENT_ADDRESS",
)
PERFORMANCE_AGENT_ADDRESS = os.getenv(
    "PERFORMANCE_AGENT_ADDRESS",
)
DEMAND_AGENT_ADDRESS = os.getenv(
    "DEMAND_AGENT_ADDRESS",
)

orchestrator_protocol = Protocol(name="supplier_orchestrator_protocol", version="1.0")


@supplier_orchestrator.on_event("startup")
async def startup(ctx: Context):
    """Initialize orchestrator on startup"""
    ctx.logger.info("Supplier Orchestrator Agent Starting Up")

    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
    ctx.logger.info(f"Financial Agent: {FINANCIAL_AGENT_ADDRESS}")
    ctx.logger.info(f"Risk Agent: {RISK_AGENT_ADDRESS}")
    ctx.logger.info(f"Performance Agent: {PERFORMANCE_AGENT_ADDRESS}")
    ctx.logger.info(f"Demand Agent: {DEMAND_AGENT_ADDRESS}")

    ctx.storage.set("active_sessions", {})
    ctx.storage.set("approved_suppliers", [])
    ctx.storage.set(
        "selected_supplier", None
    )  # Track the supplier selected for monitoring

    # Storage for tracking responses from both agents
    ctx.storage.set("pending_responses", {})

    # Initialize message trace for debugging
    ctx.storage.set(
        "message_trace",
        {
            "received_from_asi": [],
            "sent_to_compliance": [],
            "received_from_compliance": [],
            "sent_to_asi": [],
        },
    )

    # Verify connection on startup
    conn_status = verify_orchestrator_connection(ctx)
    ctx.logger.info(f"Connection Status: {conn_status['status']}")


@supplier_orchestrator.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Supplier Orchestrator Agent Shutting Down")


@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle chat messages from user via ASI:1"""

    # Step 0: Verify and log message received from ASI:1
    log_message_transmission(
        ctx, "RECEIVED", "ChatMessage", str(msg.msg_id), {"sender": sender}
    )

    # Step 1: Send acknowledgment immediately
    ctx.logger.info("Sending acknowledgment...")
    ack = ChatAcknowledgement(
        timestamp="",
        acknowledged_msg_id=msg.msg_id,
    )
    await ctx.send(sender, ack)

    # Step 2: Extract user query from message content
    ctx.logger.info(f"Received ChatMessage from {sender}")
    user_query = None

    for content_item in msg.content:
        if isinstance(content_item, TextContent):
            user_query = content_item.text
            break

    if not user_query:
        ctx.logger.warning("No text content in message")
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text="Error: No query text provided in message",
                )
            ],
        )
        await ctx.send(sender, error_response)
        log_message_transmission(
            ctx, "SENT", "ChatMessage", str(error_response.msg_id), {"type": "error"}
        )
        return

    ctx.logger.info(f"User Query: {user_query}")

    # Step 3: Determine if this is a monitoring request or find supplier request
    user_query_lower = user_query.lower()
    monitoring_keywords = [
        "monitor",
        "update",
        "status",
        "check on",
        "check up",
        "how is",
        "report on",
    ]
    is_monitoring_request = any(
        keyword in user_query_lower for keyword in monitoring_keywords
    )

    # Step 4: Store session information
    msg_id = str(msg.msg_id)
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions[msg_id] = {
        "sender": sender,
        "query": user_query,
        "mode": "monitor" if is_monitoring_request else "find",
    }
    ctx.storage.set("active_sessions", active_sessions)
    ctx.logger.info(f"Session stored with ID: {msg_id}")
    ctx.logger.info(
        f"Mode detected: {'MONITORING' if is_monitoring_request else 'FIND SUPPLIER'}"
    )

    # Step 5: Route based on mode
    if is_monitoring_request:
        # MONITORING MODE: Forward to Performance and Demand Agents
        ctx.logger.info("=" * 70)
        ctx.logger.info("MONITORING MODE - FORWARDING TO PERFORMANCE AND DEMAND AGENTS")
        ctx.logger.info("=" * 70)

        # Get the selected supplier (from the find_supplier phase)
        selected_supplier = ctx.storage.get("selected_supplier")

        if not selected_supplier:
            ctx.logger.warning("No selected supplier available for monitoring")
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text="Error: No supplier selected for monitoring. Please run the find_supplier feature first to select a supplier.",
                    )
                ],
            )
            await ctx.send(sender, error_response)
            log_message_transmission(
                ctx,
                "SENT",
                "ChatMessage",
                str(error_response.msg_id),
                {"type": "error"},
            )
            return

        # Initialize pending responses for monitoring (both performance and demand)
        pending_responses = ctx.storage.get("pending_responses") or {}
        pending_responses[msg_id] = {
            "performance_response": None,
            "demand_response": None,
            "sender": sender,
            "user_query": user_query,
            "mode": "monitor",
        }
        ctx.storage.set("pending_responses", pending_responses)

        try:
            # Send to Performance Agent with the selected supplier
            ctx.logger.info(
                f"Sending to Performance Agent: {PERFORMANCE_AGENT_ADDRESS}"
            )
            ctx.logger.info(f"Monitoring supplier: {selected_supplier}")
            performance_request = PerformanceRequest(
                request_id=msg_id,
                supplier_name=selected_supplier,
                timestamp="",
            )

            await ctx.send(PERFORMANCE_AGENT_ADDRESS, performance_request)

            ctx.logger.info(f"PerformanceRequest sent to Performance Agent")

            # Send to Demand Agent with the selected supplier
            ctx.logger.info(
                f"Sending to Demand Agent: {DEMAND_AGENT_ADDRESS}"
            )
            ctx.logger.info(f"Monitoring supplier: {selected_supplier}")
            demand_request = DemandRequest(
                request_id=msg_id,
                supplier_name=selected_supplier,
                timestamp="",
            )

            await ctx.send(DEMAND_AGENT_ADDRESS, demand_request)

            ctx.logger.info(f"DemandRequest sent to Demand Agent")
            ctx.logger.info("=" * 70)
            ctx.logger.info("WAITING FOR PERFORMANCE AND DEMAND RESPONSES...")
            ctx.logger.info("=" * 70)

        except Exception as e:
            ctx.logger.error(f"Error sending to monitoring agents: {e}")

            # Clean up pending responses
            pending_responses.pop(msg_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            # Send error response to user
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Error forwarding monitoring request: {str(e)}",
                    )
                ],
            )
            await ctx.send(sender, error_response)
            log_message_transmission(
                ctx,
                "SENT",
                "ChatMessage",
                str(error_response.msg_id),
                {"type": "error"},
            )

        return

    # Step 6: FIND SUPPLIER MODE - Forward to ALL THREE Agents (Compliance, Financial, Risk)
    ctx.logger.info("=" * 70)
    ctx.logger.info("FORWARDING TO ALL THREE AGENTS")
    ctx.logger.info("=" * 70)

    # Initialize pending responses tracking for this request
    pending_responses = ctx.storage.get("pending_responses") or {}
    pending_responses[msg_id] = {
        "compliance_response": None,
        "financial_response": None,
        "risk_response": None,
        "sender": sender,
        "user_query": user_query,
    }
    ctx.storage.set("pending_responses", pending_responses)

    # Store the user query as the supplier being evaluated (will be updated once selected)
    ctx.storage.set("current_evaluation_supplier", user_query)

    try:
        # Send to Compliance Agent
        ctx.logger.info(f"Sending to Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
        compliance_request = ComplianceRequest(
            request_id=msg_id,
            supplier_name=user_query,
            industry="general",
            company_values=user_query,
            timestamp="",
        )

        # Validate request before sending
        is_valid, error_msg = validate_request_before_sending(ctx, compliance_request)
        if not is_valid:
            raise ValueError(f"Compliance request validation failed: {error_msg}")

        await ctx.send(COMPLIANCE_AGENT_ADDRESS, compliance_request)

        # Log transmission
        log_message_transmission(
            ctx,
            "SENT",
            "ComplianceRequest",
            msg_id,
            {
                "supplier_name": compliance_request.supplier_name,
                "industry": compliance_request.industry,
            },
        )

        ctx.logger.info(f"ComplianceRequest sent to Compliance Agent")

        # Send to Financial Agent
        ctx.logger.info(f"Sending to Financial Agent: {FINANCIAL_AGENT_ADDRESS}")
        financial_request = FinancialRequest(
            request_id=msg_id,
            supplier_name=user_query,
            industry="general",
            timestamp="",
        )

        await ctx.send(FINANCIAL_AGENT_ADDRESS, financial_request)

        # Log transmission
        log_message_transmission(
            ctx,
            "SENT",
            "FinancialRequest",
            msg_id,
            {
                "supplier_name": financial_request.supplier_name,
                "industry": financial_request.industry,
            },
        )

        ctx.logger.info(f"FinancialRequest sent to Financial Agent")

        # Send to Risk Management Agent
        ctx.logger.info(f"Sending to Risk Agent: {RISK_AGENT_ADDRESS}")
        risk_request = RiskRequest(
            request_id=msg_id,
            supplier_name=user_query,
            industry="general",
            timestamp="",
        )

        await ctx.send(RISK_AGENT_ADDRESS, risk_request)

        # Log transmission
        log_message_transmission(
            ctx,
            "SENT",
            "RiskRequest",
            msg_id,
            {
                "supplier_name": risk_request.supplier_name,
                "industry": risk_request.industry,
            },
        )

        ctx.logger.info(f"RiskRequest sent to Risk Agent")
        ctx.logger.info("=" * 70)
        ctx.logger.info("WAITING FOR ALL THREE RESPONSES...")
        ctx.logger.info("=" * 70)

    except Exception as e:
        ctx.logger.error(f"Error sending to agents: {e}")

        # Clean up pending responses
        pending_responses.pop(msg_id, None)
        ctx.storage.set("pending_responses", pending_responses)

        # Send error response to user
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text=f"Error forwarding request to agents: {str(e)}",
                )
            ],
        )
        await ctx.send(sender, error_response)
        log_message_transmission(
            ctx, "SENT", "ChatMessage", str(error_response.msg_id), {"type": "error"}
        )


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgment messages from user"""
    ctx.logger.info(
        f"Acknowledgment received from {sender} for message: {msg.acknowledged_msg_id}"
    )


compliance_protocol = Protocol(name="compliance_response_protocol", version="1.0")


@compliance_protocol.on_message(model=ComplianceResponse)
async def handle_compliance_response(
    ctx: Context, sender: str, msg: ComplianceResponse
):
    """Handle compliance response and wait for other responses before sending to user"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Compliance Response (1/3)")
    ctx.logger.info("=" * 60)

    # Verify message received from Compliance Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "ComplianceResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "compliance_score": msg.compliance_score,
        },
    )

    try:
        # Store compliance response in pending_responses
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["compliance_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Compliance response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Score: {msg.compliance_score}/100")

        # Check if we have both responses now
        await check_and_send_combined_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling compliance response: {e}")
        import traceback

        traceback.print_exc()


# Add Financial Response Protocol
financial_protocol = Protocol(name="financial_response_protocol", version="1.0")

# Add Risk Response Protocol
risk_protocol = Protocol(name="risk_response_protocol", version="1.0")

# Add Performance Response Protocol
performance_protocol = Protocol(name="performance_response_protocol", version="1.0")

# Add Demand Response Protocol
demand_protocol = Protocol(name="demand_response_protocol", version="1.0")


@financial_protocol.on_message(model=FinancialResponse)
async def handle_financial_response(ctx: Context, sender: str, msg: FinancialResponse):
    """Handle financial response and wait for other responses before sending to user"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Financial Response (2/3)")
    ctx.logger.info("=" * 60)

    # Verify message received from Financial Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "FinancialResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "financial_score": msg.financial_score,
        },
    )

    try:
        # Store financial response in pending_responses
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["financial_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Financial response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Score: {msg.financial_score}/100")

        # Check if we have both responses now
        await check_and_send_combined_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling financial response: {e}")
        import traceback

        traceback.print_exc()


@risk_protocol.on_message(model=RiskResponse)
async def handle_risk_response(ctx: Context, sender: str, msg: RiskResponse):
    """Handle risk response and wait for other responses before sending to user"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Risk Response (3/3)")
    ctx.logger.info("=" * 60)

    # Verify message received from Risk Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "RiskResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "risk_score": msg.risk_score,
        },
    )

    try:
        # Store risk response in pending_responses
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["risk_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Risk response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Score: {msg.risk_score}/100")

        # Check if we have all responses now
        await check_and_send_combined_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling risk response: {e}")
        import traceback

        traceback.print_exc()


@demand_protocol.on_message(model=DemandResponse)
async def handle_demand_response(ctx: Context, sender: str, msg: DemandResponse):
    """Handle demand forecast response and wait for performance response"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Demand Forecast Response (1/2)")
    ctx.logger.info("=" * 60)

    # Verify message received from Demand Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "DemandResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "overall_performance": msg.overall_performance,
        },
    )

    try:
        # Store demand response in pending_responses
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["demand_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Demand response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Overall Performance: {msg.overall_performance}")

        # Check if we have both monitoring responses now
        await check_and_send_monitoring_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling demand response: {e}")
        import traceback

        traceback.print_exc()


@performance_protocol.on_message(model=PerformanceResponse)
async def handle_performance_response(
    ctx: Context, sender: str, msg: PerformanceResponse
):
    """Handle performance monitoring response and wait for demand response"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Performance Monitoring Response (2/2)")
    ctx.logger.info("=" * 60)

    # Verify message received from Performance Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "PerformanceResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "overall_risk_level": msg.overall_risk_level,
        },
    )

    try:
        # Store performance response in pending_responses
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["performance_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Performance response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Overall Risk: {msg.overall_risk_level}")
        ctx.logger.info(f"Alerts: {len(msg.alerts)}")

        # Check if we have both monitoring responses now
        await check_and_send_monitoring_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling performance response: {e}")
        import traceback

        traceback.print_exc()


async def check_and_send_monitoring_response(ctx: Context, request_id: str):
    """Check if both monitoring responses are received, combine them, and send to user"""
    pending_responses = ctx.storage.get("pending_responses") or {}

    if request_id not in pending_responses:
        ctx.logger.warning(f"No pending response for request {request_id}")
        return

    response_data = pending_responses[request_id]
    performance_response = response_data.get("performance_response")
    demand_response = response_data.get("demand_response")

    # Check if we have BOTH monitoring responses
    if performance_response is None or demand_response is None:
        ctx.logger.info(f"Still waiting for monitoring responses...")
        ctx.logger.info(
            f"Performance: {'RECEIVED' if performance_response else 'PENDING'}"
        )
        ctx.logger.info(f"Demand: {'RECEIVED' if demand_response else 'PENDING'}")
        return

    # We have both responses! Combine them
    ctx.logger.info("=" * 70)
    ctx.logger.info("ALL MONITORING RESPONSES RECEIVED - COMBINING RESULTS")
    ctx.logger.info("=" * 70)

    user_sender = response_data.get("sender")

    if not user_sender:
        ctx.logger.error(f"No sender found for request {request_id}")
        return

    # Build combined monitoring report
    performance_alerts_text = ""
    if performance_response.get("alerts"):
        performance_alerts_text = "\n".join(
            [f"  • {alert}" for alert in performance_response.get("alerts", [])]
        )
    else:
        performance_alerts_text = "  • No active alerts"

    demand_alerts_text = ""
    if demand_response.get("alerts"):
        demand_alerts_text = "\n".join(
            [f"  • {alert}" for alert in demand_response.get("alerts", [])]
        )
    else:
        demand_alerts_text = "  • No active alerts"

    response_text = f"""
SUPPLIER MONITORING UPDATE

Supplier: {performance_response.get('supplier_name', 'N/A')}
Overall Risk Level: {performance_response.get('overall_risk_level', 'N/A')}
Overall Performance: {demand_response.get('overall_performance', 'N/A')}

===== EXTERNAL FACTORS (PERFORMANCE MONITORING) =====

WEATHER CONDITIONS
Status: {performance_response.get('weather_status', 'N/A')}
{performance_response.get('weather_details', 'N/A')}

---

LABOR & STRIKES
Status: {performance_response.get('strike_status', 'N/A')}
{performance_response.get('strike_details', 'N/A')}

---

POLITICAL ENVIRONMENT
Status: {performance_response.get('political_status', 'N/A')}
{performance_response.get('political_details', 'N/A')}

---

LEGAL & REGULATORY
Status: {performance_response.get('legal_status', 'N/A')}
{performance_response.get('legal_details', 'N/A')}

---

PERFORMANCE ALERTS ({len(performance_response.get('alerts', []))})
{performance_alerts_text}

===== OPERATIONAL METRICS (DEMAND FORECAST) =====

DELIVERY & DELAYS
Status: {demand_response.get('delay_status', 'N/A')}
{demand_response.get('delay_details', 'N/A')}

---

SHIPPING & LOGISTICS
Status: {demand_response.get('shipping_status', 'N/A')}
{demand_response.get('shipping_details', 'N/A')}

---

QUALITY TRACKING
Status: {demand_response.get('quality_status', 'N/A')}
{demand_response.get('quality_details', 'N/A')}

---

DEMAND ALERTS ({len(demand_response.get('alerts', []))})
{demand_alerts_text}

---

This comprehensive monitoring update combines external factors and operational metrics to give you a complete view of your supplier's performance. Request another update anytime to see the latest conditions.
    """

    # Send combined monitoring response back to user
    ctx.logger.info(f"Sending combined monitoring update to user: {user_sender}")

    response = ChatMessage(
        timestamp="",
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=response_text.strip()),
            EndSessionContent(type="end-session"),
        ],
    )

    await ctx.send(user_sender, response)

    log_message_transmission(
        ctx,
        "SENT",
        "ChatMessage",
        str(response.msg_id),
        {
            "type": "monitoring_update",
            "risk_level": performance_response.get("overall_risk_level"),
            "performance": demand_response.get("overall_performance"),
        },
    )

    ctx.logger.info(f"Combined monitoring update sent to user")
    ctx.logger.info(
        f"Risk Level: {performance_response.get('overall_risk_level', 'N/A')}"
    )
    ctx.logger.info(
        f"Performance: {demand_response.get('overall_performance', 'N/A')}"
    )

    # Clean up session and pending responses
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions.pop(request_id, None)
    ctx.storage.set("active_sessions", active_sessions)

    pending_responses.pop(request_id, None)
    ctx.storage.set("pending_responses", pending_responses)

    ctx.logger.info(f"Session cleaned up for request {request_id}")
    ctx.logger.info("=" * 70)


async def check_and_send_combined_response(ctx: Context, request_id: str):
    """Check if all three responses are received, combine them, and send to user"""
    pending_responses = ctx.storage.get("pending_responses") or {}

    if request_id not in pending_responses:
        ctx.logger.warning(f"No pending response for request {request_id}")
        return

    response_data = pending_responses[request_id]
    compliance_response = response_data.get("compliance_response")
    financial_response = response_data.get("financial_response")
    risk_response = response_data.get("risk_response")

    # Check if we have ALL THREE responses
    if (
        compliance_response is None
        or financial_response is None
        or risk_response is None
    ):
        ctx.logger.info(f"Still waiting for responses...")
        ctx.logger.info(
            f"Compliance: {'RECEIVED' if compliance_response else 'PENDING'}"
        )
        ctx.logger.info(f"Financial: {'RECEIVED' if financial_response else 'PENDING'}")
        ctx.logger.info(f"Risk: {'RECEIVED' if risk_response else 'PENDING'}")
        return

    # We have all three responses! Combine them
    ctx.logger.info("=" * 70)
    ctx.logger.info("ALL THREE RESPONSES RECEIVED - COMBINING RESULTS")
    ctx.logger.info("=" * 70)

    user_sender = response_data.get("sender")
    user_query = response_data.get("user_query")

    if not user_sender:
        ctx.logger.error(f"No sender found for request {request_id}")
        return

    # Determine overall approval status - ALL THREE must pass
    compliance_passed = compliance_response.get("compliance_score", 0) >= 75
    financial_passed = financial_response.get("financial_score", 0) >= 70
    risk_passed = risk_response.get("risk_score", 0) >= 75
    overall_approved = compliance_passed and financial_passed and risk_passed

    # Build dynamic supplier requirements status
    passed_requirements = []
    failed_requirements = []

    if compliance_passed:
        passed_requirements.append("compliance")
    else:
        failed_requirements.append("compliance")

    if financial_passed:
        passed_requirements.append("financial")
    else:
        failed_requirements.append("financial")

    if risk_passed:
        passed_requirements.append("risk management")
    else:
        failed_requirements.append("risk management")

    if overall_approved:
        requirements_text = (
            f"Supplier meets all requirements: {', '.join(passed_requirements)}"
        )
        # Store the approved supplier for monitoring
        ctx.storage.set(
            "selected_supplier", compliance_response.get("supplier_name", "")
        )
        ctx.logger.info(
            f"Approved supplier stored for monitoring: {compliance_response.get('supplier_name', '')}"
        )
    elif len(passed_requirements) > 0:
        requirements_text = f"Supplier meets {', '.join(passed_requirements)} but does not meet {', '.join(failed_requirements)}"
    else:
        requirements_text = "Supplier does not meet any requirements"

    # Build dynamic recommendation summary
    if overall_approved:
        approval_header = "APPROVED FOR PARTNERSHIP"
        recommendation_summary = f"Based on comprehensive analysis across compliance, financial, and risk management factors, this supplier demonstrates strong alignment with your business values and meets all required thresholds for partnership consideration."
    else:
        approval_header = "NOT APPROVED FOR PARTNERSHIP"
        if len(failed_requirements) == 3:
            recommendation_summary = "This supplier requires significant improvements across all evaluation areas (compliance, financial, and risk management) before being considered for partnership."
        elif len(failed_requirements) == 2:
            recommendation_summary = f"While {passed_requirements[0]} metrics are acceptable, this supplier must address concerns in {' and '.join(failed_requirements)} before moving forward with partnership."
        else:
            failed_area = failed_requirements[0]
            recommendation_summary = f"The supplier shows strength in {' and '.join(passed_requirements)}, but {failed_area} concerns must be mitigated to proceed with partnership."

    # Format financial details cleanly without bullet points
    financial_details = financial_response.get("financial_details", "N/A")

    # Format risk factors - include even small ones without being nice
    risk_factors = financial_response.get("risk_factors", [])
    if risk_factors:
        risk_factors_text = "\n".join([f"  • {factor}" for factor in risk_factors])
    else:
        risk_factors_text = "  • No significant financial risks identified"

    # Format violations - include even if minimal
    violations = compliance_response.get("violations", [])
    violations_count = len(violations)
    if violations:
        violations_text = "\n".join([f"  • {v}" for v in violations])
    else:
        violations_text = "  • None identified"

    # Format risk management details
    risk_details = risk_response.get("risk_details", "N/A")

    # Format risk management factors
    risk_factors_list = risk_response.get("risk_factors", [])
    if risk_factors_list:
        risk_mgmt_factors_text = "\n".join(
            [f"  • {factor}" for factor in risk_factors_list]
        )
    else:
        risk_mgmt_factors_text = "  • No significant risk factors identified"

    response_text = f"""
{approval_header}

SUPPLIER ANALYSIS REPORT

Business Owner: {compliance_response.get('supplier_name', 'N/A')}
Supplier Status: {'APPROVED' if overall_approved else 'NOT APPROVED'}

Supplier Requirements: {requirements_text}

COMPLIANCE ANALYSIS

Overall Compliance Score: {compliance_response.get('compliance_score')}/100

Ethics & Worker Treatment:
{compliance_response.get('ethics_info', 'N/A')}

Sustainability Practices:
{compliance_response.get('sustainability_info', 'N/A')}

Violations Found: {violations_count}
{violations_text}

FINANCIAL RISK ANALYSIS

Financial Risk Score: {financial_response.get('financial_score')}/100 (Higher = Lower Risk)

Operating & Cost Assessment:
{financial_details}

Risk Factors Identified: {len(risk_factors)}
{risk_factors_text}

RISK MANAGEMENT ANALYSIS

Risk Management Score: {risk_response.get('risk_score')}/100 (Higher = Lower Risk)

Risk Assessment:
{risk_details}

Identified Risk Factors: {len(risk_factors_list)}
{risk_mgmt_factors_text}

RECOMMENDATION

{recommendation_summary}
    """

    # Send combined response back to user via chat
    ctx.logger.info(f"Sending combined response to user: {user_sender}")

    response = ChatMessage(
        timestamp="",
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=response_text.strip()),
            EndSessionContent(type="end-session"),
        ],
    )

    await ctx.send(user_sender, response)

    log_message_transmission(
        ctx,
        "SENT",
        "ChatMessage",
        str(response.msg_id),
        {"type": "combined_result", "approved": overall_approved},
    )

    ctx.logger.info(f"Combined response sent to user")
    ctx.logger.info(
        f"Overall Status: {'APPROVED' if overall_approved else 'NOT APPROVED'}"
    )
    ctx.logger.info(f"Compliance: {compliance_response.get('compliance_score')}/100")
    ctx.logger.info(f"Financial: {financial_response.get('financial_score')}/100")
    ctx.logger.info(f"Risk: {risk_response.get('risk_score')}/100")

    # Clean up session and pending responses
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions.pop(request_id, None)
    ctx.storage.set("active_sessions", active_sessions)

    pending_responses.pop(request_id, None)
    ctx.storage.set("pending_responses", pending_responses)

    ctx.logger.info(f"Session cleaned up for request {request_id}")
    ctx.logger.info("=" * 70)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}"
    )


supplier_orchestrator.include(chat_proto, publish_manifest=True)
supplier_orchestrator.include(compliance_protocol, publish_manifest=True)
supplier_orchestrator.include(financial_protocol, publish_manifest=True)
supplier_orchestrator.include(risk_protocol, publish_manifest=True)
supplier_orchestrator.include(performance_protocol, publish_manifest=True)
supplier_orchestrator.include(demand_protocol, publish_manifest=True)

if __name__ == "__main__":
    supplier_orchestrator.run()
