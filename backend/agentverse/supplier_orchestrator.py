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

# Import compliance and financial models
from backend.models.compliance import ComplianceRequest, ComplianceResponse
from backend.models.financial import FinancialRequest, FinancialResponse

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

orchestrator_protocol = Protocol(name="supplier_orchestrator_protocol", version="1.0")


@supplier_orchestrator.on_event("startup")
async def startup(ctx: Context):
    """Initialize orchestrator on startup"""
    ctx.logger.info("Supplier Orchestrator Agent Starting Up")

    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
    ctx.logger.info(f"Financial Agent: {FINANCIAL_AGENT_ADDRESS}")

    ctx.storage.set("active_sessions", {})

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

    # Step 3: Store session information
    msg_id = str(msg.msg_id)
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions[msg_id] = {"sender": sender, "query": user_query}
    ctx.storage.set("active_sessions", active_sessions)
    ctx.logger.info(f"Session stored with ID: {msg_id}")

    # Step 4: Forward to BOTH Compliance AND Financial Agents
    ctx.logger.info("=" * 70)
    ctx.logger.info("FORWARDING TO BOTH AGENTS")
    ctx.logger.info("=" * 70)

    # Initialize pending responses tracking for this request
    pending_responses = ctx.storage.get("pending_responses") or {}
    pending_responses[msg_id] = {
        "compliance_response": None,
        "financial_response": None,
        "sender": sender,
        "user_query": user_query,
    }
    ctx.storage.set("pending_responses", pending_responses)

    try:
        # Send to Compliance Agent
        ctx.logger.info(f" Sending to Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
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

        ctx.logger.info(f"✓ FinancialRequest sent to Financial Agent")
        ctx.logger.info("=" * 70)
        ctx.logger.info("WAITING FOR BOTH RESPONSES...")
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
        f" Acknowledgment received from {sender} for message: {msg.acknowledged_msg_id}"
    )


compliance_protocol = Protocol(name="compliance_response_protocol", version="1.0")


@compliance_protocol.on_message(model=ComplianceResponse)
async def handle_compliance_response(
    ctx: Context, sender: str, msg: ComplianceResponse
):
    """Handle compliance response and wait for financial response before sending to user"""
    ctx.logger.info("=" * 60)
    ctx.logger.info(" Received Compliance Response (1/2)")
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


@financial_protocol.on_message(model=FinancialResponse)
async def handle_financial_response(ctx: Context, sender: str, msg: FinancialResponse):
    """Handle financial response and wait for compliance response before sending to user"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Financial Response (2/2)")
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

        ctx.logger.info(f"✓ Financial response stored")
        ctx.logger.info(f"   Supplier: {msg.supplier_name}")
        ctx.logger.info(f"   Score: {msg.financial_score}/100")

        # Check if we have both responses now
        await check_and_send_combined_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling financial response: {e}")
        import traceback

        traceback.print_exc()


async def check_and_send_combined_response(ctx: Context, request_id: str):
    """Check if both responses are received, combine them, and send to user"""
    pending_responses = ctx.storage.get("pending_responses") or {}

    if request_id not in pending_responses:
        ctx.logger.warning(f"No pending response for request {request_id}")
        return

    response_data = pending_responses[request_id]
    compliance_response = response_data.get("compliance_response")
    financial_response = response_data.get("financial_response")

    # Check if we have BOTH responses
    if compliance_response is None or financial_response is None:
        ctx.logger.info(f" Still waiting for responses...")
        ctx.logger.info(f"   Compliance: {'✓' if compliance_response else '✗'}")
        ctx.logger.info(f"   Financial: {'✓' if financial_response else '✗'}")
        return

    # We have both responses! Combine them
    ctx.logger.info("=" * 70)
    ctx.logger.info("BOTH RESPONSES RECEIVED - COMBINING RESULTS")
    ctx.logger.info("=" * 70)

    user_sender = response_data.get("sender")
    user_query = response_data.get("user_query")

    if not user_sender:
        ctx.logger.error(f"No sender found for request {request_id}")
        return

    # Determine overall approval status
    compliance_passed = compliance_response.get("compliance_score", 0) >= 75
    financial_passed = financial_response.get("financial_score", 0) >= 70
    overall_approved = compliance_passed and financial_passed

    # Build dynamic supplier requirements status
    if compliance_passed and financial_passed:
        requirements_text = "Supplier meets both compliance and financial requirements"
    elif compliance_passed and not financial_passed:
        requirements_text = "Supplier meets compliance requirements but does not meet financial requirements"
    elif not compliance_passed and financial_passed:
        requirements_text = "Supplier meets financial requirements but does not meet compliance requirements"
    else:
        requirements_text = (
            "Supplier does not meet compliance or financial requirements"
        )

    # Build dynamic recommendation summary
    if overall_approved:
        approval_header = "APPROVED FOR PARTNERSHIP"
        recommendation_summary = f"Based on the analysis, this supplier demonstrates strong alignment with your business values and meets all required thresholds for partnership consideration."
    else:
        approval_header = "NOT APPROVED FOR PARTNERSHIP"
        if not compliance_passed and not financial_passed:
            recommendation_summary = "This supplier requires significant improvements in both compliance practices and financial stability before being considered for partnership."
        elif not compliance_passed:
            recommendation_summary = "While the financial profile is acceptable, compliance concerns must be addressed before moving forward with this partnership."
        else:
            recommendation_summary = "The compliance profile is strong, but financial risks need to be mitigated to proceed with partnership."

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

    ctx.logger.info(f"✓ Combined response sent to user")
    ctx.logger.info(
        f"   Overall Status: {'APPROVED' if overall_approved else 'NOT APPROVED'}"
    )
    ctx.logger.info(f"Compliance: {compliance_response.get('compliance_score')}/100 ")
    ctx.logger.info(f"   Financial: {financial_response.get('financial_score')}/100")

    # Clean up session and pending responses
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions.pop(request_id, None)
    ctx.storage.set("active_sessions", active_sessions)

    pending_responses.pop(request_id, None)
    ctx.storage.set("pending_responses", pending_responses)

    ctx.logger.info(f"✓ Session cleaned up for request {request_id}")
    ctx.logger.info("=" * 70)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}"
    )


supplier_orchestrator.include(chat_proto, publish_manifest=True)
supplier_orchestrator.include(compliance_protocol, publish_manifest=True)
supplier_orchestrator.include(financial_protocol, publish_manifest=True)

if __name__ == "__main__":
    supplier_orchestrator.run()
