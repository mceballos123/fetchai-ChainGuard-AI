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

# Import compliance models
from backend.models.compliance import ComplianceRequest, ComplianceResponse

load_dotenv()

SUPPLIER_ORCHESTRATOR_SEED = os.getenv("SUPPLIER_ORCHESTRATOR_SEED")

supplier_orchestrator = Agent(
    name="supplier_orchestrator",
    seed=SUPPLIER_ORCHESTRATOR_SEED,
    port=8000,
    mailbox=True,  # =
)

chat_proto = Protocol(name="chat_protocol", spec=chat_protocol_spec)

COMPLIANCE_AGENT_ADDRESS = os.getenv(
    "COMPLIANCE_AGENT_ADDRESS",
)

orchestrator_protocol = Protocol(name="supplier_orchestrator_protocol", version="1.0")


@supplier_orchestrator.on_event("startup")
async def startup(ctx: Context):
    """Initialize orchestrator on startup"""
    ctx.logger.info("Supplier Orchestrator Agent Starting Up")

    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")

    ctx.storage.set("active_sessions", {})

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


@supplier_orchestrator.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Supplier Orchestrator Agent Shutting Down")

@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle chat messages from user via ASI:1"""

    # Step 0: Verify and log message received from ASI:1
    
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
        return

    ctx.logger.info(f"User Query: {user_query}")

    # Step 3: Store session information
    msg_id = str(msg.msg_id)
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions[msg_id] = {"sender": sender, "query": user_query}
    ctx.storage.set("active_sessions", active_sessions)
    ctx.logger.info(f"Session stored with ID: {msg_id}")

    # Step 4: Forward to Compliance Agent
    ctx.logger.info(f"Forwarding to Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")

    try:
        compliance_request = ComplianceRequest(
            request_id=msg_id,
            supplier_name=user_query,
            industry="general",
            company_values=user_query,
            timestamp="",
        )

        await ctx.send(COMPLIANCE_AGENT_ADDRESS, compliance_request)
        
        ctx.logger.info(f"ComplianceRequest sent to Compliance Agent")

    except Exception as e:
        ctx.logger.error(f"Error sending to Compliance Agent: {e}")

        # Send error response to user
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text=f"Error forwarding request to Compliance Agent: {str(e)}",
                )
            ],
        )
        await ctx.send(sender, error_response)


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
    # Handles response from the compliance agent
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Compliance Response")
    ctx.logger.info("=" * 60)

    # Verify message received from Compliance Agent
   

    try:
        # Retrieve session information
        active_sessions = ctx.storage.get("active_sessions") or {}
        user_sender = active_sessions.get(msg.request_id, {}).get("sender")

        if not user_sender:
            ctx.logger.warning(
                f"Could not find session for request {msg.request_id}"
            )
            return

        ctx.logger.info(f"Found session for request {msg.request_id}")
        ctx.logger.info(f"Original sender: {user_sender}")

        # Step 1: Format compliance response
        approval_status = " APPROVED" if msg.compliance_score >= 75 else " REJECTED"
        status = (
            "APPROVED for partnership"
            if msg.compliance_score >= 75
            else "NOT APPROVED - below compliance threshold"
        )

        # Build response message
        response_text = f"""
{approval_status}

Supplier: {msg.supplier_name}
Compliance Score: {msg.compliance_score}/100
Status: {status}

Ethics & Sustainability:
  • Ethics Info: {msg.ethics_info or 'N/A'}
  • Sustainability: {msg.sustainability_info or 'N/A'}

Violations Found: {len(msg.violations)}
{chr(10).join([f"  • {v}" for v in msg.violations]) if msg.violations else "  None"}
        """

        # Step 2: Send response back to user via chat
        ctx.logger.info(f"Sending response to user: {user_sender}")

        response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(type="text", text=response_text.strip()),
                EndSessionContent(type="end-session"),
            ],
        )

        await ctx.send(user_sender, response)
        
        ctx.logger.info(f"✓ Response sent to user")

        # Step 3: Clean up session
        active_sessions.pop(msg.request_id, None)
        ctx.storage.set("active_sessions", active_sessions)
        ctx.logger.info(f"Session cleaned up for request {msg.request_id}")

    except Exception as e:
        ctx.logger.error(f"Error handling compliance response: {e}")
        import traceback

        traceback.print_exc()

@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}"
    )

supplier_orchestrator.include(chat_proto, publish_manifest=True)
supplier_orchestrator.include(compliance_protocol, publish_manifest=True)

if __name__ == "__main__":
    supplier_orchestrator.run()
