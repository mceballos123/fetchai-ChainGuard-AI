
from uagents import Agent, Context, Protocol, Model
from pydantic import Field
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime

# Import models
from backend.models.compliance import ComplianceRequest, ComplianceResponse
from dotenv import load_dotenv
from backend.langgraph_logic.compliance_langgraph import ComplianceRAGSystem

load_dotenv()

compliance_agent = Agent(
    name="compliance_agent",
    seed=os.getenv("COMPLIANCE_AGENT_SEED"),
    port=8001,
    mailbox=True,
)

compliance_protocol = Protocol(name="compliance_protocol", version="1.0")

rag_system = None

@compliance_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent and RAG system on startup"""
    ctx.logger.info(" Compliance Agent starting up...")
    ctx.logger.info(f"Agent address: {compliance_agent.address}")

    # Initialize RAG system
    global rag_system
    if rag_system is None:
        rag_system = ComplianceRAGSystem()

    success = await rag_system.initialize(ctx)

    if success:
        ctx.logger.info(" Compliance Agent ready with RAG system!")
    else:
        ctx.logger.warning(" Compliance Agent running in fallback mode (no RAG)")

    # Initialize state storage for supplier information
    ctx.storage.set("supplier_history", [])
    ctx.storage.set("current_supplier", None)
    ctx.storage.set("previous_supplier", None)

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info(" Listening for ComplianceRequest messages...")

# checks if the agent is shutting down and cleans up the rag system
@compliance_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info(" Compliance Agent shutting down...")

    # Log final state before shutdown
    supplier_history = ctx.storage.get("supplier_history") or []
    ctx.logger.info(f" Total suppliers processed: {len(supplier_history)}")

def get_supplier_state(ctx: Context) -> Dict[str, Any]:
    """
    Helper function to retrieve supplier state information.

    Returns:
        dict: Contains current_supplier, previous_supplier, and supplier_history
    """
    return {
        "current_supplier": ctx.storage.get("current_supplier"),
        "previous_supplier": ctx.storage.get("previous_supplier"),
        "supplier_history": ctx.storage.get("supplier_history") or [],
    }

# process the complaicne requence from the supplier search agent
@compliance_protocol.on_message(model=ComplianceRequest, replies=ComplianceResponse)
async def handle_compliance_request(ctx: Context, sender: str, msg: ComplianceRequest):
    """
    Handle compliance check request from Find Supplier Agent.

    Process:
    1. Receive supplier info and company values
    2. Store previous state and update current state
    3. Query RAG system (Pinecone + LlamaIndex) for EPA/BBB documents
    4. Analyze compliance, ethics, sustainability
    5. Calculate combined score (0-100)
    6. Update state with results
    7. Return ComplianceResponse
    """
    ctx.logger.info(f" Received ComplianceRequest from {sender}")
    ctx.logger.info(f"   Request ID: {msg.request_id}")
    ctx.logger.info(f"   Supplier: {msg.supplier_name}")
    ctx.logger.info(f"   Industry: {msg.industry}")
    ctx.logger.info(f"   Company values: {msg.company_values}")

    # === STATE MANAGEMENT: Move current to previous ===
    previous_supplier = ctx.storage.get("current_supplier")
    if previous_supplier:
        ctx.storage.set("previous_supplier", previous_supplier)
        ctx.logger.info(
            f" Previous supplier stored: {previous_supplier.get('supplier_name', 'N/A')}"
        )

    # === STATE MANAGEMENT: Set new current supplier ===
    current_supplier_info = {
        "request_id": msg.request_id,
        "supplier_name": msg.supplier_name,
        "industry": msg.industry,
        "company_values": msg.company_values,
        "timestamp": msg.timestamp,
        "sender": sender,
    }
    ctx.storage.set("current_supplier", current_supplier_info)
    ctx.logger.info(f" Current supplier state updated: {msg.supplier_name}")

    # Log state transition if there was a previous supplier
    if previous_supplier:
        ctx.logger.info(
            f" State Transition: {previous_supplier.get('supplier_name', 'N/A')} → {msg.supplier_name}"
        )

    try:
        # Query RAG system for compliance analysis
        rag_result = await rag_system.query_compliance_documents(
            ctx=ctx,
            supplier_name=msg.supplier_name,
            company_values=msg.company_values,
            industry=msg.industry,
        )

        # Build response
        response = ComplianceResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            compliance_score=rag_result["compliance_score"],
            violations=rag_result["violations"],
            sustainability_info=rag_result["sustainability_info"],
            ethics_info=rag_result["ethics_info"],
            timestamp="",
        )

        ctx.logger.info(f" Compliance analysis complete!")
        ctx.logger.info(f"Score: {response.compliance_score}/100")
        ctx.logger.info(f"Violations: {len(response.violations)}")

        # === STATE MANAGEMENT: Add result to current state ===
        current_supplier_info["compliance_score"] = response.compliance_score
        current_supplier_info["violations"] = response.violations
        current_supplier_info["sustainability_info"] = response.sustainability_info
        current_supplier_info["ethics_info"] = response.ethics_info
        ctx.storage.set("current_supplier", current_supplier_info)

        # === STATE MANAGEMENT: Add to history ===
        supplier_history = ctx.storage.get("supplier_history") or []
        supplier_history.append(current_supplier_info)
        ctx.storage.set("supplier_history", supplier_history)
        ctx.logger.info(f" Added to history (total: {len(supplier_history)} suppliers)")

        # Send response back to sender (Find Supplier Agent)
        await ctx.send(sender, response)

        ctx.logger.info(f"Sent ComplianceResponse to {sender}")

    except Exception as e:
        ctx.logger.error(f"Error processing compliance request: {e}")

        # Send error response with low score
        error_response = ComplianceResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            compliance_score=0.0,
            violations=[f"Error processing request: {str(e)}"],
            sustainability_info="Error retrieving sustainability data",
            ethics_info="Error retrieving ethics data",
            timestamp="",
        )

        await ctx.send(sender, error_response)

compliance_agent.include(compliance_protocol, publish_manifest=True)

if __name__ == "__main__":

    compliance_agent.run()