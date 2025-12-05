from uagents import Agent, Context, Protocol, Model
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime

# Import models
from models.financial import FinancialRequest, FinancialResponse
from dotenv import load_dotenv
from langgraph_logic.financial_langgraph import (
    FinancialRAGSystem,
    build_financial_workflow,
)
from langgraph_logic.state_schemas import SupplierWorkflowState

# Import test utilities

from test_func.compliance_helpers import (
    verify_compliance_connection,
    log_request_reception,
    log_response_transmission,
    validate_financial_response_before_sending,
)

load_dotenv()

financial_agent = Agent(
    name="financial_agent",
    seed=os.getenv("FINANCIAL_AGENT_SEED"),
    port=8002,
    mailbox=True,
)

financial_protocol = Protocol(name="financial_protocol", version="1.0")

rag_system = None
financial_workflow = None


@financial_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent, RAG system, and LangGraph workflow on startup"""
    ctx.logger.info("Financial Agent starting up...")
    ctx.logger.info(f"Agent address: {financial_agent.address}")

    # Initialize RAG system
    global rag_system, financial_workflow
    if rag_system is None:
        rag_system = FinancialRAGSystem()

    success = await rag_system.initialize(ctx)

    if success:
        ctx.logger.info("Financial Agent ready with RAG system!")

        # Build LangGraph workflow
        if financial_workflow is None:
            financial_workflow = build_financial_workflow(rag_system, ctx)
            ctx.logger.info("LangGraph financial workflow built!")
    else:
        ctx.logger.warning("Financial Agent running in fallback mode (no RAG)")

    # Initialize state storage for supplier information
    ctx.storage.set("supplier_history", [])
    ctx.storage.set("current_supplier", None)
    ctx.storage.set("previous_supplier", None)

    # Track processed request IDs to prevent duplicates
    ctx.storage.set("processed_request_ids", [])

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info("Listening for FinancialRequest messages...")

    # Verify connection on startup
    conn_status = verify_compliance_connection(ctx)
    ctx.logger.info(f"Connection Status: {conn_status['status']}")


# checks if the agent is shutting down and cleans up the rag system
@financial_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Financial Agent shutting down...")

    # Log final state before shutdown
    supplier_history = ctx.storage.get("supplier_history") or []
    ctx.logger.info(f"Total suppliers processed: {len(supplier_history)}")
    ctx.logger.info("Workflow and RAG system cleanup complete")


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


# process the financial request from the supplier search agent
@financial_protocol.on_message(model=FinancialRequest, replies=FinancialResponse)
async def handle_financial_request(ctx: Context, sender: str, msg: FinancialRequest):
    """
    Handle financial risk check request from Orchestrator Agent.

    Process:
    1. Receive supplier info, B Corp URL, and industry from orchestrator
    2. Store previous state and update current state
    3. Create workflow state from request
    4. Run LangGraph financial workflow:
       - financial_check_node: Scrape B Corp page ONLY for Headquarters to extract country
       - Apply country-specific financial analysis based on US trade relationships
       - financial_router: conditional routing (score >= 60?)
       - success_node or error_node: prepare response
    5. Build FinancialResponse from workflow result
    6. Return response to orchestrator

    Note: Only extracts country from Headquarters section (e.g., "Catalonia, Spain" -> "Spain")
    """
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 FINANCIAL AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier Name: {msg.supplier_name}")
    ctx.logger.info(f"B Corp URL: {msg.b_corp_profile_url}")
    ctx.logger.info(f"Industry: {msg.industry}")
    ctx.logger.info("=" * 70)

    # Log request reception
    log_request_reception(ctx, msg.request_id, msg.supplier_name, sender)

    # === STATE MANAGEMENT: Move current to previous ===
    previous_supplier = ctx.storage.get("current_supplier")
    if previous_supplier:
        ctx.storage.set("previous_supplier", previous_supplier)

    # === STATE MANAGEMENT: Set new current supplier ===
    current_supplier_info = {
        "request_id": msg.request_id,
        "supplier_name": msg.supplier_name,
        "b_corp_profile_url": msg.b_corp_profile_url,
        "industry": msg.industry,
        "timestamp": msg.timestamp,
        "sender": sender,
    }
    ctx.storage.set("current_supplier", current_supplier_info)

    try:
        # === USING LANGGRAPH WORKFLOW WITH RAG ===

        if financial_workflow is None:
            ctx.logger.error("❌ Financial workflow not initialized!")
            raise RuntimeError("Financial workflow not ready")

        ctx.logger.info("🔄 Starting LangGraph financial workflow...")
        ctx.logger.info(
            f"Will scrape B Corp Headquarters section ONLY to extract country"
        )

        # Create workflow state from request
        workflow_state = SupplierWorkflowState(
            request_id=msg.request_id,
            timestamp=msg.timestamp,
            user_input="",
            business_type="",
            company_values="",
            industry=msg.industry,
            product_needed="",
            supplier_name=msg.supplier_name,
            supplier_location=None,
            supplier_country=None,  # Will be scraped from B Corp page
            b_corp_profile_url=msg.b_corp_profile_url,  # Critical: B Corp URL to scrape country from
            retrieved_documents=None,
            rag_context=None,
            compliance_score=None,
            ethics_info=None,
            sustainability_info=None,
            violations=None,
            financial_score=None,
            financial_info=None,
            risk_score=None,
            risk_details=None,
            risk_factors=None,
            current_step="financial_check",
            error_message=None,
            should_continue=True,
            messages=[],
        )

        result = financial_workflow.invoke(workflow_state)

        ctx.logger.info("✅ LangGraph workflow completed")

        # Extract results
        financial_score = result.get("financial_score", 70.0)
        financial_details = result.get("financial_info", "Analysis complete")
        risk_factors = result.get("risk_factors", [])

        ctx.logger.info(f"Financial Analysis Results:")
        ctx.logger.info(f"  - Score: {financial_score}/100")
        ctx.logger.info(f"  - Risk Factors: {len(risk_factors)}")
        ctx.logger.info(
            f"  - Status: {'APPROVED' if financial_score >= 60 else 'REJECTED'}"
        )

        # Build response
        response = FinancialResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            financial_score=financial_score,
            financial_details=financial_details,
            risk_factors=risk_factors if risk_factors else [],
            timestamp="",
        )

        # Validate response before sending
        is_valid, error_msg = validate_financial_response_before_sending(ctx, response)
        if not is_valid:
            ctx.logger.warning(f"Response validation failed: {error_msg}")

        # Determine status based on score (threshold: 60)
        current_step = (
            "financial_approved" if financial_score >= 60 else "financial_rejected"
        )

        # === STATE MANAGEMENT: Add result to current state ===
        # add current supplier info to the context storage
        current_supplier_info["financial_score"] = response.financial_score
        current_supplier_info["financial_details"] = response.financial_details
        current_supplier_info["current_step"] = current_step
        ctx.storage.set("current_supplier", current_supplier_info)

        # === STATE MANAGEMENT: Add to history ===
        supplier_history = ctx.storage.get("supplier_history") or []
        supplier_history.append(current_supplier_info)
        ctx.storage.set("supplier_history", supplier_history)

        # Log response transmission
        log_response_transmission(
            ctx,
            msg.request_id,
            msg.supplier_name,
            response.financial_score,
            sender,
        )

        # Send response back to sender (Orchestrator Agent)
        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("📤 FINANCIAL AGENT: SENDING RESPONSE")
        ctx.logger.info("=" * 70)
        ctx.logger.info(f"To: {sender}")
        ctx.logger.info(f"Request ID: {msg.request_id}")
        ctx.logger.info(f"Financial Score: {response.financial_score}/100")
        ctx.logger.info("=" * 70)

        await ctx.send(sender, response)

    except Exception as e:
        ctx.logger.error(f"Error processing financial request: {e}")
        import traceback

        traceback.print_exc()

        # Send error response with low score
        error_response = FinancialResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            financial_score=0.0,
            risk_factors=[f"Error processing request: {str(e)}"],
            financial_details="Error retrieving financial data",
            timestamp="",
        )

        await ctx.send(sender, error_response)


financial_agent.include(financial_protocol, publish_manifest=True)

if __name__ == "__main__":

    financial_agent.run()
