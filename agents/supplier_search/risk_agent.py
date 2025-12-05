from uagents import Agent, Context, Protocol, Model
from typing import List, Optional, Dict, Any
import os

from datetime import datetime

# Import models
from models.risk import RiskRequest, RiskResponse
from dotenv import load_dotenv
from langgraph_logic.risk_langgraph import (
    RiskAnalysisSystem,
    build_risk_workflow,
)
from langgraph_logic.state_schemas import SupplierWorkflowState

# Import test utilities
from test_func.compliance_helpers import (
    verify_risk_connection,
    validate_risk_response_before_sending,
)

load_dotenv()

risk_agent = Agent(
    name="risk_agent",
    seed=os.getenv("RISK_AGENT_SEED"),
    port=8003,
    mailbox=True,
)

risk_protocol = Protocol(name="risk_protocol", version="1.0")

analysis_system = None
risk_workflow = None


@risk_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent, analysis system, and LangGraph workflow on startup"""
    ctx.logger.info("Risk Management Agent starting up...")

    # Initialize analysis system (uses Ollama LLM, no RAG/Pinecone)
    global analysis_system, risk_workflow
    if analysis_system is None:
        analysis_system = RiskAnalysisSystem()

    success = await analysis_system.initialize(ctx)

    if success:
        ctx.logger.info("Risk Agent ready with Ollama LLM!")

        # Build LangGraph workflow
        if risk_workflow is None:
            risk_workflow = build_risk_workflow(analysis_system, ctx)
            ctx.logger.info("LangGraph risk workflow built!")
    else:
        ctx.logger.warning("Risk Management Agent running in fallback mode")

    # Initialize state storage
    ctx.storage.set("supplier_history", [])
    ctx.storage.set("current_supplier", None)

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

    # Verify connection on startup
    verify_risk_connection(ctx)


@risk_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Risk Management Agent shutting down...")


@risk_protocol.on_message(model=RiskRequest, replies=RiskResponse)
async def handle_risk_request(ctx: Context, sender: str, msg: RiskRequest):
    """
    Handle risk management check request from Orchestrator Agent.

    Process:
    1. Receive supplier info and B Corp profile URL
    2. Create workflow state from request
    3. Run LangGraph risk management workflow with RAG + Web Scraping:
       - risk_check_node: Scrape B Corp page + RAG analysis
       - risk_router: conditional routing (score >= 60?)
       - success_node or error_node: prepare response
    4. Build RiskResponse from workflow result
    5. Return response to orchestrator

    Note: Works in parallel with compliance_agent. Both must pass (>= 60) for supplier approval.
    """
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 RISK AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier Name: {msg.supplier_name}")
    ctx.logger.info(f"Industry: {msg.industry}")
    ctx.logger.info(f"B Corp Profile URL: {msg.b_corp_profile_url}")
    ctx.logger.info("=" * 70)

    # Update current supplier state
    current_supplier_info = {
        "request_id": msg.request_id,
        "supplier_name": msg.supplier_name,
        "industry": msg.industry,
        "timestamp": msg.timestamp,
        "sender": sender,
    }
    ctx.storage.set("current_supplier", current_supplier_info)

    try:
        # === USING LANGGRAPH WORKFLOW ===

        if risk_workflow is None:
            ctx.logger.error("Risk workflow not initialized!")
            raise RuntimeError("Risk workflow not ready")

        ctx.logger.info("Starting LangGraph risk workflow...")
        ctx.logger.info(f"Will scrape B Corp profile for risk analysis")

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
            supplier_country=None,
            b_corp_profile_url=msg.b_corp_profile_url,
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
            current_step="risk_check",
            error_message=None,
            should_continue=True,
            messages=[],
        )

        result = risk_workflow.invoke(workflow_state)

        ctx.logger.info("LangGraph workflow completed")

        # Extract results
        risk_score = result.get("risk_score", 70.0)
        risk_details = result.get("risk_details", "Analysis complete")
        risk_factors = result.get("risk_factors", [])

        ctx.logger.info(f"Risk Analysis Results:")
        ctx.logger.info(f"  - Score: {risk_score}/100")
        ctx.logger.info(f"  - Risk Factors: {len(risk_factors)}")
        ctx.logger.info(f"  - Status: {'APPROVED' if risk_score >= 60 else 'REJECTED'}")

        # Build response
        response = RiskResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            risk_score=risk_score,
            risk_details=risk_details,
            risk_factors=risk_factors if risk_factors else [],
            timestamp="",
        )

        # Validate response before sending
        is_valid, error_msg = validate_risk_response_before_sending(ctx, response)
        if not is_valid:
            ctx.logger.warning(f"Response validation failed: {error_msg}")

        # Determine status based on score (threshold: 60)
        current_step = "risk_approved" if risk_score >= 60 else "risk_rejected"

        # Update current supplier info with results
        current_supplier_info["risk_score"] = response.risk_score
        current_supplier_info["risk_details"] = response.risk_details
        current_supplier_info["risk_factors"] = response.risk_factors
        current_supplier_info["current_step"] = current_step
        ctx.storage.set("current_supplier", current_supplier_info)

        # Add to history
        supplier_history = ctx.storage.get("supplier_history") or []
        supplier_history.append(current_supplier_info)
        ctx.storage.set("supplier_history", supplier_history)

        # Send response back to sender (Orchestrator Agent)
        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("📤 RISK AGENT: SENDING RESPONSE")
        ctx.logger.info("=" * 70)
        ctx.logger.info(f"To: {sender}")
        ctx.logger.info(f"Request ID: {msg.request_id}")
        ctx.logger.info(f"Risk Score: {response.risk_score}/100")
        ctx.logger.info("=" * 70)

        await ctx.send(sender, response)

    except Exception as e:
        ctx.logger.error(f"Error processing risk request: {e}")
        import traceback

        traceback.print_exc()

        # Send error response with low score
        error_response = RiskResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            risk_score=0.0,
            risk_details="Error retrieving risk management data",
            risk_factors=[f"Error processing request: {str(e)}"],
            timestamp="",
        )

        await ctx.send(sender, error_response)


risk_agent.include(risk_protocol, publish_manifest=True)

if __name__ == "__main__":
    risk_agent.run()
