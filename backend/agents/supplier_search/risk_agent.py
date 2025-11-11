from uagents import Agent, Context, Protocol, Model
from typing import List, Optional, Dict, Any
import os

from datetime import datetime

# Import models
from backend.models.risk import RiskRequest, RiskResponse
from dotenv import load_dotenv
from backend.langgraph_logic.risk_langgraph import (
    RiskRAGSystem,
    build_risk_workflow,
)
from backend.langgraph_logic.state_schemas import SupplierWorkflowState

# Import test utilities
from backend.test_func import (
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

rag_system = None
risk_workflow = None


@risk_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent, RAG system, and LangGraph workflow on startup"""
    ctx.logger.info("Risk Management Agent starting up...")
    ctx.logger.info(f"Agent address: {risk_agent.address}")

    # Initialize RAG system
    global rag_system, risk_workflow
    if rag_system is None:
        rag_system = RiskRAGSystem()

    success = await rag_system.initialize(ctx)

    if success:
        ctx.logger.info("Risk Management Agent ready with RAG system!")

        # Build LangGraph workflow
        if risk_workflow is None:
            risk_workflow = build_risk_workflow(rag_system, ctx)
            ctx.logger.info("LangGraph risk management workflow built!")
    else:
        ctx.logger.warning("Risk Management Agent running in fallback mode (no RAG)")

    # Initialize state storage
    ctx.storage.set("supplier_history", [])
    ctx.storage.set("current_supplier", None)

    # Initialize request trace for debugging
    ctx.storage.set(
        "request_trace",
        {
            "received_requests": [],
            "sent_responses": [],
        },
    )

    ctx.logger.info("Listening for RiskRequest messages...")

    # Verify connection on startup
    conn_status = verify_risk_connection(ctx)
    ctx.logger.info(f"Connection Status: {conn_status['status']}")


@risk_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Risk Management Agent shutting down...")

    # Log final state before shutdown
    supplier_history = ctx.storage.get("supplier_history") or []
    ctx.logger.info(f"Total suppliers processed: {len(supplier_history)}")
    ctx.logger.info("Workflow and RAG system cleanup complete")


@risk_protocol.on_message(model=RiskRequest, replies=RiskResponse)
async def handle_risk_request(
    ctx: Context, sender: str, msg: RiskRequest
):  # Causing a issue with the risk_agent
    """
    Handle risk management check request from Orchestrator Agent.

    Process:
    1. Receive supplier info
    2. Create workflow state from request
    3. Run LangGraph risk management workflow:
       - risk_check_node: RAG analysis
       - risk_router: conditional routing (score >= 75?)
       - success_node or error_node: prepare response
    4. Build RiskResponse from workflow result
    5. Return response to orchestrator
    """
    ctx.logger.info(f"Received RiskRequest from {sender}")
    ctx.logger.info(f"   Request ID: {msg.request_id}")
    ctx.logger.info(f"   Supplier: {msg.supplier_name}")
    ctx.logger.info(f"   Industry: {msg.industry}")

    # Update current supplier state
    current_supplier_info = {
        "request_id": msg.request_id,
        "supplier_name": msg.supplier_name,
        "industry": msg.industry,
        "timestamp": msg.timestamp,
        "sender": sender,
    }
    ctx.storage.set("current_supplier", current_supplier_info)
    ctx.logger.info(f"Current supplier state updated: {msg.supplier_name}")

    try:
        # === LANGGRAPH WORKFLOW ===
        ctx.logger.info("\n" + "=" * 70)
        ctx.logger.info("RUNNING LANGGRAPH RISK MANAGEMENT WORKFLOW")
        ctx.logger.info("=" * 70)

        # Create workflow state from request
        workflow_state: SupplierWorkflowState = {
            "request_id": msg.request_id,
            "supplier_name": msg.supplier_name,
            "industry": msg.industry,
            "company_values": "",
            "product_needed": "",
            "business_type": "",
            "user_input": "",
            "compliance_score": None,
            "ethics_info": None,
            "sustainability_info": None,
            "violations": [],
            "supplier_location": None,
            "supplier_country": None,
            "retrieved_documents": None,
            "rag_context": None,
            "financial_score": None,
            "financial_info": None,
            "risk_score": None,
            "risk_details": None,
            "risk_factors": [],
            "current_step": "started",
            "error_message": None,
            "should_continue": True,
            "messages": [],
            "timestamp": msg.timestamp,
        }

        # Run the workflow
        if risk_workflow is None:
            ctx.logger.warning("Workflow not initialized, using direct RAG query")
            rag_result = await rag_system.query_risk_documents(
                ctx=ctx,
                supplier_name=msg.supplier_name,
                industry=msg.industry,
            )
            workflow_result = {
                "risk_score": rag_result["risk_score"],
                "risk_details": rag_result["risk_details"],
                "risk_factors": rag_result["risk_factors"],
                "current_step": (
                    "risk_approved"
                    if rag_result["risk_score"] >= 75
                    else "risk_rejected"
                ),
            }
        else:
            # Run the compiled LangGraph workflow
            workflow_result = risk_workflow.invoke(workflow_state)

        ctx.logger.info("=" * 70)
        ctx.logger.info(f"Workflow completed: {workflow_result.get('current_step')}")
        ctx.logger.info("=" * 70 + "\n")

        # Build response based on workflow result
        response = RiskResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            risk_score=workflow_result.get("risk_score", 0.0),
            risk_details=workflow_result.get("risk_details", "") or "",
            risk_factors=workflow_result.get("risk_factors", []),
            timestamp="",
        )

        # Validate response before sending
        is_valid, error_msg = validate_risk_response_before_sending(ctx, response)
        if not is_valid:
            ctx.logger.warning(f"Response validation failed: {error_msg}")

        ctx.logger.info("Risk management analysis complete!")
        ctx.logger.info(f"Score: {response.risk_score}/100")
        ctx.logger.info(f"Risk Factors: {len(response.risk_factors)}")
        ctx.logger.info(f"Status: {workflow_result.get('current_step', 'unknown')}")

        # Update current supplier info with results
        current_supplier_info["risk_score"] = response.risk_score
        current_supplier_info["risk_details"] = response.risk_details
        current_supplier_info["risk_factors"] = response.risk_factors
        current_supplier_info["current_step"] = workflow_result.get("current_step")
        ctx.storage.set("current_supplier", current_supplier_info)

        # Add to history
        supplier_history = ctx.storage.get("supplier_history") or []
        supplier_history.append(current_supplier_info)
        ctx.storage.set("supplier_history", supplier_history)
        ctx.logger.info(f"Added to history (total: {len(supplier_history)} suppliers)")

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

        ctx.logger.info(f"Sent RiskResponse to {sender}")

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
