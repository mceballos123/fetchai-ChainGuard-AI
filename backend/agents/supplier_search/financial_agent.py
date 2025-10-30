from uagents import Agent, Context, Protocol, Model
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime

# Import models
from backend.models.financial import FinancialRequest, FinancialResponse
from dotenv import load_dotenv
from backend.langgraph_logic.financial_langgraph import (
    FinancialRAGSystem,
    build_financial_workflow,
)
from backend.langgraph_logic.state_schemas import SupplierWorkflowState

# Import test utilities
from backend.test_func import (
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
    1. Receive supplier info and industry
    2. Store previous state and update current state
    3. Create workflow state from request
    4. Run LangGraph financial workflow:
       - financial_check_node: RAG analysis
       - financial_router: conditional routing (score >= 70?)
       - success_node or error_node: prepare response
    5. Build FinancialResponse from workflow result
    6. Return response to orchestrator
    """
    ctx.logger.info(f"📥 Received FinancialRequest from {sender}")
    ctx.logger.info(f"   Request ID: {msg.request_id}")
    ctx.logger.info(f"   Supplier: {msg.supplier_name}")
    ctx.logger.info(f"   Industry: {msg.industry}")

    # Log request reception
    log_request_reception(ctx, msg.request_id, msg.supplier_name, sender)

    # === STATE MANAGEMENT: Move current to previous ===
    previous_supplier = ctx.storage.get("current_supplier")
    if previous_supplier:
        ctx.storage.set("previous_supplier", previous_supplier)
        ctx.logger.info(
            f"Previous supplier stored: {previous_supplier.get('supplier_name', 'N/A')}"
        )

    # === STATE MANAGEMENT: Set new current supplier ===
    current_supplier_info = {
        "request_id": msg.request_id,
        "supplier_name": msg.supplier_name,
        "industry": msg.industry,
        "timestamp": msg.timestamp,
        "sender": sender,
    }
    ctx.storage.set("current_supplier", current_supplier_info)
    ctx.logger.info(f"Current supplier state updated: {msg.supplier_name}")

    # Log state transition if there was a previous supplier
    if previous_supplier:
        ctx.logger.info(
            f"State Transition: {previous_supplier.get('supplier_name', 'N/A')} → {msg.supplier_name}"
        )

    try:
        # === LANGGRAPH WORKFLOW ===
        ctx.logger.info("\n" + "=" * 70)
        ctx.logger.info("RUNNING LANGGRAPH FINANCIAL WORKFLOW")
        ctx.logger.info("=" * 70)

        # Create workflow state from request
        workflow_state: SupplierWorkflowState = {
            "request_id": msg.request_id,
            "supplier_name": msg.supplier_name,
            "company_values": "",
            "industry": msg.industry,
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
            "risk_info": None,
            "current_step": "started",
            "error_message": None,
            "should_continue": True,
            "messages": [],
            "timestamp": msg.timestamp,
        }

        # Run the workflow
        if financial_workflow is None:
            ctx.logger.warning("Workflow not initialized, using direct RAG query")
            rag_result = await rag_system.query_financial_documents(
                ctx=ctx,
                supplier_name=msg.supplier_name,
                industry=msg.industry,
            )
            workflow_result = {
                "financial_score": rag_result["financial_score"],
                "financial_info": rag_result["financial_details"],
                "current_step": (
                    "financial_approved"
                    if rag_result["financial_score"] >= 70
                    else "financial_rejected"
                ),
            }
        else:
            # Run the compiled LangGraph workflow
            workflow_result = financial_workflow.invoke(workflow_state)

        ctx.logger.info("=" * 70)
        ctx.logger.info(f"Workflow completed: {workflow_result.get('current_step')}")
        ctx.logger.info("=" * 70 + "\n")

        # Build response based on workflow result
        response = FinancialResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            financial_score=workflow_result.get("financial_score", 0.0),
            financial_details=workflow_result.get("financial_info", "") or "",
            risk_factors=[],  # Could extract from workflow_result if needed
            timestamp="",
        )

        # Validate response before sending
        is_valid, error_msg = validate_financial_response_before_sending(ctx, response)
        if not is_valid:
            ctx.logger.warning(f"Response validation failed: {error_msg}")

        ctx.logger.info(f"Financial analysis complete!")
        ctx.logger.info(f"Financial Score: {response.financial_score}/100")
        ctx.logger.info(f"Status: {workflow_result.get('current_step', 'unknown')}")

        # === STATE MANAGEMENT: Add result to current state ===
        # add current supplier info to the context storage
        current_supplier_info["financial_score"] = response.financial_score
        current_supplier_info["financial_details"] = response.financial_details
        current_supplier_info["current_step"] = workflow_result.get("current_step")
        ctx.storage.set("current_supplier", current_supplier_info)

        # === STATE MANAGEMENT: Add to history ===
        supplier_history = ctx.storage.get("supplier_history") or []
        supplier_history.append(current_supplier_info)
        ctx.storage.set("supplier_history", supplier_history)
        ctx.logger.info(f"Added to history (total: {len(supplier_history)} suppliers)")

        # Log response transmission
        log_response_transmission(
            ctx,
            msg.request_id,
            msg.supplier_name,
            response.financial_score,
            sender,
        )

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

        ctx.logger.info(f"Sent FinancialResponse to {sender}")

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
