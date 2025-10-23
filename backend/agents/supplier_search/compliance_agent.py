from uagents import Agent, Context, Protocol, Model
from pydantic import Field
from typing import List, Optional, Dict, Any
import os
import json
from datetime import datetime

# Import models
from backend.models.compliance import ComplianceRequest, ComplianceResponse
from dotenv import load_dotenv
from backend.langgraph_logic.compliance_langgraph import (
    ComplianceRAGSystem,
    build_compliance_workflow,
)
from backend.langgraph_logic.state_schemas import SupplierWorkflowState

# Import test utilities
from backend.test_func import (
    verify_compliance_connection,
    log_request_reception,
    log_response_transmission,
    validate_response_before_sending,
)

load_dotenv()

compliance_agent = Agent(
    name="compliance_agent",
    seed=os.getenv("COMPLIANCE_AGENT_SEED"),
    port=8001,
    mailbox=True,
)

compliance_protocol = Protocol(name="compliance_protocol", version="1.0")

rag_system = None
compliance_workflow = None


@compliance_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent, RAG system, and LangGraph workflow on startup"""
    ctx.logger.info(" Compliance Agent starting up...")
    ctx.logger.info(f"Agent address: {compliance_agent.address}")

    # Initialize RAG system
    global rag_system, compliance_workflow
    if rag_system is None:
        rag_system = ComplianceRAGSystem()

    success = await rag_system.initialize(ctx)

    if success:
        ctx.logger.info(" Compliance Agent ready with RAG system!")

        # Build LangGraph workflow
        if compliance_workflow is None:
            compliance_workflow = build_compliance_workflow(rag_system, ctx)
            ctx.logger.info(" ✓ LangGraph compliance workflow built!")
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

    # Verify connection on startup
    conn_status = verify_compliance_connection(ctx)
    ctx.logger.info(f"Connection Status: {conn_status['status']}")


# checks if the agent is shutting down and cleans up the rag system
@compliance_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info(" Compliance Agent shutting down...")

    # Log final state before shutdown
    supplier_history = ctx.storage.get("supplier_history") or []
    ctx.logger.info(f" Total suppliers processed: {len(supplier_history)}")
    ctx.logger.info(" Workflow and RAG system cleanup complete")


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
    Handle compliance check request from Orchestrator Agent.

    Process:
    1. Receive supplier info and company values
    2. Store previous state and update current state
    3. Create workflow state from request
    4. Run LangGraph compliance workflow:
       - compliance_check_node: RAG analysis
       - compliance_router: conditional routing (score >= 75?)
       - success_node or error_node: prepare response
    5. Build ComplianceResponse from workflow result
    6. Return response to orchestrator
    """
    ctx.logger.info(f" Received ComplianceRequest from {sender}")
    ctx.logger.info(f"   Request ID: {msg.request_id}")
    ctx.logger.info(f"   Supplier: {msg.supplier_name}")
    ctx.logger.info(f"   Industry: {msg.industry}")
    ctx.logger.info(f"   Company values: {msg.company_values}")

    # Log request reception
    log_request_reception(ctx, msg.request_id, msg.supplier_name, sender)

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
        # === LANGGRAPH WORKFLOW ===
        ctx.logger.info("\n" + "=" * 70)
        ctx.logger.info("🔄 RUNNING LANGGRAPH COMPLIANCE WORKFLOW")
        ctx.logger.info("=" * 70)

        # Create workflow state from request
        workflow_state: SupplierWorkflowState = {
            "request_id": msg.request_id,
            "supplier_name": msg.supplier_name,
            "company_values": msg.company_values,
            "industry": msg.industry,
            "product_needed": "",
            "business_type": "",
            "user_input": msg.company_values,
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
        if compliance_workflow is None:
            ctx.logger.warning("Workflow not initialized, using direct RAG query")
            rag_result = await rag_system.query_compliance_documents(
                ctx=ctx,
                supplier_name=msg.supplier_name,
                company_values=msg.company_values,
                industry=msg.industry,
            )
            workflow_result = {
                "compliance_score": rag_result["compliance_score"],
                "violations": rag_result["violations"],
                "sustainability_info": rag_result["sustainability_info"],
                "ethics_info": rag_result["ethics_info"],
                "current_step": (
                    "compliance_approved"
                    if rag_result["compliance_score"] >= 75
                    else "compliance_rejected"
                ),
            }
        else:
            # Run the compiled LangGraph workflow
            workflow_result = compliance_workflow.invoke(workflow_state)

        ctx.logger.info("=" * 70)
        ctx.logger.info(f"✓ Workflow completed: {workflow_result.get('current_step')}")
        ctx.logger.info("=" * 70 + "\n")

        # Build response based on workflow result
        response = ComplianceResponse(
            request_id=msg.request_id,
            supplier_name=msg.supplier_name,
            compliance_score=workflow_result.get("compliance_score", 0.0),
            violations=workflow_result.get("violations", []),
            sustainability_info=workflow_result.get("sustainability_info", "") or "",
            ethics_info=workflow_result.get("ethics_info", "") or "",
            timestamp="",
        )

        # Validate response before sending
        is_valid, error_msg = validate_response_before_sending(ctx, response)
        if not is_valid:
            ctx.logger.warning(f"Response validation failed: {error_msg}")

        ctx.logger.info(f" Compliance analysis complete!")
        ctx.logger.info(f"Score: {response.compliance_score}/100")
        ctx.logger.info(f"Violations: {len(response.violations)}")
        ctx.logger.info(f"Status: {workflow_result.get('current_step', 'unknown')}")

        # === STATE MANAGEMENT: Add result to current state ===
        # add current supplier info to the context storage
        current_supplier_info["compliance_score"] = response.compliance_score
        current_supplier_info["violations"] = response.violations
        current_supplier_info["sustainability_info"] = response.sustainability_info
        current_supplier_info["ethics_info"] = response.ethics_info
        current_supplier_info["current_step"] = workflow_result.get("current_step")
        ctx.storage.set("current_supplier", current_supplier_info)

        # === STATE MANAGEMENT: Add to history ===
        supplier_history = ctx.storage.get("supplier_history") or []
        supplier_history.append(current_supplier_info)
        ctx.storage.set("supplier_history", supplier_history)
        ctx.logger.info(f" Added to history (total: {len(supplier_history)} suppliers)")

        # Log response transmission
        log_response_transmission(
            ctx,
            msg.request_id,
            msg.supplier_name,
            response.compliance_score,
            sender,
        )

        # Send response back to sender (Orchestrator Agent)
        await ctx.send(sender, response)

        ctx.logger.info(f"Sent ComplianceResponse to {sender}")

    except Exception as e:
        ctx.logger.error(f"Error processing compliance request: {e}")
        import traceback

        traceback.print_exc()

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
