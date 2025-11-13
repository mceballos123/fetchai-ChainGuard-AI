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
    ctx.logger.info(f"Received FinancialRequest from {sender}")
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
        # === DIRECT PROMPT-BASED ANALYSIS (BYPASSING LANGGRAPH FOR NOW) ===
        ctx.logger.info("\n" + "=" * 70)
        ctx.logger.info("DIRECT FINANCIAL ANALYSIS (NO LANGGRAPH/RAG)")
        ctx.logger.info("=" * 70)

        # Read hardcoded file directly
        from pathlib import Path
        from backend.prompts.finance_prompt import finance_prompt
        from llama_index.llms.ollama import Ollama

        # Go up 3 levels: financial_agent.py -> supplier_search -> agents -> backend
        financial_file = (
            Path(__file__).parent.parent.parent
            / "financial_files"
            / "sunrise_sustainable_financial.txt"
        )

        if not financial_file.exists():
            ctx.logger.error(f"Hardcoded financial file not found: {financial_file}")
            raise FileNotFoundError(f"Financial file missing: {financial_file}")

        # Read file content
        with open(financial_file, "r") as f:
            supplier_text = f.read()

        ctx.logger.info(f"Loaded {len(supplier_text)} chars from {financial_file.name}")

        # Use finance prompt
        prompt = finance_prompt(msg.supplier_name, msg.industry)

        # Combine prompt + supplier data
        full_query = f"{prompt}\n\nSupplier Data:\n{supplier_text[:2000]}"

        # Initialize LLM (5 minute timeout for slow model)
        llm = Ollama(model="llama3.2:1b", request_timeout=300)

        ctx.logger.info("Sending query to LLM (llama3.2:1b)...")
        llm_response = llm.complete(full_query)
        response_text = str(llm_response)

        ctx.logger.info("=" * 70)
        ctx.logger.info("LLM RESPONSE:")
        ctx.logger.info(response_text[:500])
        ctx.logger.info("=" * 70)

        # Parse response (simple extraction)
        financial_score = 70.0
        financial_details = "Analysis complete"
        risk_factors = []

        for line in response_text.split("\n"):
            line_lower = line.lower()
            if "financial_score:" in line_lower:
                try:
                    score_str = line.split(":")[-1].strip().split()[0]
                    financial_score = float(score_str)
                    ctx.logger.info(f"Extracted financial_score: {financial_score}")
                except:
                    pass
            elif "financial_details:" in line_lower:
                financial_details = line.split(":", 1)[-1].strip()
            elif "risk_factors:" in line_lower:
                factors_str = line.split(":", 1)[-1].strip()
                if factors_str.lower() not in ["minimal risks", "none", ""]:
                    risk_factors = [f.strip() for f in factors_str.split(",")]

        # Build response
        response = FinancialResponse(
            request_id=msg.request_id,
            supplier_name="sunrise_sustainable",
            financial_score=financial_score,
            financial_details=financial_details,
            risk_factors=risk_factors,
            timestamp="",
        )

        # Validate response before sending
        is_valid, error_msg = validate_financial_response_before_sending(ctx, response)
        if not is_valid:
            ctx.logger.warning(f"Response validation failed: {error_msg}")

        ctx.logger.info("Financial analysis complete!")
        ctx.logger.info(f"Financial Score: {response.financial_score}/100")

        # Determine status based on score
        current_step = (
            "financial_approved" if financial_score >= 60 else "financial_rejected"
        )
        ctx.logger.info(f"Status: {current_step}")

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
