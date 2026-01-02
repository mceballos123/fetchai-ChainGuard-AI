"""
Supplier Info Agent - Extracts detailed company information from SEC filings
Uses LangGraph workflow to determine company type and extract relevant information
"""

from typing import Optional
from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from models.supplier_info import (
    SupplierInfoRequest,
    SupplierInfoResponse,
    SupplierInfoResult,
    SupplierInfoState,
)
from langgraph_logic.supplier_info import supplier_info_graph
import os
import nest_asyncio
from datetime import datetime

nest_asyncio.apply()
load_dotenv()

supplier_info_agent = Agent(
    name="supplier_info_agent",
    seed=os.getenv("SUPPLIER_INFO_SEED"),
    port=8007,
    mailbox=True,
)

supplier_info_protocol = Protocol(name="supplier_info_protocol", version="1.0")


@supplier_info_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize supplier info agent on startup"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("SUPPLIER INFO AGENT STARTING UP")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Port: 8007")
    ctx.logger.info("Extracts company information from SEC filings (10-Q/20-F)")
    ctx.logger.info("=" * 70)

    # Initialize storage
    ctx.storage.set("processed_requests", [])
    ctx.storage.set("request_count", 0)

    ctx.logger.info(
        "Supplier Info Agent Ready - Listening for SupplierInfoRequest messages..."
    )


@supplier_info_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("SUPPLIER INFO AGENT SHUTTING DOWN")
    ctx.logger.info("=" * 70)

    request_count = ctx.storage.get("request_count") or 0
    ctx.logger.info(f"Total requests processed: {request_count}")


@supplier_info_protocol.on_message(
    model=SupplierInfoRequest, replies=SupplierInfoResponse
)
async def handle_supplier_info_request(
    ctx: Context, sender: str, msg: SupplierInfoRequest
):
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 SUPPLIER INFO AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Company: {msg.company_name}")
    ctx.logger.info(f"CIK: {msg.cik}")
    ctx.logger.info(f"Country: {msg.country}")
    ctx.logger.info("=" * 70)
    ctx.logger.info("")
    ctx.logger.info("🔍 Starting LangGraph Supplier Info Extraction...")
    ctx.logger.info("=" * 70)

    try:
        # Check for duplicate requests
        processed_requests = ctx.storage.get("processed_requests") or []
        if msg.request_id in processed_requests:
            ctx.logger.warning(f"Duplicate request: {msg.request_id}")
            return

        processed_requests.append(msg.request_id)
        ctx.storage.set("processed_requests", processed_requests)

        # Increment request count
        request_count = ctx.storage.get("request_count") or 0
        request_count += 1
        ctx.storage.set("request_count", request_count)

        # Build initial state for LangGraph
        initial_state: SupplierInfoState = {
            "request_id": msg.request_id,
            "company_name": msg.company_name,
            "cik": msg.cik,
            "country": msg.country,
            "ticker": msg.ticker,
            "is_foreign": None,
            "filing_type": None,
            "latest_filing_url": None,
            "filing_content": None,
            "company_description": None,
            "history": None,
            "success": False,
            "error_message": None,
            "score": 100,
            "ctx": ctx,
        }

        ctx.logger.info(f"🔄 Executing LangGraph workflow for {msg.company_name}")

        # Execute the LangGraph workflow
        final_state = supplier_info_graph.invoke(initial_state)

        ctx.logger.info("LangGraph Workflow Completed")

        # Process the results
        if final_state["success"] and final_state.get("company_description"):
            ctx.logger.info(f"✅ Successfully extracted company information!")
            ctx.logger.info(f"  Company: {msg.company_name}")
            ctx.logger.info(
                f"  Type: {'Foreign' if final_state['is_foreign'] else 'American'}"
            )
            ctx.logger.info(f"  Filing Type: {final_state['filing_type']}")
            ctx.logger.info(
                f"  Description Length: {len(final_state['company_description'])} chars"
            )

            # Build the result
            supplier_info = SupplierInfoResult(
                company_name=msg.company_name,
                cik=msg.cik,
                country=msg.country,
                is_foreign=final_state["is_foreign"],
                filing_type=final_state["filing_type"],
                company_description=final_state["company_description"],
                history=final_state.get("history", ""),
                latest_filing_url=final_state.get("latest_filing_url", ""),
            )

            response = SupplierInfoResponse(
                request_id=msg.request_id,
                success=True,
                supplier_info=supplier_info,
                timestamp=datetime.now().isoformat(),
            )

            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("📤 SUPPLIER INFO AGENT: SENDING RESPONSE")
            ctx.logger.info("=" * 70)
            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: True")
            ctx.logger.info("=" * 70)

            await ctx.send(sender, response)

        else:
            # Handle errors
            error_msg = final_state.get("error_message") or "Failed to extract company information"
            ctx.logger.warning(f"❌ Failed to extract company info: {error_msg}")

            error_response = SupplierInfoResponse(
                request_id=msg.request_id,
                success=False,
                error_message=error_msg,
                timestamp=datetime.now().isoformat(),
            )

            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("📤 SUPPLIER INFO AGENT: SENDING ERROR RESPONSE")
            ctx.logger.info("=" * 70)
            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: False")
            ctx.logger.info(f"Error: {error_msg}")
            ctx.logger.info("=" * 70)

            await ctx.send(sender, error_response)

    except Exception as e:
        ctx.logger.error(f"Error processing supplier info request: {e}")
        import traceback

        traceback.print_exc()

        error_response = SupplierInfoResponse(
            request_id=msg.request_id,
            success=False,
            error_message=str(e),
            timestamp=datetime.now().isoformat(),
        )

        await ctx.send(sender, error_response)


supplier_info_agent.include(supplier_info_protocol, publish_manifest=True)

if __name__ == "__main__":
    supplier_info_agent.run()
