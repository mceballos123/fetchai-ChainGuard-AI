"""
Finance Info Agent - Extracts financial information from SEC 10-Q filings
Uses LangGraph workflow to extract key financial metrics from quarterly reports
"""

from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from models.finance_info import (
    FinanceInfoRequest,
    FinanceInfoResponse,
    FinancialData,
    FinanceInfoState,
)
from langgraph_logic.finance_info_langgraph import finance_info_graph
import os
import nest_asyncio
from datetime import datetime

nest_asyncio.apply()
load_dotenv()

finance_info_agent = Agent(
    name="finance_info_agent",
    seed=os.getenv("FINANCE_INFO_SEED"),
    port=8008,
    mailbox=True,
)
finance_info_protocol = Protocol(name="finance_info_protocol", version="1.0")


@finance_info_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize finance info agent on startup"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("FINANCE INFO AGENT STARTING UP")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Port: 8008")
    ctx.logger.info("Extracts financial information from SEC 10-Q filings")
    ctx.logger.info("Supported: US Companies only")
    ctx.logger.info("=" * 70)

    # Initialize storage
    ctx.storage.set("processed_requests", [])
    ctx.storage.set("request_count", 0)

    ctx.logger.info(
        "Finance Info Agent Ready - Listening for FinanceInfoRequest messages..."
    )


@finance_info_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("FINANCE INFO AGENT SHUTTING DOWN")
    ctx.logger.info("=" * 70)

    request_count = ctx.storage.get("request_count") or 0
    ctx.logger.info(f"Total requests processed: {request_count}")


@finance_info_protocol.on_message(
    model=FinanceInfoRequest, replies=FinanceInfoResponse
)
async def handle_finance_info_request(
    ctx: Context, sender: str, msg: FinanceInfoRequest
):
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 FINANCE INFO AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Company: {msg.company_name}")
    ctx.logger.info(f"CIK: {msg.cik}")
    ctx.logger.info(f"Country: {msg.country}")
    ctx.logger.info("=" * 70)
    ctx.logger.info("")
    ctx.logger.info("🔍 Starting LangGraph Financial Info Extraction...")
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
        initial_state: FinanceInfoState = {
            "request_id": msg.request_id,
            "company_name": msg.company_name,
            "cik": msg.cik,
            "country": msg.country,
            "ticker": msg.ticker,
            "filing_type": None,
            "latest_filing_url": None,
            "filing_date": None,
            "overview": None,
            "financial_condition": None,
            "revenue_info": None,
            "legal_proceedings": None,
            "success": False,
            "error_message": None,
            "score": 100,
            "ctx": ctx,
        }

        ctx.logger.info(f"🔄 Executing LangGraph workflow for {msg.company_name}")

        # Execute the LangGraph workflow
        final_state = finance_info_graph.invoke(initial_state)

        ctx.logger.info("LangGraph Workflow Completed")

        # Process the results
        if final_state["success"] and (
            final_state.get("overview") or final_state.get("financial_condition")
        ):
            ctx.logger.info(f"✅ Successfully extracted financial information!")
            ctx.logger.info(f"  Company: {msg.company_name}")
            ctx.logger.info(f"  Filing Type: {final_state['filing_type']}")
            ctx.logger.info(f"  Filing Date: {final_state.get('filing_date', 'N/A')}")

            # Build the financial data result
            financial_data = FinancialData(
                company_name=msg.company_name,
                cik=msg.cik,
                filing_type=final_state["filing_type"],
                filing_date=final_state.get("filing_date", ""),
                overview=final_state.get("overview", ""),
                financial_condition=final_state.get("financial_condition", ""),
                revenue_info=final_state.get("revenue_info", ""),
                legal_proceedings=final_state.get("legal_proceedings", ""),
                filing_url=final_state.get("latest_filing_url", ""),
                extraction_summary=f"Extracted financial data from {final_state['filing_type']} filing",
            )

            response = FinanceInfoResponse(
                request_id=msg.request_id,
                success=True,
                financial_data=financial_data,
                timestamp=datetime.now().isoformat(),
            )

            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("📤 FINANCE INFO AGENT: SENDING RESPONSE")
            ctx.logger.info("=" * 70)
            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: True")
            ctx.logger.info("=" * 70)

            await ctx.send(sender, response)

        else:
            # Handle errors
            error_msg = final_state.get(
                "error_message", "Failed to extract financial information"
            )
            ctx.logger.warning(f"❌ Failed to extract financial info: {error_msg}")

            error_response = FinanceInfoResponse(
                request_id=msg.request_id,
                success=False,
                error_message=error_msg,
                timestamp=datetime.now().isoformat(),
            )

            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("📤 FINANCE INFO AGENT: SENDING ERROR RESPONSE")
            ctx.logger.info("=" * 70)
            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: False")
            ctx.logger.info(f"Error: {error_msg}")
            ctx.logger.info("=" * 70)

            await ctx.send(sender, error_response)

    except Exception as e:
        ctx.logger.error(f"Error processing finance info request: {e}")
        import traceback

        traceback.print_exc()

        error_response = FinanceInfoResponse(
            request_id=msg.request_id,
            success=False,
            error_message=str(e),
            timestamp=datetime.now().isoformat(),
        )

        await ctx.send(sender, error_response)


finance_info_agent.include(finance_info_protocol, publish_manifest=True)

if __name__ == "__main__":
    finance_info_agent.run()
