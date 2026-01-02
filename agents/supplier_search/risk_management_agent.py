"""
Risk Management Agent - Extracts risk factors from SEC filings (10-Q and 20-F)
Uses LangGraph workflow to identify and extract risk information
"""

from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from models.risk_info import (
    RiskInfoRequest,
    RiskInfoResponse,
    RiskData,
    RiskInfoState,
)
from langgraph_logic.risk_info_langgraph import risk_info_graph
import os
import nest_asyncio
from datetime import datetime

nest_asyncio.apply()
load_dotenv()

risk_management_agent = Agent(
    name="risk_management_agent",
    seed=os.getenv("RISK_MANAGEMENT_SEED"),
    port=8009,
    mailbox=True,
)

risk_management_protocol = Protocol(name="risk_management_protocol", version="1.0")


@risk_management_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize risk management agent on startup"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("RISK MANAGEMENT AGENT STARTING UP")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Port: 8009")
    ctx.logger.info("Extracts risk factors from SEC filings (10-Q/20-F)")
    ctx.logger.info("Supported: Both US and Foreign Companies")
    ctx.logger.info("=" * 70)

    # Initialize storage
    ctx.storage.set("processed_requests", [])
    ctx.storage.set("request_count", 0)

    ctx.logger.info(
        "Risk Management Agent Ready - Listening for RiskInfoRequest messages..."
    )


@risk_management_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("RISK MANAGEMENT AGENT SHUTTING DOWN")
    ctx.logger.info("=" * 70)

    request_count = ctx.storage.get("request_count") or 0
    ctx.logger.info(f"Total requests processed: {request_count}")


@risk_management_protocol.on_message(model=RiskInfoRequest, replies=RiskInfoResponse)
async def handle_risk_info_request(
    ctx: Context, sender: str, msg: RiskInfoRequest
):
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 RISK MANAGEMENT AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Company: {msg.company_name}")
    ctx.logger.info(f"CIK: {msg.cik}")
    ctx.logger.info(f"Country: {msg.country}")
    ctx.logger.info("=" * 70)
    ctx.logger.info("")
    ctx.logger.info("🔍 Starting LangGraph Risk Factors Extraction...")
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
        initial_state: RiskInfoState = {
            "request_id": msg.request_id,
            "company_name": msg.company_name,
            "cik": msg.cik,
            "country": msg.country,
            "ticker": msg.ticker,
            "is_foreign": None,
            "filing_type": None,
            "latest_filing_url": None,
            "filing_date": None,
            "summary_risk_factors": None,
            "operational_risks": None,
            "financial_risks": None,
            "legal_regulatory_risks": None,
            "all_risk_factors": None,
            "success": False,
            "error_message": None,
            "score": 100,
            "ctx": ctx,
        }

        ctx.logger.info(f"🔄 Executing LangGraph workflow for {msg.company_name}")

        # Execute the LangGraph workflow
        final_state = risk_info_graph.invoke(initial_state)

        ctx.logger.info("LangGraph Workflow Completed")

        # Process the results
        if final_state["success"] and (
            final_state.get("summary_risk_factors") or final_state.get("all_risk_factors")
        ):
            ctx.logger.info(f"✅ Successfully extracted risk information!")
            ctx.logger.info(f"  Company: {msg.company_name}")
            ctx.logger.info(
                f"  Type: {'Foreign' if final_state['is_foreign'] else 'American'}"
            )
            ctx.logger.info(f"  Filing Type: {final_state['filing_type']}")
            ctx.logger.info(f"  Filing Date: {final_state.get('filing_date', 'N/A')}")
            ctx.logger.info(
                f"  Risk Factors Found: {len(final_state.get('all_risk_factors', []))}"
            )

            # Build the risk data result
            risk_data = RiskData(
                company_name=msg.company_name,
                cik=msg.cik,
                filing_type=final_state["filing_type"],
                filing_date=final_state.get("filing_date", ""),
                is_foreign=final_state["is_foreign"],
                summary_risk_factors=final_state.get("summary_risk_factors", ""),
                operational_risks=final_state.get("operational_risks", ""),
                financial_risks=final_state.get("financial_risks", ""),
                legal_regulatory_risks=final_state.get("legal_regulatory_risks", ""),
                all_risk_factors=final_state.get("all_risk_factors", []),
                filing_url=final_state.get("latest_filing_url", ""),
                extraction_summary=f"Extracted {len(final_state.get('all_risk_factors', []))} risk factors from {final_state['filing_type']} filing",
            )

            response = RiskInfoResponse(
                request_id=msg.request_id,
                success=True,
                risk_data=risk_data,
                timestamp=datetime.now().isoformat(),
            )

            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("📤 RISK MANAGEMENT AGENT: SENDING RESPONSE")
            ctx.logger.info("=" * 70)
            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: True")
            ctx.logger.info("=" * 70)

            await ctx.send(sender, response)

        else:
            # Handle errors
            error_msg = final_state.get("error_message") or "Failed to extract risk information"
            ctx.logger.warning(f"❌ Failed to extract risk info: {error_msg}")

            error_response = RiskInfoResponse(
                request_id=msg.request_id,
                success=False,
                error_message=error_msg,
                timestamp=datetime.now().isoformat(),
            )

            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("📤 RISK MANAGEMENT AGENT: SENDING ERROR RESPONSE")
            ctx.logger.info("=" * 70)
            ctx.logger.info(f"To: {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: False")
            ctx.logger.info(f"Error: {error_msg}")
            ctx.logger.info("=" * 70)

            await ctx.send(sender, error_response)

    except Exception as e:
        ctx.logger.error(f"Error processing risk info request: {e}")
        import traceback

        traceback.print_exc()

        error_response = RiskInfoResponse(
            request_id=msg.request_id,
            success=False,
            error_message=str(e),
            timestamp=datetime.now().isoformat(),
        )

        await ctx.send(sender, error_response)


risk_management_agent.include(risk_management_protocol, publish_manifest=True)

if __name__ == "__main__":
    risk_management_agent.run()
