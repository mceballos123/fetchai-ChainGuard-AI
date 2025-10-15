from uagents import Agent, Context, Protocol
from pydantic import Field
from datetime import datetime, UTC
from typing import Dict, List, Optional
import asyncio

from backend.models.client_message import ClientMessage, SupplierSearchRequest
from backend.models.supplier_recommendation import SupplierRecommendation, SupplierSearchResponse, ErrorResponse
from backend.models.financial_analyst import FinancialAnalysisRequest, FinancialAnalysisResponse
from backend.models.risk import SupplierRiskRequest, SupplierRiskResponse
from backend.models.compliance import ComplianceRequest, ComplianceResponse


# Set the addresses of the other agents when they are deployed
#Note: CHange it when the agents are deployed to asi:1
COMPLIANCE_AGENT_ADDRESS = ""  # Set when compliance agent is deployed
FINANCIAL_AGENT_ADDRESS = ""   # Set when financial agent is deployed
RISK_AGENT_ADDRESS = ""         # Set when risk agent is deployed

# Orchestrato agent that coordinates thge other agents and passes the request to the risk, complaince and fianical agents
supplier_orchestrator = Agent(
    name = 'supplier_orchestrator',
    seed = 'supplier_orchestrator_key_56321_2025',
    mailbox = True
)

# Protocal for the orechestrato agent to communicate with the other agents

supplier_orchestrator_protocol = Protocol(name = 'supplier_orchestrator_protocol',version ="1.0")

@supplier_orchestrator.on_event("startup")
async def startup(ctx:Context):
    ctx.logger.info("Starting up supplier orchestrator agent")
    ctx.storage.set("active_requests", {})

@supplier_orchestrator_protocol.on_event("shutdown")
async def shutdown(ctx:Context):
    ctx.logger.info("Shutting down supplier orchestrator agent")


# Tells orchestrator agent to pass the request to the risk, complaince and fianical agents
@supplier_orchestrator_protocol.on_message(model = SupplierSearchRequest, replies=[SupplierSearchResponse, ErrorResponse])
async def handle_supplier_search_request(ctx:Context, sender:str, msg:SupplierSearchRequest):
    ctx.logger.info(f"Received supplier search request from {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Client requirements: {msg.client_requirements}")

    try:
        request_state = {
            "request_id": msg.request_id,
            "sender":sender,
            "client_requirements":msg.client_requirements,
            "responses": {},
            "start_time": datetime.now(UTC).isoformat(),
        }

        active_requests = ctx.storage.get("active_requests") or {}
        active_requests[msg.request_id] = request_state
        ctx.storage.set("active_requests", active_requests)

        # To do pass the request ot the risk, complaince and finacial agents but after I implement the agent first
        # pasas langgraph langchain for this since it requires the documents and the information from the documents to be passed to the agent
    except Exception as e:
        ctx.logger.error(f"Error handling supplier search request: {e}")
        error_response = ErrorResponse(request_id=msg.request_id, error=str(e), error_type="internal_error")
        await ctx.send(sender, error_response) 

    