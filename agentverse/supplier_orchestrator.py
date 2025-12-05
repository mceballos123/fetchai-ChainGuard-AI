from uuid import uuid4
from uagents import Agent, Context, Protocol
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Dict, List, Any
from supabase import create_client, Client

# backend/agents/supplier_search
# Import chat protocol components
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
    chat_protocol_spec,
)

# Import compliance, financial, risk, performance, demand, and logistics models
from models.compliance import ComplianceRequest, ComplianceResponse
from models.financial import FinancialRequest, FinancialResponse
from models.risk import RiskRequest, RiskResponse
from models.performance import PerformanceRequest, PerformanceResponse
from models.demand import DemandRequest, DemandResponse
from models.logistics import LogisticsRequest, LogisticsResponse
from models.find_supplier import FindSupplierRequest, FindSupplierResponse

# Import test utilities
from test_func.orchestrator_helpers import (
    verify_orchestrator_connection,
    log_message_transmission,
    validate_request_before_sending,
)

load_dotenv()

SUPPLIER_ORCHESTRATOR_SEED = os.getenv("SUPPLIER_ORCHESTRATOR_SEED")


def determine_product_category(user_query: str) -> str:
    """
    Determine the product category from the user query.

    Maps business keywords to product categories:
    - food: coffee, pizza, bakery, restaurant, tea, chocolate, food, grocery, cafe
    - clothing: clothing, apparel, fashion, textile, garment, shoes, accessories
    - electronics: technology, software, electronics, computer, phone, hardware

    Args:
        user_query: The user's search query

    Returns:
        Product category: "food", "clothing", or "electronics"
    """
    query_lower = user_query.lower()

    # Food category keywords
    food_keywords = [
        "coffee",
        "pizza",
        "bakery",
        "restaurant",
        "tea",
        "chocolate",
        "food",
        "grocery",
        "cafe",
        "catering",
        "beverage",
        "organic",
        "farm",
        "produce",
        "meat",
        "dairy",
        "snack",
        "juice",
        "wine",
        "beer",
        "bread",
        "pasta",
        "sauce",
        "spice",
        "ingredient",
    ]

    # Clothing category keywords
    clothing_keywords = [
        "clothing",
        "apparel",
        "fashion",
        "textile",
        "garment",
        "shoes",
        "accessories",
        "wear",
        "fabric",
        "leather",
        "denim",
        "cotton",
        "wool",
        "silk",
        "uniform",
        "sportswear",
        "footwear",
        "bag",
        "hat",
    ]

    # Electronics category keywords
    electronics_keywords = [
        "technology",
        "software",
        "electronics",
        "computer",
        "phone",
        "hardware",
        "device",
        "gadget",
        "tech",
        "digital",
        "smart",
        "appliance",
        "machine",
        "equipment",
        "component",
        "chip",
    ]

    # Check for matches
    for keyword in food_keywords:
        if keyword in query_lower:
            return "food"

    for keyword in clothing_keywords:
        if keyword in query_lower:
            return "clothing"

    for keyword in electronics_keywords:
        if keyword in query_lower:
            return "electronics"

    # Default to food if no match
    return "food"


def extract_user_country(user_query: str) -> str:
    """
    Extract the user's country from their query.

    Looks for patterns like:
    - "I'm a United Kingdom business owner"
    - "I'm in Germany"
    - "I'm from Spain"
    - "I am a UK business"
    - "I'm a German company"

    Args:
        user_query: The user's search query

    Returns:
        Country name (default: "United States")
    """
    query_lower = user_query.lower()

    # List of countries to look for (with variations)
    countries = {
        # UK variations
        "united kingdom": "United Kingdom",
        "uk": "United Kingdom",
        "british": "United Kingdom",
        "britain": "United Kingdom",
        "england": "United Kingdom",
        "scotland": "United Kingdom",
        "wales": "United Kingdom",
        # US variations
        "united states": "United States",
        "usa": "United States",
        "us": "United States",
        "american": "United States",
        "america": "United States",
        # EU countries
        "germany": "Germany",
        "german": "Germany",
        "france": "France",
        "french": "France",
        "spain": "Spain",
        "spanish": "Spain",
        "italy": "Italy",
        "italian": "Italy",
        "netherlands": "Netherlands",
        "dutch": "Netherlands",
        "holland": "Netherlands",
        "belgium": "Belgium",
        "belgian": "Belgium",
        "portugal": "Portugal",
        "portuguese": "Portugal",
        "austria": "Austria",
        "austrian": "Austria",
        "ireland": "Ireland",
        "irish": "Ireland",
        "sweden": "Sweden",
        "swedish": "Sweden",
        "norway": "Norway",
        "norwegian": "Norway",
        "denmark": "Denmark",
        "danish": "Denmark",
        "finland": "Finland",
        "finnish": "Finland",
        "poland": "Poland",
        "polish": "Poland",
        "greece": "Greece",
        "greek": "Greece",
        "czech": "Czech Republic",
        "hungary": "Hungary",
        "hungarian": "Hungary",
        "romania": "Romania",
        "romanian": "Romania",
        # Other developed economies
        "canada": "Canada",
        "canadian": "Canada",
        "australia": "Australia",
        "australian": "Australia",
        "new zealand": "New Zealand",
        "japan": "Japan",
        "japanese": "Japan",
        "south korea": "South Korea",
        "korean": "South Korea",
        "singapore": "Singapore",
        "singaporean": "Singapore",
        "switzerland": "Switzerland",
        "swiss": "Switzerland",
        # Latin America
        "mexico": "Mexico",
        "mexican": "Mexico",
        "brazil": "Brazil",
        "brazilian": "Brazil",
        "argentina": "Argentina",
        "argentinian": "Argentina",
        "chile": "Chile",
        "chilean": "Chile",
        "colombia": "Colombia",
        "colombian": "Colombia",
        "peru": "Peru",
        "peruvian": "Peru",
        # Asia
        "china": "China",
        "chinese": "China",
        "india": "India",
        "indian": "India",
        "taiwan": "Taiwan",
        "taiwanese": "Taiwan",
        "thailand": "Thailand",
        "thai": "Thailand",
        "vietnam": "Vietnam",
        "vietnamese": "Vietnam",
        "indonesia": "Indonesia",
        "indonesian": "Indonesia",
        "malaysia": "Malaysia",
        "malaysian": "Malaysia",
        "philippines": "Philippines",
        "filipino": "Philippines",
        # Middle East / Africa
        "israel": "Israel",
        "israeli": "Israel",
        "united arab emirates": "United Arab Emirates",
        "uae": "United Arab Emirates",
        "dubai": "United Arab Emirates",
        "saudi arabia": "Saudi Arabia",
        "saudi": "Saudi Arabia",
        "south africa": "South Africa",
        "south african": "South Africa",
        "egypt": "Egypt",
        "egyptian": "Egypt",
        "morocco": "Morocco",
        "moroccan": "Morocco",
        "kenya": "Kenya",
        "kenyan": "Kenya",
        "nigeria": "Nigeria",
        "nigerian": "Nigeria",
        # Other
        "turkey": "Turkey",
        "turkish": "Turkey",
        "russia": "Russia",
        "russian": "Russia",
        "ukraine": "Ukraine",
        "ukrainian": "Ukraine",
    }

    # Check for country mentions (longer matches first to avoid partial matches)
    # Sort by length descending to check longer matches first
    sorted_countries = sorted(countries.keys(), key=len, reverse=True)

    for country_key in sorted_countries:
        if country_key in query_lower:
            return countries[country_key]

    # Default to United States if no country found
    return "United States"


# Supabase configuration for monitoring data storage
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Global Supabase client (initialized at startup)
# Note: We use a global variable because ctx.storage only accepts JSON-serializable data
supabase_client: Client | None = None

supplier_orchestrator = Agent(
    name="supplier_orchestrator",
    seed=SUPPLIER_ORCHESTRATOR_SEED,
    port=8000,
    mailbox=True,
)

chat_proto = Protocol(name="chat_protocol", spec=chat_protocol_spec)

COMPLIANCE_AGENT_ADDRESS = os.getenv(
    "COMPLIANCE_AGENT_ADDRESS",
)
FINANCIAL_AGENT_ADDRESS = os.getenv(
    "FINANCIAL_AGENT_ADDRESS",
)
RISK_AGENT_ADDRESS = os.getenv(
    "RISK_AGENT_ADDRESS",
)
PERFORMANCE_AGENT_ADDRESS = os.getenv(
    "PERFORMANCE_AGENT_ADDRESS",
)
DEMAND_AGENT_ADDRESS = os.getenv(
    "DEMAND_AGENT_ADDRESS",
)
LOGISTICS_AGENT_ADDRESS = os.getenv(
    "LOGISTICS_AGENT_ADDRESS",
)
FIND_SUPPLIER_AGENT_ADDRESS = os.getenv(
    "FIND_SUPPLIER_ADDRESS",
)

orchestrator_protocol = Protocol(name="supplier_orchestrator_protocol", version="1.0")


@supplier_orchestrator.on_event("startup")
async def startup(ctx: Context):
    """Initialize orchestrator on startup"""
    ctx.logger.info("Supplier Orchestrator Agent Starting Up")

    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Find Supplier Agent: {FIND_SUPPLIER_AGENT_ADDRESS}")
    ctx.logger.info(f"Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
    ctx.logger.info(f"Financial Agent: {FINANCIAL_AGENT_ADDRESS}")
    ctx.logger.info(f"Risk Agent: {RISK_AGENT_ADDRESS}")
    ctx.logger.info(f"Performance Agent: {PERFORMANCE_AGENT_ADDRESS}")
    ctx.logger.info(f"Demand Agent: {DEMAND_AGENT_ADDRESS}")
    ctx.logger.info(f"Logistics Agent: {LOGISTICS_AGENT_ADDRESS}")

    # Initialize Supabase client for monitoring data storage
    global supabase_client
    ctx.logger.info(f"Supabase URL: {SUPABASE_URL}")
    ctx.logger.info(f"Supabase Key: {SUPABASE_KEY[:20]}..." if SUPABASE_KEY else "None")
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            ctx.logger.info("✅ Supabase client initialized successfully")
        except Exception as e:
            ctx.logger.error(f"❌ Failed to initialize Supabase client: {e}")
            supabase_client = None
    else:
        ctx.logger.warning(
            "⚠️ Supabase credentials not found - monitoring data will not be stored"
        )
        supabase_client = None

    ctx.storage.set("active_sessions", {})
    ctx.storage.set("approved_suppliers", [])
    ctx.storage.set(
        "selected_supplier", None
    )  # Track the supplier selected for monitoring
    ctx.storage.set(
        "selected_product_category", None
    )  # Track the product category (food, clothing, electronics)

    # Storage for tracking responses from both agents
    ctx.storage.set("pending_responses", {})

    # Initialize message trace for debugging
    ctx.storage.set(
        "message_trace",
        {
            "received_from_asi": [],
            "sent_to_compliance": [],
            "received_from_compliance": [],
            "sent_to_asi": [],
        },
    )

    # Verify connection on startup
    conn_status = verify_orchestrator_connection(ctx)
    ctx.logger.info(f"Connection Status: {conn_status['status']}")


@supplier_orchestrator.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Supplier Orchestrator Agent Shutting Down")


@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle chat messages from user via ASI:1"""

    # Step 0: Verify and log message received from ASI:1
    log_message_transmission(
        ctx, "RECEIVED", "ChatMessage", str(msg.msg_id), {"sender": sender}
    )

    # Step 1: Send acknowledgment immediately
    ctx.logger.info("Sending acknowledgment...")
    ack = ChatAcknowledgement(
        timestamp="",
        acknowledged_msg_id=msg.msg_id,
    )
    await ctx.send(sender, ack)

    # Step 2: Extract user query from message content
    ctx.logger.info(f"Received ChatMessage from {sender}")
    user_query = None

    for content_item in msg.content:
        if isinstance(content_item, TextContent):
            user_query = content_item.text
            break

    if not user_query:
        ctx.logger.warning("No text content in message")
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text="Error: No query text provided in message",
                )
            ],
        )
        await ctx.send(sender, error_response)
        log_message_transmission(
            ctx, "SENT", "ChatMessage", str(error_response.msg_id), {"type": "error"}
        )
        return

    ctx.logger.info(f"User Query: {user_query}")

    # Step 3: Determine if this is a monitoring request or find supplier request
    user_query_lower = user_query.lower()
    monitoring_keywords = [
        "monitor",
        "update",
        "status",
        "check on",
        "check up",
        "how is",
        "report on",
    ]
    is_monitoring_request = any(
        keyword in user_query_lower for keyword in monitoring_keywords
    )

    # Step 4: Store session information
    msg_id = str(msg.msg_id)
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions[msg_id] = {
        "sender": sender,
        "query": user_query,
        "mode": "monitor" if is_monitoring_request else "find",
    }
    ctx.storage.set("active_sessions", active_sessions)
    ctx.logger.info(f"Session stored with ID: {msg_id}")
    ctx.logger.info(
        f"Mode detected: {'MONITORING' if is_monitoring_request else 'FIND SUPPLIER'}"
    )

    # Step 5: Route based on mode
    if is_monitoring_request:
        # MONITORING MODE: Forward to Performance, Demand, and Logistics Agents (ALL IN PARALLEL)
        ctx.logger.info("=" * 70)
        ctx.logger.info(
            "MONITORING MODE - FORWARDING TO PERFORMANCE, DEMAND, AND LOGISTICS AGENTS"
        )
        ctx.logger.info("=" * 70)

        # Get the selected supplier and product category (from the find_supplier phase)
        selected_supplier = ctx.storage.get("selected_supplier")
        product_category = ctx.storage.get("selected_product_category") or "food"

        if not selected_supplier:
            ctx.logger.warning("No selected supplier available for monitoring")
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text="Error: No supplier selected for monitoring. Please run the find_supplier feature first to select a supplier.",
                    )
                ],
            )
            await ctx.send(sender, error_response)
            log_message_transmission(
                ctx,
                "SENT",
                "ChatMessage",
                str(error_response.msg_id),
                {"type": "error"},
            )
            return

        # Initialize pending responses for monitoring (all 3 monitoring agents)
        pending_responses = ctx.storage.get("pending_responses") or {}
        pending_responses[msg_id] = {
            "performance_response": None,
            "demand_response": None,
            "logistics_response": None,
            "sender": sender,
            "user_query": user_query,
            "mode": "monitor",
        }
        ctx.storage.set("pending_responses", pending_responses)

        try:
            # Send to Performance Agent with the selected supplier and product category
            ctx.logger.info(
                f"Sending to Performance Agent: {PERFORMANCE_AGENT_ADDRESS}"
            )
            ctx.logger.info(f"Monitoring supplier: {selected_supplier}")
            ctx.logger.info(f"Product category: {product_category}")
            performance_request = PerformanceRequest(
                request_id=msg_id,
                supplier_name=selected_supplier,
                product_category=product_category,
                timestamp="",
            )

            await ctx.send(PERFORMANCE_AGENT_ADDRESS, performance_request)

            ctx.logger.info(f"PerformanceRequest sent to Performance Agent")

            # Send to Demand Agent with the selected supplier and product category
            ctx.logger.info(f"Sending to Demand Agent: {DEMAND_AGENT_ADDRESS}")
            ctx.logger.info(f"Monitoring supplier: {selected_supplier}")
            ctx.logger.info(f"Product category: {product_category}")
            demand_request = DemandRequest(
                request_id=msg_id,
                supplier_name=selected_supplier,
                product_category=product_category,
                timestamp="",
            )

            await ctx.send(DEMAND_AGENT_ADDRESS, demand_request)

            ctx.logger.info(f"DemandRequest sent to Demand Agent")

            # Send to Logistics Agent with the selected supplier and product category
            ctx.logger.info(f"Sending to Logistics Agent: {LOGISTICS_AGENT_ADDRESS}")
            ctx.logger.info(f"Monitoring supplier: {selected_supplier}")
            ctx.logger.info(f"Product category: {product_category}")
            logistics_request = LogisticsRequest(
                request_id=msg_id,
                supplier_name=selected_supplier,
                product_category=product_category,
                timestamp="",
            )

            await ctx.send(LOGISTICS_AGENT_ADDRESS, logistics_request)

            ctx.logger.info(f"LogisticsRequest sent to Logistics Agent")
            ctx.logger.info("=" * 70)
            ctx.logger.info(
                "WAITING FOR PERFORMANCE, DEMAND, AND LOGISTICS RESPONSES..."
            )
            ctx.logger.info("=" * 70)

        except Exception as e:
            ctx.logger.error(f"Error sending to monitoring agents: {e}")

            # Clean up pending responses
            pending_responses.pop(msg_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            # Send error response to user
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Error forwarding monitoring request: {str(e)}",
                    )
                ],
            )
            await ctx.send(sender, error_response)
            log_message_transmission(
                ctx,
                "SENT",
                "ChatMessage",
                str(error_response.msg_id),
                {"type": "error"},
            )

        return

    # Step 6: FIND SUPPLIER MODE - First forward to find_supplier agent
    ctx.logger.info("=" * 70)
    ctx.logger.info("FIND SUPPLIER MODE - FORWARDING TO FIND SUPPLIER AGENT")
    ctx.logger.info("=" * 70)

    # Initialize pending responses tracking for this request
    pending_responses = ctx.storage.get("pending_responses") or {}
    pending_responses[msg_id] = {
        "find_supplier_response": None,
        "compliance_response": None,
        "financial_response": None,
        "risk_response": None,
        "sender": sender,
        "user_query": user_query,
        "mode": "find",
    }
    ctx.storage.set("pending_responses", pending_responses)

    # Extract business category from user query
    user_query_lower = user_query.lower()
    business_category = "general"

    # Try to identify business category
    category_keywords = {
        "coffee": "coffee",
        "pizza": "pizza",
        "chocolate": "chocolate",
        "food": "food",
        "restaurant": "restaurant",
        "clothing": "clothing",
        "apparel": "clothing",
        "fashion": "fashion",
        "tea": "tea",
        "bakery": "bakery",
        "furniture": "furniture",
        "technology": "technology",
        "software": "software",
    }

    for keyword, category in category_keywords.items():
        if keyword in user_query_lower:
            business_category = category
            break

    try:
        # Send to Find Supplier Agent first
        ctx.logger.info(
            f"Sending to Find Supplier Agent: {FIND_SUPPLIER_AGENT_ADDRESS}"
        )
        find_supplier_request = FindSupplierRequest(
            request_id=msg_id,
            user_query=user_query,
            business_category=business_category,
            timestamp="",
        )

        await ctx.send(FIND_SUPPLIER_AGENT_ADDRESS, find_supplier_request)

        log_message_transmission(
            ctx,
            "SENT",
            "FindSupplierRequest",
            msg_id,
            {
                "user_query": user_query,
                "business_category": business_category,
            },
        )

        ctx.logger.info(f"FindSupplierRequest sent to Find Supplier Agent")
        ctx.logger.info(f"Category: {business_category}")
        ctx.logger.info("=" * 70)
        ctx.logger.info("WAITING FOR FIND SUPPLIER RESPONSE...")
        ctx.logger.info("=" * 70)

        # NOTE: We'll wait for find_supplier response before forwarding to the 3 analysis agents
        # For now, we just send to find_supplier and will handle the response separately

    except Exception as e:
        ctx.logger.error(f"Error sending to find_supplier agent: {e}")

        # Clean up pending responses
        pending_responses.pop(msg_id, None)
        ctx.storage.set("pending_responses", pending_responses)

        # Send error response to user
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text=f"Error forwarding request to find_supplier agent: {str(e)}",
                )
            ],
        )
        await ctx.send(sender, error_response)
        log_message_transmission(
            ctx, "SENT", "ChatMessage", str(error_response.msg_id), {"type": "error"}
        )

    # COMMENTED OUT FOR NOW - Will send to these agents after receiving find_supplier response
    """
    try:
        # Send to Compliance Agent
        ctx.logger.info(f"Sending to Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
        compliance_request = ComplianceRequest(
            request_id=msg_id,
            supplier_name="sunrise_sustainable",
            industry="general",
            company_values=user_query,
            timestamp="",
        )

        # Validate request before sending
        is_valid, error_msg = validate_request_before_sending(ctx, compliance_request)
        if not is_valid:
            raise ValueError(f"Compliance request validation failed: {error_msg}")

        await ctx.send(COMPLIANCE_AGENT_ADDRESS, compliance_request)

        # Log transmission
        log_message_transmission(
            ctx,
            "SENT",
            "ComplianceRequest",
            msg_id,
            {
                "supplier_name": compliance_request.supplier_name,
                "industry": compliance_request.industry,
            },
        )

        ctx.logger.info(f"ComplianceRequest sent to Compliance Agent")

        # Send to Financial Agent
        ctx.logger.info(f"Sending to Financial Agent: {FINANCIAL_AGENT_ADDRESS}")
        financial_request = FinancialRequest(
            request_id=msg_id,
            supplier_name="sunrise_sustainable",
            industry="general",
            timestamp="",
        )

        await ctx.send(FINANCIAL_AGENT_ADDRESS, financial_request)

        log_message_transmission(
            ctx,
            "SENT",
            "FinancialRequest",
            msg_id,
            {
                "supplier_name": financial_request.supplier_name,
                "industry": financial_request.industry,
            },
        )

        ctx.logger.info(f"FinancialRequest sent to Financial Agent")

        # Send to Risk Agent
        ctx.logger.info(f"Sending to Risk Agent: {RISK_AGENT_ADDRESS}")
        risk_request = RiskRequest(
            request_id=msg_id,
            supplier_name="sunrise_sustainable",
            industry="general",
            timestamp="",
        )

        await ctx.send(RISK_AGENT_ADDRESS, risk_request)

        log_message_transmission(
            ctx,
            "SENT",
            "RiskRequest",
            msg_id,
            {
                "supplier_name": risk_request.supplier_name,
                "industry": risk_request.industry,
            },
        )

        ctx.logger.info(f"RiskRequest sent to Risk Agent")
        ctx.logger.info("=" * 70)
        ctx.logger.info("WAITING FOR ALL RESPONSES (Compliance, Financial, Risk)...")
        ctx.logger.info("=" * 70)

    except Exception as e:
        ctx.logger.error(f"Error sending to agents: {e}")

        # Clean up pending responses
        pending_responses.pop(msg_id, None)
        ctx.storage.set("pending_responses", pending_responses)

        # Send error response to user
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text=f"Error forwarding request to agents: {str(e)}",
                )
            ],
        )
        await ctx.send(sender, error_response)
        log_message_transmission(
            ctx, "SENT", "ChatMessage", str(error_response.msg_id), {"type": "error"}
        )
    """


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgment messages from user"""
    ctx.logger.info(
        f"Acknowledgment received from {sender} for message: {msg.acknowledged_msg_id}"
    )


compliance_protocol = Protocol(name="compliance_response_protocol", version="1.0")


@compliance_protocol.on_message(model=ComplianceResponse)
async def handle_compliance_response(
    ctx: Context, sender: str, msg: ComplianceResponse
):
    """Handle compliance response and wait for other responses before sending to user"""
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📨 [1/3] RECEIVED COMPLIANCE RESPONSE")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier: {msg.supplier_name}")
    ctx.logger.info(f"Compliance Score: {msg.compliance_score}/100")
    ctx.logger.info(f"Violations Count: {len(msg.violations)}")
    ctx.logger.info("=" * 70)

    # Verify message received from Compliance Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "ComplianceResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "compliance_score": msg.compliance_score,
        },
    )

    try:
        # Store compliance response in pending_responses
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["compliance_response"] = msg.model_dump()
        pending_responses[msg.request_id][
            "status"
        ] = "compliance_received"  # Update status
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Compliance response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Score: {msg.compliance_score}/100")

        # SEQUENTIAL: Now trigger Financial Agent (Step 2/3)
        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("✅ COMPLIANCE COMPLETE - MOVING TO STEP 2: FINANCIAL")
        ctx.logger.info("=" * 70)

        # Get supplier info from stored data
        best_supplier_data = pending_responses[msg.request_id].get("best_supplier")
        user_query = pending_responses[msg.request_id].get("user_query")

        if not best_supplier_data:
            ctx.logger.error(
                "No supplier data found, cannot proceed to financial agent"
            )
            return

        # Send to Financial Agent
        ctx.logger.info("📤 [STEP 2/3] SENDING TO FINANCIAL AGENT")
        ctx.logger.info(f"Agent Address: {FINANCIAL_AGENT_ADDRESS}")
        ctx.logger.info(f"Request ID: {msg.request_id}")

        # Financial agent will scrape B Corp URL to extract supplier country
        b_corp_url = best_supplier_data.get("b_corp_profile_url", "")
        ctx.logger.info(
            f"B Corp URL (will be scraped for supplier country): {b_corp_url}"
        )

        # Extract user's country from their query (e.g., "I'm a UK business owner")
        user_country = extract_user_country(user_query or "")
        ctx.logger.info(f"User Country (extracted from query): {user_country}")

        financial_request = FinancialRequest(
            request_id=msg.request_id,
            supplier_name=best_supplier_data.get("company_name"),
            industry=best_supplier_data.get("industry", "general"),
            b_corp_profile_url=b_corp_url,
            user_country=user_country,  # User's country extracted from query
            timestamp="",
        )

        ctx.logger.info(f"Financial Request Details:")
        ctx.logger.info(f"  - Supplier: {financial_request.supplier_name}")
        ctx.logger.info(f"  - B Corp URL: {financial_request.b_corp_profile_url}")
        ctx.logger.info(f"  - Industry: {financial_request.industry}")
        ctx.logger.info(f"  - User Country: {financial_request.user_country}")
        ctx.logger.info(f"  - Supplier Country will be scraped from B Corp page")

        await ctx.send(FINANCIAL_AGENT_ADDRESS, financial_request)

        log_message_transmission(
            ctx,
            "SENT",
            "FinancialRequest",
            msg.request_id,
            {
                "supplier_name": financial_request.supplier_name,
                "b_corp_profile_url": financial_request.b_corp_profile_url,
                "industry": financial_request.industry,
                "user_country": financial_request.user_country,
            },
        )

        ctx.logger.info(f"✅ FinancialRequest successfully sent to Financial Agent")
        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("⏳ WAITING FOR FINANCIAL RESPONSE...")
        ctx.logger.info("=" * 70)
        ctx.logger.info("Progress:")
        ctx.logger.info("  [✓] Compliance Agent - COMPLETE")
        ctx.logger.info("  [ ] Financial Agent - IN PROGRESS")
        ctx.logger.info("  [ ] Risk Agent - WAITING")
        ctx.logger.info("=" * 70)

    except Exception as e:
        ctx.logger.error(f"Error handling compliance response: {e}")
        import traceback

        traceback.print_exc()


# Add Financial Response Protocol
financial_protocol = Protocol(name="financial_response_protocol", version="1.0")

# Add Risk Response Protocol
risk_protocol = Protocol(name="risk_response_protocol", version="1.0")

# Add Performance Response Protocol
performance_protocol = Protocol(name="performance_response_protocol", version="1.0")

# Add Demand Response Protocol
demand_protocol = Protocol(name="demand_response_protocol", version="1.0")

# Add Logistics Response Protocol
logistics_protocol = Protocol(name="logistics_response_protocol", version="1.0")

# Add Find Supplier Response Protocol
find_supplier_protocol = Protocol(name="find_supplier_response_protocol", version="1.0")


@find_supplier_protocol.on_message(model=FindSupplierResponse)
async def handle_find_supplier_response(
    ctx: Context, sender: str, msg: FindSupplierResponse
):
    """Handle find supplier response and forward to 3 analysis agents in parallel"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 RECEIVED FIND SUPPLIER RESPONSE")
    ctx.logger.info("=" * 70)

    log_message_transmission(
        ctx,
        "RECEIVED",
        "FindSupplierResponse",
        msg.request_id,
        {
            "success": msg.success,
            "search_category": msg.search_category,
            "total_results": msg.total_results_found,
        },
    )

    try:
        # Get pending response tracking
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        # Store find_supplier response
        pending_responses[msg.request_id]["find_supplier_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        user_sender = pending_responses[msg.request_id].get("sender")
        user_query = pending_responses[msg.request_id].get("user_query")

        if not user_sender:
            ctx.logger.error(f"No sender found for request {msg.request_id}")
            return

        # Check if search was successful
        if not msg.success:
            ctx.logger.error(f"❌ Find supplier search failed: {msg.error_message}")

            # Send error message to user
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"❌ Supplier Search Failed\n\nCategory: {msg.search_category}\nError: {msg.error_message}\n\nPlease try a different search term.",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            await ctx.send(user_sender, error_response)

            # Clean up
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            return

        # Search was successful
        best_supplier = msg.best_supplier

        if not best_supplier:
            ctx.logger.warning("⚠️ No supplier found in successful response")

            no_results_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"⚠️ No Suppliers Found\n\nCategory: {msg.search_category}\nWe couldn't find any B Corporation certified companies matching your search.\n\nPlease try a different category or search term.",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            await ctx.send(user_sender, no_results_response)

            # Clean up
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            return

        # Supplier found successfully - Store it and forward to analysis agents
        ctx.logger.info("=" * 70)
        ctx.logger.info("✅ SUPPLIER SUCCESSFULLY FOUND")
        ctx.logger.info("=" * 70)
        ctx.logger.info(f"Supplier Name: {best_supplier.company_name}")
        ctx.logger.info(f"Location: {best_supplier.location}")
        ctx.logger.info(f"Industry: {best_supplier.industry}")
        ctx.logger.info(f"B Corp Profile: {best_supplier.b_corp_profile_url}")
        ctx.logger.info(f"Description: {best_supplier.description}")
        ctx.logger.info(f"Total Results Found: {msg.total_results_found}")
        ctx.logger.info(f"Search Category: {msg.search_category}")
        ctx.logger.info("=" * 70)

        # Store the selected supplier for monitoring
        ctx.storage.set("selected_supplier", best_supplier.company_name)

        # Determine and store product category from user query
        product_category = determine_product_category(user_query)
        ctx.storage.set("selected_product_category", product_category)

        ctx.logger.info(
            f"✅ Stored supplier for monitoring: {best_supplier.company_name}"
        )
        ctx.logger.info(f"✅ Product category determined: {product_category}")

        # NOW FORWARD TO COMPLIANCE AGENT FIRST (SEQUENTIAL PROCESSING)
        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("🚀 STARTING SEQUENTIAL ANALYSIS - STEP 1: COMPLIANCE")
        ctx.logger.info("=" * 70)
        ctx.logger.info("Sequential Analysis Order:")
        ctx.logger.info(f"  1. Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
        ctx.logger.info(
            f"  2. Financial Agent: {FINANCIAL_AGENT_ADDRESS} (after compliance)"
        )
        ctx.logger.info(f"  3. Risk Agent: {RISK_AGENT_ADDRESS} (after financial)")
        ctx.logger.info("=" * 70)

        # Store supplier info for subsequent agents
        pending_responses[msg.request_id]["best_supplier"] = best_supplier.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        try:
            # Send ONLY to Compliance Agent first
            ctx.logger.info("")
            ctx.logger.info("📤 [STEP 1/3] SENDING TO COMPLIANCE AGENT")
            ctx.logger.info(f"Agent Address: {COMPLIANCE_AGENT_ADDRESS}")
            ctx.logger.info(f"Request ID: {msg.request_id}")

            compliance_request = ComplianceRequest(
                request_id=msg.request_id,
                supplier_name=best_supplier.company_name,
                industry=best_supplier.industry or "general",
                company_values=user_query,
                b_corp_profile_url=best_supplier.b_corp_profile_url or "",
                timestamp="",
            )

            ctx.logger.info(f"Compliance Request Details:")
            ctx.logger.info(f"  - Supplier: {compliance_request.supplier_name}")
            ctx.logger.info(f"  - Industry: {compliance_request.industry}")
            ctx.logger.info(
                f"  - Company Values: {compliance_request.company_values[:50]}..."
            )
            ctx.logger.info(f"  - B Corp URL: {compliance_request.b_corp_profile_url}")

            # Validate request before sending
            is_valid, error_msg = validate_request_before_sending(
                ctx, compliance_request
            )
            if not is_valid:
                raise ValueError(f"Compliance request validation failed: {error_msg}")

            await ctx.send(COMPLIANCE_AGENT_ADDRESS, compliance_request)

            # Log transmission
            log_message_transmission(
                ctx,
                "SENT",
                "ComplianceRequest",
                msg.request_id,
                {
                    "supplier_name": compliance_request.supplier_name,
                    "industry": compliance_request.industry,
                },
            )

            ctx.logger.info(
                f"✅ ComplianceRequest successfully sent to Compliance Agent"
            )
            ctx.logger.info("")
            ctx.logger.info("=" * 70)
            ctx.logger.info("⏳ WAITING FOR COMPLIANCE RESPONSE...")
            ctx.logger.info("=" * 70)
            ctx.logger.info("Next steps after compliance completes:")
            ctx.logger.info("  [ ] Compliance Agent - IN PROGRESS")
            ctx.logger.info("  [ ] Financial Agent - WAITING")
            ctx.logger.info("  [ ] Risk Agent - WAITING")
            ctx.logger.info("=" * 70)

        except Exception as e:
            ctx.logger.error(f"Error sending to analysis agents: {e}")
            import traceback

            traceback.print_exc()

            # Clean up pending responses
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            # Send error response to user
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Error forwarding supplier to analysis agents: {str(e)}",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )
            await ctx.send(user_sender, error_response)
            log_message_transmission(
                ctx,
                "SENT",
                "ChatMessage",
                str(error_response.msg_id),
                {"type": "error"},
            )

    except Exception as e:
        ctx.logger.error(f"❌ Error handling find supplier response: {e}")
        import traceback

        traceback.print_exc()


@financial_protocol.on_message(model=FinancialResponse)
async def handle_financial_response(ctx: Context, sender: str, msg: FinancialResponse):
    """Handle financial response and wait for other responses before sending to user"""
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📨 [2/3] RECEIVED FINANCIAL RESPONSE")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier: {msg.supplier_name}")
    ctx.logger.info(f"Financial Score: {msg.financial_score}/100")
    ctx.logger.info(f"Risk Factors Count: {len(msg.risk_factors)}")
    ctx.logger.info("=" * 70)

    # Verify message received from Financial Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "FinancialResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "financial_score": msg.financial_score,
        },
    )

    try:
        # Store financial response in pending_responses
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["financial_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Financial response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Score: {msg.financial_score}/100")

        # SEQUENTIAL: Now trigger Risk Agent (Step 3/3)
        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("✅ FINANCIAL COMPLETE - MOVING TO STEP 3: RISK")
        ctx.logger.info("=" * 70)

        # Get supplier info from stored data
        best_supplier_data = pending_responses[msg.request_id].get("best_supplier")

        if not best_supplier_data:
            ctx.logger.error("No supplier data found, cannot proceed to risk agent")
            return

        # Send to Risk Agent
        ctx.logger.info("📤 [STEP 3/3] SENDING TO RISK AGENT")
        ctx.logger.info(f"Agent Address: {RISK_AGENT_ADDRESS}")
        ctx.logger.info(f"Request ID: {msg.request_id}")

        risk_request = RiskRequest(
            request_id=msg.request_id,
            supplier_name=best_supplier_data.get("company_name"),
            industry=best_supplier_data.get("industry", "general"),
            b_corp_profile_url=best_supplier_data.get("b_corp_profile_url"),
            timestamp="",
        )

        ctx.logger.info(f"Risk Request Details:")
        ctx.logger.info(f"  - Supplier: {risk_request.supplier_name}")
        ctx.logger.info(f"  - Industry: {risk_request.industry}")
        ctx.logger.info(f"  - B Corp URL: {risk_request.b_corp_profile_url}")

        await ctx.send(RISK_AGENT_ADDRESS, risk_request)

        log_message_transmission(
            ctx,
            "SENT",
            "RiskRequest",
            msg.request_id,
            {
                "supplier_name": risk_request.supplier_name,
                "industry": risk_request.industry,
            },
        )

        ctx.logger.info(f"✅ RiskRequest successfully sent to Risk Agent")
        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("⏳ WAITING FOR RISK RESPONSE...")
        ctx.logger.info("=" * 70)
        ctx.logger.info("Progress:")
        ctx.logger.info("  [✓] Compliance Agent - COMPLETE")
        ctx.logger.info("  [✓] Financial Agent - COMPLETE")
        ctx.logger.info("  [ ] Risk Agent - IN PROGRESS")
        ctx.logger.info("=" * 70)

    except Exception as e:
        ctx.logger.error(f"Error handling financial response: {e}")
        import traceback

        traceback.print_exc()


@risk_protocol.on_message(model=RiskResponse)
async def handle_risk_response(ctx: Context, sender: str, msg: RiskResponse):
    """Handle risk response and wait for other responses before sending to user"""
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📨 [3/3] RECEIVED RISK RESPONSE")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier: {msg.supplier_name}")
    ctx.logger.info(f"Risk Score: {msg.risk_score}/100")
    ctx.logger.info(
        f"Risk Factors Count: {len(msg.risk_factors) if msg.risk_factors else 0}"
    )
    ctx.logger.info("=" * 70)

    # Verify message received from Risk Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "RiskResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "risk_score": msg.risk_score,
        },
    )

    try:
        # Store risk response in pending_responses
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["risk_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Risk response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Score: {msg.risk_score}/100")

        # SEQUENTIAL: All 3 agents complete - combine and send results
        ctx.logger.info("")
        ctx.logger.info("=" * 70)
        ctx.logger.info("✅ ALL 3 AGENTS COMPLETE - COMBINING RESULTS")
        ctx.logger.info("=" * 70)
        ctx.logger.info("Progress:")
        ctx.logger.info("  [✓] Compliance Agent - COMPLETE")
        ctx.logger.info("  [✓] Financial Agent - COMPLETE")
        ctx.logger.info("  [✓] Risk Agent - COMPLETE")
        ctx.logger.info("=" * 70)

        # Check if we have all responses now and send combined response
        await check_and_send_combined_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling risk response: {e}")
        import traceback

        traceback.print_exc()


@demand_protocol.on_message(model=DemandResponse)
async def handle_demand_response(ctx: Context, sender: str, msg: DemandResponse):
    """Handle demand forecast response and wait for performance response"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Demand Forecast Response (1/2)")
    ctx.logger.info("=" * 60)

    # Verify message received from Demand Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "DemandResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "overall_performance": msg.overall_performance,
        },
    )

    try:
        # Store demand response in pending_responses
        pending_responses = ctx.storage.get("pending_responses")  # or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["demand_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Demand response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Overall Performance: {msg.overall_performance}")

        # Check if we have both monitoring responses now
        await check_and_send_monitoring_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling demand response: {e}")
        import traceback

        traceback.print_exc()


@performance_protocol.on_message(model=PerformanceResponse)
async def handle_performance_response(
    ctx: Context, sender: str, msg: PerformanceResponse
):
    """Handle performance monitoring response and wait for demand/logistics responses"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Performance Monitoring Response (1/3)")
    ctx.logger.info("=" * 60)

    # Verify message received from Performance Agent
    log_message_transmission(
        ctx,
        "RECEIVED",
        "PerformanceResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "overall_risk_level": msg.overall_risk_level,
        },
    )

    try:
        # Store performance response in pending_responses
        pending_responses = ctx.storage.get("pending_responses")  # or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["performance_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Performance response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Overall Risk: {msg.overall_risk_level}")
        ctx.logger.info(f"Alerts: {len(msg.alerts)}")

        # Check if we have all monitoring responses now
        await check_and_send_monitoring_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling performance response: {e}")
        import traceback

        traceback.print_exc()


@logistics_protocol.on_message(model=LogisticsResponse)
async def handle_logistics_response(ctx: Context, sender: str, msg: LogisticsResponse):
    """Handle logistics monitoring response and wait for all responses"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("Received Logistics Monitoring Response (3/3)")
    ctx.logger.info("=" * 60)

    # Verify message received from Logistics Agent

    ctx.logger.info(f"Logistics Response for request_id: {msg.request_id}")
    log_message_transmission(
        ctx,
        "RECEIVED",
        "LogisticsResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "overall_logistics_status": msg.overall_logistics_status,
        },
    )

    try:
        # Store logistics response in pending_responses
        pending_responses = ctx.storage.get("pending_responses")  # or {}
        ctx.logger.info(f"Pending responses: {pending_responses}")

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["logistics_response"] = msg.model_dump()
        ctx.logger.info(
            f"Logistics response stored: {pending_responses[msg.request_id]['logistics_response']}"
        )
        ctx.storage.set("pending_responses", pending_responses)

        ctx.logger.info(f"Logistics response stored")
        ctx.logger.info(f"Supplier: {msg.supplier_name}")
        ctx.logger.info(f"Current Inventory: {msg.current_inventory_level}")
        ctx.logger.info(f"Predicted Inventory: {msg.predicted_inventory_level}")
        ctx.logger.info(f"Overall Logistics Status: {msg.overall_logistics_status}")
        ctx.logger.info(f"Alerts: {len(msg.alerts)}")

        # Check if we have all monitoring responses now
        await check_and_send_monitoring_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling logistics response: {e}")
        import traceback

        traceback.print_exc()


async def store_monitoring_data_in_supabase(
    ctx: Context,
    supplier_name: str,
    performance_response: Dict[str, Any],
    demand_response: Dict[str, Any],
    logistics_response: Dict[str, Any],
) -> bool:
    """
    Store monitoring agent responses in Supabase database.

    Args:
        ctx: Agent context
        supplier_name: Name of the supplier being monitored
        performance_response: Performance agent response data
        demand_response: Demand agent response data
        logistics_response: Logistics agent response data

    Returns:
        bool: True if successful, False otherwise
    """
    global supabase_client

    try:
        if not supabase_client:
            ctx.logger.warning(
                "⚠️ Supabase client not initialized - skipping data storage"
            )
            return False

        # Determine overall status based on all three agent responses
        performance_risk = performance_response.get("overall_risk_level", "UNKNOWN")
        demand_performance = demand_response.get("overall_performance", "UNKNOWN")
        logistics_status = logistics_response.get("overall_logistics_status", "UNKNOWN")

        # Create overall status summary
        overall_status = f"Risk: {performance_risk} | Performance: {demand_performance} | Logistics: {logistics_status}"

        # Prepare data for insertion
        monitoring_data = {
            "supplier_name": supplier_name,
            "demand_content": demand_response,
            "performance_content": performance_response,
            "logistics_content": logistics_response,
            "overall_status": overall_status,
        }

        ctx.logger.info("📊 Storing monitoring data in Supabase...")
        ctx.logger.info(f"   Supplier: {supplier_name}")
        ctx.logger.info(f"   Overall Status: {overall_status}")

        # Insert data into monitoring_conversations table
        result = (
            supabase_client.table("monitoring_conversations")
            .insert(monitoring_data)
            .execute()
        )

        ctx.logger.info("✅ Successfully stored monitoring data in Supabase")
        ctx.logger.info(
            f"   Record ID: {result.data[0]['id'] if result.data else 'N/A'}"
        )

        return True

    except Exception as e:
        ctx.logger.error(f"❌ Error storing monitoring data in Supabase: {e}")
        import traceback

        traceback.print_exc()
        return False


async def check_and_send_monitoring_response(ctx: Context, request_id: str):
    """Check if all 3 monitoring responses are received, combine them, and send to user"""
    pending_responses = ctx.storage.get("pending_responses") or {}

    if request_id not in pending_responses:
        ctx.logger.warning(f"No pending response for request {request_id}")
        return

    response_data = pending_responses[request_id]
    performance_response = response_data.get("performance_response")
    demand_response = response_data.get("demand_response")
    logistics_response = response_data.get("logistics_response")

    # Check if we have ALL THREE monitoring responses
    if (
        performance_response is None
        or demand_response is None
        or logistics_response is None
    ):
        ctx.logger.info(f"Still waiting for monitoring responses...")
        ctx.logger.info(
            f"Performance: {'RECEIVED' if performance_response else 'PENDING'}"
        )
        ctx.logger.info(f"Demand: {'RECEIVED' if demand_response else 'PENDING'}")
        ctx.logger.info(f"Logistics: {'RECEIVED' if logistics_response else 'PENDING'}")
        return

    # We have all three responses! Combine them
    ctx.logger.info("=" * 70)
    ctx.logger.info("ALL 3 MONITORING RESPONSES RECEIVED - COMBINING RESULTS")
    ctx.logger.info("=" * 70)

    # Store monitoring data in Supabase database
    supplier_name = performance_response.get("supplier_name", "Unknown")
    await store_monitoring_data_in_supabase(
        ctx,
        supplier_name,
        performance_response,
        demand_response,
        logistics_response,
    )

    user_sender = response_data.get("sender")

    if not user_sender:
        ctx.logger.error(f"No sender found for request {request_id}")
        return

    # Build combined monitoring report
    performance_alerts_text = ""
    if performance_response.get("alerts"):
        performance_alerts_text = "\n".join(
            [f"  • {alert}" for alert in performance_response.get("alerts", [])]
        )
    else:
        performance_alerts_text = "  • No active alerts"

    demand_alerts_text = ""
    if demand_response.get("alerts"):
        demand_alerts_text = "\n".join(
            [f"  • {alert}" for alert in demand_response.get("alerts", [])]
        )
    else:
        demand_alerts_text = "  • No active alerts"

    logistics_alerts_text = ""
    if logistics_response.get("alerts"):
        logistics_alerts_text = "\n".join(
            [f"  • {alert}" for alert in logistics_response.get("alerts", [])]
        )
    else:
        logistics_alerts_text = "  • No active alerts"

    response_text = f"""
SUPPLIER MONITORING UPDATE

Supplier: {performance_response.get('supplier_name', 'N/A')}
Overall Risk Level: {performance_response.get('overall_risk_level', 'N/A')}
Overall Performance: {demand_response.get('overall_performance', 'N/A')}
Logistics Status: {logistics_response.get('overall_logistics_status', 'N/A')}

===== EXTERNAL FACTORS (PERFORMANCE MONITORING) =====

WEATHER CONDITIONS
Status: {performance_response.get('weather_status', 'N/A')}
{performance_response.get('weather_details', 'N/A')}

---

LABOR & STRIKES
Status: {performance_response.get('strike_status', 'N/A')}
{performance_response.get('strike_details', 'N/A')}

---

POLITICAL ENVIRONMENT
Status: {performance_response.get('political_status', 'N/A')}
{performance_response.get('political_details', 'N/A')}

---

LEGAL & REGULATORY
Status: {performance_response.get('legal_status', 'N/A')}
{performance_response.get('legal_details', 'N/A')}

---

PERFORMANCE ALERTS ({len(performance_response.get('alerts', []))})
{performance_alerts_text}

===== OPERATIONAL METRICS (DEMAND FORECAST) =====

DELIVERY & DELAYS
Status: {demand_response.get('delay_status', 'N/A')}
{demand_response.get('delay_details', 'N/A')}

---

SHIPPING & LOGISTICS
Status: {demand_response.get('shipping_status', 'N/A')}
{demand_response.get('shipping_details', 'N/A')}

---

QUALITY TRACKING
Status: {demand_response.get('quality_status', 'N/A')}
{demand_response.get('quality_details', 'N/A')}

---

DEMAND ALERTS ({len(demand_response.get('alerts', []))})
{demand_alerts_text}

===== INVENTORY & LOGISTICS MANAGEMENT =====

CURRENT INVENTORY LEVEL
Status: {logistics_response.get('current_inventory_level', 'N/A')}
Current Units: {logistics_response.get('current_inventory_units', 0)}
{logistics_response.get('inventory_details', 'N/A')}

---

INVENTORY FORECAST
Predicted Level: {logistics_response.get('predicted_inventory_level', 'N/A')}
Predicted Units: {logistics_response.get('predicted_inventory_units', 0)}
{logistics_response.get('inventory_forecast', 'N/A')}

---

RESTOCKING STATUS
Status: {logistics_response.get('restocking_status', 'N/A')}
{logistics_response.get('restocking_details', 'N/A')}

---

LOGISTICS ALERTS ({len(logistics_response.get('alerts', []))})
{logistics_alerts_text}

---

This comprehensive 24/7 monitoring update combines external factors, operational metrics, and inventory management to give you a complete view of your supplier's performance. Request another update anytime to see the latest conditions.
    """

    # Send combined monitoring response back to user
    ctx.logger.info(f"Sending combined monitoring update to user: {user_sender}")

    response = ChatMessage(
        timestamp="",
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=response_text.strip()),
            EndSessionContent(type="end-session"),
        ],
    )

    await ctx.send(user_sender, response)

    log_message_transmission(
        ctx,
        "SENT",
        "ChatMessage",
        str(response.msg_id),
        {
            "type": "monitoring_update",
            "risk_level": performance_response.get("overall_risk_level"),
            "performance": demand_response.get("overall_performance"),
        },
    )

    ctx.logger.info(f"Combined monitoring update sent to user")
    ctx.logger.info(
        f"Risk Level: {performance_response.get('overall_risk_level', 'N/A')}"
    )
    ctx.logger.info(f"Performance: {demand_response.get('overall_performance', 'N/A')}")
    ctx.logger.info(
        f"Logistics Status: {logistics_response.get('overall_logistics_status', 'N/A')}"
    )

    # Clean up session and pending responses
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions.pop(request_id, None)
    ctx.storage.set("active_sessions", active_sessions)

    pending_responses.pop(request_id, None)
    ctx.storage.set("pending_responses", pending_responses)

    ctx.logger.info(f"Session cleaned up for request {request_id}")
    ctx.logger.info("=" * 70)


async def check_and_send_combined_response(ctx: Context, request_id: str):
    """Check if all three responses are received, combine them, and send to user"""
    pending_responses = ctx.storage.get("pending_responses") or {}

    if request_id not in pending_responses:
        ctx.logger.warning(f"No pending response for request {request_id}")
        return

    response_data = pending_responses[request_id]
    compliance_response = response_data.get("compliance_response")
    financial_response = response_data.get("financial_response")
    risk_response = response_data.get("risk_response")

    # Check if we have ALL THREE responses
    if (
        compliance_response is None
        or financial_response is None
        or risk_response is None
    ):
        ctx.logger.info(f"Still waiting for responses...")
        ctx.logger.info(
            f"Compliance: {'RECEIVED' if compliance_response else 'PENDING'}"
        )
        ctx.logger.info(f"Financial: {'RECEIVED' if financial_response else 'PENDING'}")
        ctx.logger.info(f"Risk: {'RECEIVED' if risk_response else 'PENDING'}")
        return

    # We have all three responses! Combine them
    ctx.logger.info("=" * 70)
    ctx.logger.info("ALL THREE RESPONSES RECEIVED - COMBINING RESULTS")
    ctx.logger.info("=" * 70)

    user_sender = response_data.get("sender")
    user_query = response_data.get("user_query")

    if not user_sender:
        ctx.logger.error(f"No sender found for request {request_id}")
        return

    # Determine overall approval status - ALL THREE must pass (60 is the approval threshold)
    compliance_passed = compliance_response.get("compliance_score", 0) >= 60
    financial_passed = financial_response.get("financial_score", 0) >= 60
    risk_passed = risk_response.get("risk_score", 0) >= 60
    overall_approved = compliance_passed and financial_passed and risk_passed

    # Build dynamic supplier requirements status
    passed_requirements = []
    failed_requirements = []

    if compliance_passed:
        passed_requirements.append("compliance")
    else:
        failed_requirements.append("compliance")

    if financial_passed:
        passed_requirements.append("financial")
    else:
        failed_requirements.append("financial")

    if risk_passed:
        passed_requirements.append("risk management")
    else:
        failed_requirements.append("risk management")

    if overall_approved:
        requirements_text = (
            f"Supplier meets all requirements: {', '.join(passed_requirements)}"
        )
        # Store the approved supplier for monitoring
        ctx.storage.set(
            "selected_supplier", compliance_response.get("supplier_name", "")
        )
        ctx.logger.info(
            f"Approved supplier stored for monitoring: {compliance_response.get('supplier_name', '')}"
        )
    elif len(passed_requirements) > 0:
        requirements_text = f"Supplier meets {', '.join(passed_requirements)} but does not meet {', '.join(failed_requirements)}"
    else:
        requirements_text = "Supplier does not meet any requirements"

    # Build dynamic recommendation summary
    if overall_approved:
        approval_header = "APPROVED FOR PARTNERSHIP"
        recommendation_summary = f"Based on comprehensive analysis across compliance, financial, and risk management factors, this supplier demonstrates strong alignment with your business values and meets all required thresholds for partnership consideration."
    else:
        approval_header = "NOT APPROVED FOR PARTNERSHIP"
        if len(failed_requirements) == 3:
            recommendation_summary = "This supplier requires significant improvements across all evaluation areas (compliance, financial, and risk management) before being considered for partnership."
        elif len(failed_requirements) == 2:
            recommendation_summary = f"While {passed_requirements[0]} metrics are acceptable, this supplier must address concerns in {' and '.join(failed_requirements)} before moving forward with partnership."
        else:
            failed_area = failed_requirements[0]
            recommendation_summary = f"The supplier shows strength in {' and '.join(passed_requirements)}, but {failed_area} concerns must be mitigated to proceed with partnership."

    # Format financial details cleanly without bullet points
    financial_details = financial_response.get("financial_details", "N/A")

    # Format risk factors - include even small ones without being nice
    risk_factors = financial_response.get("risk_factors", [])
    if risk_factors:
        risk_factors_text = "\n".join([f"  • {factor}" for factor in risk_factors])
    else:
        risk_factors_text = "  • No significant financial risks identified"

    # Format violations - include even if minimal
    violations = compliance_response.get("violations", [])
    violations_count = len(violations)
    if violations:
        violations_text = "\n".join([f"  • {v}" for v in violations])
    else:
        violations_text = "  • None identified"

    # Format risk management details
    risk_details = risk_response.get("risk_details", "N/A")

    # Format risk management factors
    risk_factors_list = risk_response.get("risk_factors", [])
    if risk_factors_list:
        risk_mgmt_factors_text = "\n".join(
            [f"  • {factor}" for factor in risk_factors_list]
        )
    else:
        risk_mgmt_factors_text = "  • No significant risk factors identified"

    response_text = f"""
{approval_header}

SUPPLIER ANALYSIS REPORT

Business Owner: {compliance_response.get('supplier_name', 'N/A')}
Supplier Status: {'APPROVED' if overall_approved else 'NOT APPROVED'}

Supplier Requirements: {requirements_text}

COMPLIANCE ANALYSIS

Overall Compliance Score: {compliance_response.get('compliance_score')}/100

Ethics & Worker Treatment:
{compliance_response.get('ethics_info', 'N/A')}

Sustainability Practices:
{compliance_response.get('sustainability_info', 'N/A')}

Violations Found: {violations_count}
{violations_text}

FINANCIAL RISK ANALYSIS

Financial Risk Score: {financial_response.get('financial_score')}/100 (Higher = Lower Risk)

Operating & Cost Assessment:
{financial_details}

Risk Factors Identified: {len(risk_factors)}
{risk_factors_text}

RISK MANAGEMENT ANALYSIS

Risk Management Score: {risk_response.get('risk_score')}/100 (Higher = Lower Risk)

Risk Assessment:
{risk_details}

Identified Risk Factors: {len(risk_factors_list)}
{risk_mgmt_factors_text}

RECOMMENDATION

{recommendation_summary}
    """

    # Send combined response back to user via chat
    ctx.logger.info(f"Sending combined response to user: {user_sender}")

    response = ChatMessage(
        timestamp="",
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=response_text.strip()),
            EndSessionContent(type="end-session"),
        ],
    )

    await ctx.send(user_sender, response)

    log_message_transmission(
        ctx,
        "SENT",
        "ChatMessage",
        str(response.msg_id),
        {"type": "combined_result", "approved": overall_approved},
    )

    ctx.logger.info(f"Combined response sent to user")
    ctx.logger.info(
        f"Overall Status: {'APPROVED' if overall_approved else 'NOT APPROVED'}"
    )
    ctx.logger.info(f"Compliance: {compliance_response.get('compliance_score')}/100")
    ctx.logger.info(f"Financial: {financial_response.get('financial_score')}/100")
    ctx.logger.info(f"Risk: {risk_response.get('risk_score')}/100")

    # Clean up session and pending responses
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions.pop(request_id, None)
    ctx.storage.set("active_sessions", active_sessions)

    pending_responses.pop(request_id, None)
    ctx.storage.set("pending_responses", pending_responses)

    ctx.logger.info(f"Session cleaned up for request {request_id}")
    ctx.logger.info("=" * 70)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}"
    )


supplier_orchestrator.include(chat_proto, publish_manifest=True)
supplier_orchestrator.include(find_supplier_protocol, publish_manifest=True)
supplier_orchestrator.include(compliance_protocol, publish_manifest=True)
supplier_orchestrator.include(financial_protocol, publish_manifest=True)
supplier_orchestrator.include(risk_protocol, publish_manifest=True)
supplier_orchestrator.include(performance_protocol, publish_manifest=True)
supplier_orchestrator.include(demand_protocol, publish_manifest=True)
supplier_orchestrator.include(logistics_protocol, publish_manifest=True)

if __name__ == "__main__":
    supplier_orchestrator.run()
