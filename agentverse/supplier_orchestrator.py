from uuid import uuid4
from uagents import Agent, Context, Protocol
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from supabase import create_client, Client
import google.generativeai as genai


from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
    chat_protocol_spec,
)

# Import the four agent models
from models.find_supplier import FindSupplierRequest, FindSupplierResponse
from models.supplier_info import SupplierInfoRequest, SupplierInfoResponse
from models.finance_info import FinanceInfoRequest, FinanceInfoResponse
from models.risk_info import RiskInfoRequest, RiskInfoResponse

# Import test utilities
from test_func.orchestrator_helpers import (
    verify_orchestrator_connection,
    log_message_transmission,
    validate_request_before_sending,
)

# Import instruction prompts
from prompts.instructions import (
    INSTRUCTIONS_PROMPT,
    GREETING_RESPONSE,
    HELP_RESPONSE,
)

load_dotenv()

SUPPLIER_ORCHESTRATOR_SEED = os.getenv("SUPPLIER_ORCHESTRATOR_SEED")

# Initialize Gemini for personalized responses
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "max_output_tokens": 500,
        "temperature": 0.5,
    },
)


def generate_personalized_response(
    user_message: str,
    query_type: str,
    fallback_response: str,
    supplier_name: str = None,
) -> str:
    """
    Generate a personalized response using Google Gemini LLM.

    Args:
        user_message: The user's original message
        query_type: Type of query (greeting, help, irrelevant, need_supplier)
        fallback_response: Static response to use if LLM fails
        supplier_name: Name of the supplier (if user has one)

    Returns:
        Personalized response string
    """
    try:
        context_map = {
            "greeting": "Greet warmly. Introduce ChainGuard AI briefly.",
            "help": "Explain ChainGuard AI's features: find suppliers, get supplier info, financial analysis, risk assessment.",
            "irrelevant": "Politely redirect to ChainGuard AI features.",
            "need_supplier": "Tell user to find a supplier first before requesting info.",
        }

        context = context_map.get(query_type, "Guide user on ChainGuard AI usage.")
        prompt = f"{context} User said: '{user_message}'. Respond in max 2 sentences."

        response = gemini_model.generate_content(prompt)

        if not response.candidates:
            return fallback_response

        candidate = response.candidates[0]

        if candidate.finish_reason != 1:
            return fallback_response

        if not candidate.content or not candidate.content.parts:
            return fallback_response

        generated_response = candidate.content.parts[0].text.strip()

        if generated_response and len(generated_response) > 20:
            return generated_response
        else:
            return fallback_response

    except Exception as e:
        return fallback_response


def clear_session_data(ctx: Context) -> None:
    """Clear all session data to start fresh"""
    ctx.storage.set("selected_supplier", None)
    ctx.storage.set("supplier_cik", None)
    ctx.storage.set("supplier_country", None)
    ctx.storage.set("supplier_ticker", None)
    ctx.storage.set("active_sessions", {})
    ctx.storage.set("pending_responses", {})
    ctx.logger.info("Session data cleared")


def is_reset_command(user_query: str) -> bool:
    """
    Check if the user is explicitly requesting a session reset.
    """
    query_lower = user_query.lower().strip()

    reset_patterns = [
        "reset",
        "start over",
        "new search",
        "clear",
        "clear session",
        "new supplier",
        "find new supplier",
        "start fresh",
        "fresh start",
        "begin again",
        "restart",
    ]

    for pattern in reset_patterns:
        if pattern in query_lower:
            return True

    return False


def analyze_query_relevance(
    user_query: str, has_selected_supplier: bool = False
) -> Tuple[bool, str, str]:
    """
    Analyze whether the user query is relevant to ChainGuard AI's capabilities.

    Returns:
        Tuple of (is_relevant, query_type, response_message)
        - is_relevant: True if this is a valid ChainGuard request
        - query_type: "find_supplier", "supplier_info", "finance_info", "risk_info", "greeting", "help", "reset", "need_supplier", or "irrelevant"
        - response_message: Pre-built response for non-actionable queries
    """
    query_lower = user_query.lower().strip()

    # Check for explicit reset command first
    if is_reset_command(user_query):
        return True, "reset", ""

    # Empty or very short queries
    if len(query_lower) < 2:
        return False, "irrelevant", GREETING_RESPONSE

    # SUPPLIER INFO REQUEST - "I want to know more about [supplier]"
    supplier_info_patterns = [
        "know more about",
        "more info about",
        "information about",
        "tell me about",
        "details about",
        "more about",
        "learn about",
        "show me info",
        "get info about",
        "supplier info",
    ]

    has_supplier_info_pattern = any(
        pattern in query_lower for pattern in supplier_info_patterns
    )

    if has_supplier_info_pattern:
        if has_selected_supplier:
            return True, "supplier_info", ""
        else:
            return (
                False,
                "need_supplier",
                "Please find a supplier first using the supplier search feature before requesting detailed information.",
            )

    # FINANCE INFO REQUEST - "I want to check finance info"
    finance_info_patterns = [
        "finance",
        "financial",
        "financial info",
        "finance info",
        "financial data",
        "financial analysis",
        "check finance",
        "financial condition",
        "revenue",
        "financial performance",
    ]

    has_finance_pattern = any(
        pattern in query_lower for pattern in finance_info_patterns
    )

    if has_finance_pattern:
        if has_selected_supplier:
            return True, "finance_info", ""
        else:
            return (
                False,
                "need_supplier",
                "Please find a supplier first using the supplier search feature before requesting financial information.",
            )

    # RISK INFO REQUEST - "I want to know the risk"
    risk_info_patterns = [
        "risk",
        "risks",
        "risk factors",
        "risk assessment",
        "risk analysis",
        "risk info",
        "risk information",
        "check risk",
        "what are the risks",
    ]

    has_risk_pattern = any(pattern in query_lower for pattern in risk_info_patterns)

    if has_risk_pattern:
        if has_selected_supplier:
            return True, "risk_info", ""
        else:
            return (
                False,
                "need_supplier",
                "Please find a supplier first using the supplier search feature before requesting risk information.",
            )

    # SUPPLIER SEARCH INDICATORS
    find_supplier_patterns = [
        "find",
        "search",
        "looking for",
        "look for",
        "need a",
        "want a",
        "want to find",
        "get me",
        "get a",
        "show me",
        "i need",
        "i want",
    ]

    supplier_intent_keywords = [
        "supplier",
        "vendor",
        "source",
        "b corp",
        "b-corp",
        "bcorp",
    ]

    business_owner_patterns = [
        "i'm a",
        "i am a",
        "im a",
        "business owner",
        "business",
        "company",
        "shop",
        "store",
        "owner",
    ]

    has_supplier_keyword = any(kw in query_lower for kw in supplier_intent_keywords)
    has_find_pattern = any(pattern in query_lower for pattern in find_supplier_patterns)
    has_business_pattern = any(
        pattern in query_lower for pattern in business_owner_patterns
    )

    # Strong supplier intent
    if has_supplier_keyword:
        return True, "find_supplier", ""

    # Business owner with find intent
    if has_find_pattern and has_business_pattern:
        return True, "find_supplier", ""

    # GREETING INDICATORS
    greeting_patterns = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "howdy",
        "what's up",
        "whats up",
        "yo",
    ]

    if len(query_lower.split()) <= 3:
        is_greeting = any(
            query_lower.startswith(g) or query_lower == g for g in greeting_patterns
        )
        if is_greeting:
            return False, "greeting", GREETING_RESPONSE

    # HELP/INFO INDICATORS
    help_patterns = [
        "help",
        "how do",
        "how does",
        "how to",
        "what is",
        "what's",
        "what are",
        "explain",
        "tell me about",
        "can you",
        "what can",
        "features",
        "capabilities",
        "instructions",
        "tutorial",
        "guide",
    ]

    has_help_pattern = any(pattern in query_lower for pattern in help_patterns)
    if has_help_pattern:
        chainguard_mentions = [
            "chainguard",
            "chain guard",
            "this agent",
            "this platform",
            "you",
        ]
        if any(mention in query_lower for mention in chainguard_mentions):
            return False, "help", HELP_RESPONSE
        return False, "help", HELP_RESPONSE

    # If none of the above, it's likely irrelevant
    return False, "irrelevant", GREETING_RESPONSE


def extract_supplier_name_from_query(user_query: str) -> str:
    """
    Extract supplier name from queries like "I want to know more about [Company Name]"

    Args:
        user_query: The user's query

    Returns:
        Extracted supplier name or empty string if not found
    """
    query_lower = user_query.lower()

    # Patterns to look for
    patterns = [
        "know more about ",
        "more info about ",
        "information about ",
        "tell me about ",
        "details about ",
        "more about ",
        "learn about ",
    ]

    for pattern in patterns:
        if pattern in query_lower:
            # Extract text after the pattern
            idx = query_lower.find(pattern)
            after_pattern = user_query[idx + len(pattern):].strip()

            # Remove common trailing words
            for end_word in [" supplier", " company", " business"]:
                if after_pattern.lower().endswith(end_word):
                    after_pattern = after_pattern[:-len(end_word)].strip()

            return after_pattern

    return ""


# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Global Supabase client
supabase_client: Client | None = None

supplier_orchestrator = Agent(
    name="supplier_orchestrator",
    seed=SUPPLIER_ORCHESTRATOR_SEED,
    port=8000,
    mailbox=True,
)

chat_proto = Protocol(name="chat_protocol", spec=chat_protocol_spec)

# Agent Addresses
FIND_SUPPLIER_AGENT_ADDRESS = os.getenv("FIND_SUPPLIER_AGENT_ADDRESS")
SUPPLIER_INFO_AGENT_ADDRESS = os.getenv("SUPPLIER_INFO_AGENT_ADDRESS")
FINANCE_INFO_AGENT_ADDRESS = os.getenv("FINANCE_INFO_AGENT_ADDRESS")
RISK_MANAGEMENT_AGENT_ADDRESS = os.getenv("RISK_MANAGEMENT_AGENT_ADDRESS")

orchestrator_protocol = Protocol(name="supplier_orchestrator_protocol", version="1.0")


@supplier_orchestrator.on_event("startup")
async def startup(ctx: Context):
    """Initialize orchestrator on startup"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("SUPPLIER ORCHESTRATOR STARTING UP")
    ctx.logger.info("=" * 70)
    ctx.logger.info("Integrated Agents:")
    ctx.logger.info("  1. Find Supplier Agent")
    ctx.logger.info("  2. Supplier Info Agent")
    ctx.logger.info("  3. Finance Info Agent")
    ctx.logger.info("  4. Risk Management Agent")
    ctx.logger.info("=" * 70)

    # Initialize Supabase client
    global supabase_client
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            ctx.logger.info("Supabase client initialized")
        except Exception as e:
            ctx.logger.error(f"Failed to initialize Supabase: {e}")
            supabase_client = None
    else:
        supabase_client = None

    # Initialize storage
    ctx.storage.set("active_sessions", {})
    ctx.storage.set("selected_supplier", None)
    ctx.storage.set("supplier_cik", None)
    ctx.storage.set("supplier_country", None)
    ctx.storage.set("supplier_ticker", None)
    ctx.storage.set("pending_responses", {})
    ctx.storage.set(
        "message_trace",
        {
            "received_from_asi": [],
            "sent_to_agents": [],
            "received_from_agents": [],
            "sent_to_asi": [],
        },
    )

    # Verify connection on startup
    conn_status = verify_orchestrator_connection(ctx)
    ctx.logger.info(f"Connection Status: {conn_status['status']}")


@supplier_orchestrator.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("SUPPLIER ORCHESTRATOR SHUTTING DOWN")
    ctx.logger.info("=" * 70)


@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle chat messages from user via ASI:1"""

    # Log message received from ASI:1
    log_message_transmission(
        ctx, "RECEIVED", "ChatMessage", str(msg.msg_id), {"sender": sender}
    )

    # Send acknowledgment immediately
    ack = ChatAcknowledgement(
        timestamp="",
        acknowledged_msg_id=msg.msg_id,
    )
    await ctx.send(sender, ack)

    # Extract user query from message content
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

    # Analyze query relevance
    selected_supplier = ctx.storage.get("selected_supplier")
    has_supplier = selected_supplier is not None and selected_supplier != ""

    is_relevant, query_type, instruction_response = analyze_query_relevance(
        user_query, has_selected_supplier=has_supplier
    )

    ctx.logger.info(f"Query Type Detected: {query_type}")
    ctx.logger.info(f"Has Supplier: {has_supplier}")

    # Handle reset command
    if query_type == "reset":
        clear_session_data(ctx)

        reset_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text='Session Reset Complete!\n\nYour previous supplier data has been cleared. You can now start a fresh search.\n\nTo find a new supplier, tell me:\n- What type of business you have (e.g., coffee shop, restaurant, clothing store)\n- Your location/country\n\nExample: "I\'m a UK business owner looking for a coffee supplier"',
                ),
                EndSessionContent(type="end-session"),
            ],
        )
        await ctx.send(sender, reset_response)
        log_message_transmission(
            ctx,
            "SENT",
            "ChatMessage",
            str(reset_response.msg_id),
            {"type": "reset_confirmation"},
        )
        return

    # If this is a new supplier search and we already have a supplier, auto-reset
    if query_type == "find_supplier" and has_supplier:
        clear_session_data(ctx)
        has_supplier = False

    # Handle irrelevant or informational queries
    if not is_relevant:
        # Generate a personalized response using Gemini
        personalized_response = generate_personalized_response(
            user_message=user_query,
            query_type=query_type,
            fallback_response=instruction_response,
            supplier_name=selected_supplier if has_supplier else None,
        )

        instruction_message = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text=personalized_response.strip(),
                ),
                EndSessionContent(type="end-session"),
            ],
        )
        await ctx.send(sender, instruction_message)
        log_message_transmission(
            ctx,
            "SENT",
            "ChatMessage",
            str(instruction_message.msg_id),
            {"type": "personalized_instruction", "query_type": query_type},
        )
        return

    # Store session information
    msg_id = str(msg.msg_id)
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions[msg_id] = {
        "sender": sender,
        "query": user_query,
        "query_type": query_type,
    }
    ctx.storage.set("active_sessions", active_sessions)

    # Route to appropriate agent based on query type
    try:
        if query_type == "find_supplier":
            # Extract business category from user query
            business_category = "general"
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

            user_query_lower = user_query.lower()
            for keyword, category in category_keywords.items():
                if keyword in user_query_lower:
                    business_category = category
                    break

            # Initialize pending response tracking
            pending_responses = ctx.storage.get("pending_responses") or {}
            pending_responses[msg_id] = {
                "find_supplier_response": None,
                "sender": sender,
                "user_query": user_query,
            }
            ctx.storage.set("pending_responses", pending_responses)

            # Send to Find Supplier Agent
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

        elif query_type == "supplier_info":
            # Get stored supplier info
            supplier_name = ctx.storage.get("selected_supplier")
            supplier_cik = ctx.storage.get("supplier_cik")
            supplier_country = ctx.storage.get("supplier_country")
            supplier_ticker = ctx.storage.get("supplier_ticker") or ""

            if not supplier_name or not supplier_cik:
                ctx.logger.error("Missing supplier information for supplier_info request")
                error_response = ChatMessage(
                    timestamp="",
                    msg_id=uuid4(),
                    content=[
                        TextContent(
                            type="text",
                            text="Error: Supplier information not found. Please find a supplier first.",
                        )
                    ],
                )
                await ctx.send(sender, error_response)
                return

            # Initialize pending response tracking
            pending_responses = ctx.storage.get("pending_responses") or {}
            pending_responses[msg_id] = {
                "supplier_info_response": None,
                "sender": sender,
                "user_query": user_query,
            }
            ctx.storage.set("pending_responses", pending_responses)

            # Send to Supplier Info Agent
            supplier_info_request = SupplierInfoRequest(
                request_id=msg_id,
                company_name=supplier_name,
                cik=supplier_cik,
                country=supplier_country,
                ticker=supplier_ticker,
                timestamp="",
            )

            await ctx.send(SUPPLIER_INFO_AGENT_ADDRESS, supplier_info_request)

            log_message_transmission(
                ctx,
                "SENT",
                "SupplierInfoRequest",
                msg_id,
                {"company_name": supplier_name, "cik": supplier_cik},
            )

        elif query_type == "finance_info":
            # Get stored supplier info
            supplier_name = ctx.storage.get("selected_supplier")
            supplier_cik = ctx.storage.get("supplier_cik")
            supplier_country = ctx.storage.get("supplier_country")
            supplier_ticker = ctx.storage.get("supplier_ticker") or ""

            if not supplier_name or not supplier_cik:
                ctx.logger.error("Missing supplier information for finance_info request")
                error_response = ChatMessage(
                    timestamp="",
                    msg_id=uuid4(),
                    content=[
                        TextContent(
                            type="text",
                            text="Error: Supplier information not found. Please find a supplier first.",
                        )
                    ],
                )
                await ctx.send(sender, error_response)
                return

            # Initialize pending response tracking
            pending_responses = ctx.storage.get("pending_responses") or {}
            pending_responses[msg_id] = {
                "finance_info_response": None,
                "sender": sender,
                "user_query": user_query,
            }
            ctx.storage.set("pending_responses", pending_responses)

            # Send to Finance Info Agent
            finance_info_request = FinanceInfoRequest(
                request_id=msg_id,
                company_name=supplier_name,
                cik=supplier_cik,
                country=supplier_country,
                ticker=supplier_ticker,
                timestamp="",
            )

            await ctx.send(FINANCE_INFO_AGENT_ADDRESS, finance_info_request)

            log_message_transmission(
                ctx,
                "SENT",
                "FinanceInfoRequest",
                msg_id,
                {"company_name": supplier_name, "cik": supplier_cik},
            )

        elif query_type == "risk_info":
            # Get stored supplier info
            supplier_name = ctx.storage.get("selected_supplier")
            supplier_cik = ctx.storage.get("supplier_cik")
            supplier_country = ctx.storage.get("supplier_country")
            supplier_ticker = ctx.storage.get("supplier_ticker") or ""

            if not supplier_name or not supplier_cik:
                ctx.logger.error("Missing supplier information for risk_info request")
                error_response = ChatMessage(
                    timestamp="",
                    msg_id=uuid4(),
                    content=[
                        TextContent(
                            type="text",
                            text="Error: Supplier information not found. Please find a supplier first.",
                        )
                    ],
                )
                await ctx.send(sender, error_response)
                return

            # Initialize pending response tracking
            pending_responses = ctx.storage.get("pending_responses") or {}
            pending_responses[msg_id] = {
                "risk_info_response": None,
                "sender": sender,
                "user_query": user_query,
            }
            ctx.storage.set("pending_responses", pending_responses)

            # Send to Risk Management Agent
            risk_info_request = RiskInfoRequest(
                request_id=msg_id,
                company_name=supplier_name,
                cik=supplier_cik,
                country=supplier_country,
                ticker=supplier_ticker,
                timestamp="",
            )

            await ctx.send(RISK_MANAGEMENT_AGENT_ADDRESS, risk_info_request)

            log_message_transmission(
                ctx,
                "SENT",
                "RiskInfoRequest",
                msg_id,
                {"company_name": supplier_name, "cik": supplier_cik},
            )

    except Exception as e:
        ctx.logger.error(f"Error routing request: {e}")
        import traceback

        traceback.print_exc()

        # Clean up pending responses
        pending_responses = ctx.storage.get("pending_responses") or {}
        pending_responses.pop(msg_id, None)
        ctx.storage.set("pending_responses", pending_responses)

        # Send error response to user
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text=f"Error processing request: {str(e)}",
                )
            ],
        )
        await ctx.send(sender, error_response)
        log_message_transmission(
            ctx, "SENT", "ChatMessage", str(error_response.msg_id), {"type": "error"}
        )


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgment messages from user"""
    ctx.logger.info(
        f"Acknowledgment received from {sender} for message: {msg.acknowledged_msg_id}"
    )


# Protocol handlers for the four agents
find_supplier_protocol = Protocol(name="find_supplier_response_protocol", version="1.0")
supplier_info_protocol = Protocol(name="supplier_info_response_protocol", version="1.0")
finance_info_protocol = Protocol(name="finance_info_response_protocol", version="1.0")
risk_info_protocol = Protocol(name="risk_info_response_protocol", version="1.0")


@find_supplier_protocol.on_message(model=FindSupplierResponse)
async def handle_find_supplier_response(
    ctx: Context, sender: str, msg: FindSupplierResponse
):
    """Handle find supplier response"""

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

        user_sender = pending_responses[msg.request_id].get("sender")

        if not user_sender:
            ctx.logger.error(f"No sender found for request {msg.request_id}")
            return

        # Check if search was successful
        if not msg.success:
            ctx.logger.error(f"Find supplier search failed: {msg.error_message}")

            # Send error message to user
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Supplier Search Failed\n\nCategory: {msg.search_category}\nError: {msg.error_message}\n\nPlease try a different search term.",
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
            ctx.logger.warning("No supplier found in successful response")

            no_results_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"No Suppliers Found\n\nCategory: {msg.search_category}\nWe couldn't find any B Corporation certified companies matching your search.\n\nPlease try a different category or search term.",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            await ctx.send(user_sender, no_results_response)

            # Clean up
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            return

        # Supplier found successfully - Store it
        ctx.logger.info(f"Found supplier: {best_supplier.company_name}")

        # Store supplier information for future requests
        ctx.storage.set("selected_supplier", best_supplier.company_name)
        ctx.storage.set("supplier_cik", best_supplier.cik)
        ctx.storage.set("supplier_country", msg.country or "United States")
        ctx.storage.set("supplier_ticker", best_supplier.ticker or "")

        # Build response text
        response_text = f"""
SUPPLIER FOUND

Supplier: {best_supplier.company_name}
Industry: {best_supplier.industry or 'N/A'}
Country: {msg.country or 'N/A'}
CIK: {best_supplier.cik}

B Corp Profile: {best_supplier.b_corp_profile_url or 'N/A'}

---

{msg.search_summary}

---

What would you like to do next?

1. "I want to know more about {best_supplier.company_name}" - Get detailed company information
2. "Show me the financial info" - View financial analysis
3. "What are the risk factors?" - See risk assessment

You can also search for a different supplier by starting a new search.
        """

        # Send response to user
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
            {"type": "supplier_found", "supplier": best_supplier.company_name},
        )

        # Clean up
        pending_responses.pop(msg.request_id, None)
        ctx.storage.set("pending_responses", pending_responses)

    except Exception as e:
        ctx.logger.error(f"Error handling find supplier response: {e}")
        import traceback

        traceback.print_exc()


@supplier_info_protocol.on_message(model=SupplierInfoResponse)
async def handle_supplier_info_response(
    ctx: Context, sender: str, msg: SupplierInfoResponse
):
    """Handle supplier info response"""

    log_message_transmission(
        ctx,
        "RECEIVED",
        "SupplierInfoResponse",
        msg.request_id,
        {"success": msg.success},
    )

    try:
        # Get pending response tracking
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        user_sender = pending_responses[msg.request_id].get("sender")

        if not user_sender:
            ctx.logger.error(f"No sender found for request {msg.request_id}")
            return

        # Check if request was successful
        if not msg.success or not msg.supplier_info:
            ctx.logger.error(f"Supplier info request failed: {msg.error_message}")

            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Supplier Information Request Failed\n\nError: {msg.error_message}\n\nPlease try again later.",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            await ctx.send(user_sender, error_response)

            # Clean up
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            return

        # Build response text
        supplier_info = msg.supplier_info

        response_text = f"""
SUPPLIER INFORMATION

Company: {supplier_info.company_name}
CIK: {supplier_info.cik}
Country: {supplier_info.country}
Type: {'Foreign Company' if supplier_info.is_foreign else 'US Company'}
Filing Type: {supplier_info.filing_type}

---

COMPANY DESCRIPTION

{supplier_info.company_description}

---

HISTORY

{supplier_info.history or 'No history information available'}

---

Filing URL: {supplier_info.latest_filing_url or 'N/A'}

---

What would you like to do next?

- "Show me the financial info" - View financial analysis
- "What are the risk factors?" - See risk assessment
- Search for a different supplier
        """

        # Send response to user
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
            {"type": "supplier_info", "company": supplier_info.company_name},
        )

        # Clean up
        pending_responses.pop(msg.request_id, None)
        ctx.storage.set("pending_responses", pending_responses)

    except Exception as e:
        ctx.logger.error(f"Error handling supplier info response: {e}")
        import traceback

        traceback.print_exc()


@finance_info_protocol.on_message(model=FinanceInfoResponse)
async def handle_finance_info_response(
    ctx: Context, sender: str, msg: FinanceInfoResponse
):
    """Handle finance info response"""

    log_message_transmission(
        ctx,
        "RECEIVED",
        "FinanceInfoResponse",
        msg.request_id,
        {"success": msg.success},
    )

    try:
        # Get pending response tracking
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        user_sender = pending_responses[msg.request_id].get("sender")

        if not user_sender:
            ctx.logger.error(f"No sender found for request {msg.request_id}")
            return

        # Check if request was successful
        if not msg.success or not msg.financial_data:
            ctx.logger.error(f"Finance info request failed: {msg.error_message}")

            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Financial Information Request Failed\n\nError: {msg.error_message}\n\nPlease try again later.",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            await ctx.send(user_sender, error_response)

            # Clean up
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            return

        # Build response text
        financial_data = msg.financial_data

        response_text = f"""
FINANCIAL INFORMATION

Company: {financial_data.company_name}
Filing Type: {financial_data.filing_type}
Filing Date: {financial_data.filing_date or 'N/A'}

---

COMPANY OVERVIEW

{financial_data.overview or 'No overview available'}

---

FINANCIAL CONDITION

{financial_data.financial_condition or 'No financial condition information available'}

---

REVENUE INFORMATION

{financial_data.revenue_info or 'No revenue information available'}

---

LEGAL PROCEEDINGS

{financial_data.legal_proceedings or 'No legal proceedings reported'}

---

Filing URL: {financial_data.filing_url or 'N/A'}

Summary: {financial_data.extraction_summary}

---

What would you like to do next?

- "I want to know more about {financial_data.company_name}" - Get detailed company information
- "What are the risk factors?" - See risk assessment
- Search for a different supplier
        """

        # Send response to user
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
            {"type": "finance_info", "company": financial_data.company_name},
        )

        # Clean up
        pending_responses.pop(msg.request_id, None)
        ctx.storage.set("pending_responses", pending_responses)

    except Exception as e:
        ctx.logger.error(f"Error handling finance info response: {e}")
        import traceback

        traceback.print_exc()


@risk_info_protocol.on_message(model=RiskInfoResponse)
async def handle_risk_info_response(ctx: Context, sender: str, msg: RiskInfoResponse):
    """Handle risk info response"""

    log_message_transmission(
        ctx,
        "RECEIVED",
        "RiskInfoResponse",
        msg.request_id,
        {"success": msg.success},
    )

    try:
        # Get pending response tracking
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        user_sender = pending_responses[msg.request_id].get("sender")

        if not user_sender:
            ctx.logger.error(f"No sender found for request {msg.request_id}")
            return

        # Check if request was successful
        if not msg.success or not msg.risk_data:
            ctx.logger.error(f"Risk info request failed: {msg.error_message}")

            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Risk Information Request Failed\n\nError: {msg.error_message}\n\nPlease try again later.",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            await ctx.send(user_sender, error_response)

            # Clean up
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            return

        # Build response text
        risk_data = msg.risk_data

        # Format risk factors list
        risk_factors_text = ""
        if risk_data.all_risk_factors:
            risk_factors_text = "\n".join(
                [f"  {i+1}. {factor}" for i, factor in enumerate(risk_data.all_risk_factors)]
            )
        else:
            risk_factors_text = "  No specific risk factors identified"

        response_text = f"""
RISK ASSESSMENT

Company: {risk_data.company_name}
Filing Type: {risk_data.filing_type}
Filing Date: {risk_data.filing_date or 'N/A'}
Type: {'Foreign Company' if risk_data.is_foreign else 'US Company'}

---

SUMMARY OF KEY RISK FACTORS

{risk_data.summary_risk_factors or 'No summary available'}

---

OPERATIONAL RISKS

{risk_data.operational_risks or 'No operational risks identified'}

---

FINANCIAL RISKS

{risk_data.financial_risks or 'No financial risks identified'}

---

LEGAL & REGULATORY RISKS

{risk_data.legal_regulatory_risks or 'No legal/regulatory risks identified'}

---

ALL IDENTIFIED RISK FACTORS ({len(risk_data.all_risk_factors)})

{risk_factors_text}

---

Filing URL: {risk_data.filing_url or 'N/A'}

Summary: {risk_data.extraction_summary}

---

What would you like to do next?

- "I want to know more about {risk_data.company_name}" - Get detailed company information
- "Show me the financial info" - View financial analysis
- Search for a different supplier
        """

        # Send response to user
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
            {"type": "risk_info", "company": risk_data.company_name},
        )

        # Clean up
        pending_responses.pop(msg.request_id, None)
        ctx.storage.set("pending_responses", pending_responses)

    except Exception as e:
        ctx.logger.error(f"Error handling risk info response: {e}")
        import traceback

        traceback.print_exc()


# Include all protocols
supplier_orchestrator.include(chat_proto, publish_manifest=True)
supplier_orchestrator.include(find_supplier_protocol, publish_manifest=True)
supplier_orchestrator.include(supplier_info_protocol, publish_manifest=True)
supplier_orchestrator.include(finance_info_protocol, publish_manifest=True)
supplier_orchestrator.include(risk_info_protocol, publish_manifest=True)

if __name__ == "__main__":
    supplier_orchestrator.run()
