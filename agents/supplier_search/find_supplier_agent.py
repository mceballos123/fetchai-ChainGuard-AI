"""
Find Supplier Agent - Multi-Country Supplier Search
Uses LangGraph workflow to search for suppliers in US (SEC API) or UK (Companies House API)
"""

from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from models.find_supplier import (
    FindSupplierRequest,
    FindSupplierResponse,
    SupplierSearchState,
)
# Import the LangGraph workflow
from langgraph_logic.find_supplier_langgraph import build_supplier_search_graph
import os
import re
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

# ========== Agent Initialization ==========

find_supplier_agent = Agent(
    name="find_supplier_agent",
    seed=os.getenv("FIND_SUPPLIER_SEED"),
    port=8006,
    mailbox=True,
)

find_supplier_protocol = Protocol(name="find_supplier_protocol", version="1.0")

# Build the LangGraph workflow (imported from langgraph_logic module)
supplier_search_graph = build_supplier_search_graph()


# ========== Event Handlers ==========

@find_supplier_agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize find_supplier agent on startup"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("FIND SUPPLIER AGENT STARTING UP (MULTI-COUNTRY)")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"Agent Name: {ctx.agent.name}")
    ctx.logger.info(f"Agent Address: {ctx.agent.address}")
    ctx.logger.info(f"Port: 8006")
    ctx.logger.info("Supported Countries: US (SEC API), UK (Companies House API)")
    ctx.logger.info("=" * 70)

    # Initialize storage
    ctx.storage.set("search_history", [])
    ctx.storage.set("processed_request_ids", [])

    ctx.logger.info(
        "Find Supplier Agent Ready - Listening for FindSupplierRequest messages..."
    )


@find_supplier_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("FIND SUPPLIER AGENT SHUTTING DOWN")
    ctx.logger.info("=" * 70)

    search_history = ctx.storage.get("search_history") or []
    ctx.logger.info(f"Total searches processed: {len(search_history)}")


# ========== Helper Functions ==========

def extract_business_category(user_query: str) -> str:
    """
    Extract business category from user query

    Args:
        user_query: User's search query

    Returns:
        Business category string or "general" if not found
    """
    user_query_lower = user_query.lower()

    # Common business categories
    categories = [
        "coffee", "tea", "food", "restaurant", "pizza", "burger", "cafe",
        "clothing", "apparel", "fashion", "textile", "bakery", "chocolate",
        "beer", "wine", "beverage", "furniture", "design", "manufacturing",
        "technology", "software", "consulting", "cosmetics", "beauty",
        "wellness", "agriculture", "farming", "organic", "water"
    ]

    # Check if any category is mentioned in the query
    for category in categories:
        if category in user_query_lower:
            return category

    # Fallback: extract significant words from query
    words = re.findall(r"\b[a-z]+\b", user_query_lower)
    if len(words) > 2:
        skip_words = {
            "i", "need", "want", "find", "looking", "for", "a", "an",
            "the", "some", "get", "me", "in", "from", "us", "uk"
        }
        for word in words:
            if word not in skip_words and len(word) > 3:
                return word

    return "general"


# ========== Message Handlers ==========

@find_supplier_protocol.on_message(
    model=FindSupplierRequest, replies=FindSupplierResponse
)
async def handle_find_supplier_request(
    ctx: Context, sender: str, msg: FindSupplierRequest
):
    """
    Handle FindSupplierRequest messages

    This handler:
    1. Extracts the business category from the user query
    2. Invokes the LangGraph workflow to search for suppliers
    3. Returns the results to the sender
    """
    ctx.logger.info("")
    ctx.logger.info("=" * 70)
    ctx.logger.info("📥 FIND SUPPLIER AGENT: RECEIVED REQUEST")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"From: {sender}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"User Query: {msg.user_query}")
    ctx.logger.info(f"Business Category: {msg.business_category}")
    ctx.logger.info("=" * 70)
    ctx.logger.info("")
    ctx.logger.info("🔍 Starting LangGraph Multi-Country Supplier Search...")
    ctx.logger.info("=" * 70)

    try:
        # Check for duplicate requests
        processed_ids = ctx.storage.get("processed_request_ids") or []
        if msg.request_id in processed_ids:
            ctx.logger.warning(f"Duplicate request: {msg.request_id}")
            return

        # Mark request as processed
        processed_ids.append(msg.request_id)
        ctx.storage.set("processed_request_ids", processed_ids)

        # Extract or use provided business category
        search_category = (
            msg.business_category
            if msg.business_category != "general"
            else extract_business_category(msg.user_query)
        )

        # Prepare initial state for LangGraph workflow
        initial_state: SupplierSearchState = {
            "request_id": msg.request_id,
            "user_query": msg.user_query,
            "business_category": search_category,
            "country": None,
            "search_query": "",
            "suppliers": [],
            "success": False,
            "error_message": None,
            "search_summary": "",
            "ctx": ctx,  # Pass uAgent context to LangGraph nodes
        }

        ctx.logger.info(f"🔄 Executing LangGraph workflow for category: {search_category}")

        # Invoke the LangGraph workflow
        final_state = supplier_search_graph.invoke(initial_state)

        ctx.logger.info("LangGraph Workflow Completed")

        # Process successful results
        if final_state["success"] and final_state["suppliers"]:
            suppliers = final_state["suppliers"]
            country = final_state["country"]

            ctx.logger.info(f"✅ Found {len(suppliers)} Suppliers in {country}!")
            for idx, supplier in enumerate(suppliers, 1):
                ctx.logger.info(f"  {idx}. {supplier.company_name}")

            # Create successful response
            response = FindSupplierResponse(
                request_id=msg.request_id,
                success=True,
                suppliers=suppliers,
                country=country,
                search_category=search_category,
                total_results_found=len(suppliers),
                search_summary=final_state["search_summary"],
                # Legacy fields for backwards compatibility
                best_supplier=suppliers[0] if suppliers else None,
                alternative_suppliers=suppliers[1:] if len(suppliers) > 1 else [],
            )

            # Update search history
            search_history = ctx.storage.get("search_history") or []
            search_history.append(
                {
                    "request_id": msg.request_id,
                    "country": country,
                    "category": search_category,
                    "suppliers_found": len(suppliers),
                }
            )
            ctx.storage.set("search_history", search_history)

            ctx.logger.info(f"📤 Sending response to {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: True")
            ctx.logger.info(f"Suppliers: {len(suppliers)}")

            await ctx.send(sender, response)
        else:
            # Handle error case
            ctx.logger.warning("❌ No suppliers found")
            ctx.logger.warning(
                f"Error: {final_state.get('error_message', 'No suppliers found')}"
            )

            error_response = FindSupplierResponse(
                request_id=msg.request_id,
                success=False,
                country=final_state.get("country", ""),
                search_category=search_category,
                total_results_found=0,
                search_summary=final_state.get("search_summary", "Suppliers not found"),
                error_message=final_state.get("error_message", "No suppliers found"),
            )

            ctx.logger.info(f"📤 Sending error response to {sender}")
            ctx.logger.info(f"Request ID: {msg.request_id}")
            ctx.logger.info(f"Success: False")
            ctx.logger.info(f"Error: {error_response.error_message}")

            await ctx.send(sender, error_response)

    except Exception as e:
        # Handle unexpected errors
        ctx.logger.error(f"💥 Error processing request: {e}")
        import traceback
        traceback.print_exc()

        error_response = FindSupplierResponse(
            request_id=msg.request_id,
            success=False,
            search_category=msg.business_category,
            total_results_found=0,
            search_summary="Internal error during supplier search",
            error_message=str(e),
        )

        await ctx.send(sender, error_response)


# ========== Agent Setup ==========

# Include the protocol in the agent
find_supplier_agent.include(find_supplier_protocol, publish_manifest=True)

if __name__ == "__main__":
    find_supplier_agent.run()
