"""
Test script for Find Supplier Agent

This script tests the find_supplier agent locally without running the full orchestrator.
It sends a direct FindSupplierRequest and waits for the response.

Usage:
    python test_find_supplier.py
"""

from uagents import Agent, Context, Protocol
from models.find_supplier import FindSupplierRequest, FindSupplierResponse
from datetime import datetime, UTC
from uuid import uuid4
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Test client agent
test_client = Agent(
    name="test_find_supplier_client",
    seed="test_find_supplier_client_seed_12345",
    port=8007,
    endpoint=["http://localhost:8007/submit"],
)

# Get find_supplier agent address from environment
FIND_SUPPLIER_AGENT_ADDRESS = os.getenv("FIND_SUPPLIER_AGENT_ADDRESS")

if not FIND_SUPPLIER_AGENT_ADDRESS:
    print("❌ Error: FIND_SUPPLIER_AGENT_ADDRESS not set in .env")
    print("\nPlease:")
    print("1. Start find_supplier_agent.py first")
    print("2. Copy its address from logs")
    print("3. Add to .env file: FIND_SUPPLIER_AGENT_ADDRESS=agent1q...")
    exit(1)

test_protocol = Protocol(name="test_find_supplier_protocol", version="1.0")


@test_client.on_event("startup")
async def startup(ctx: Context):
    """Send test request to find_supplier agent on startup"""
    ctx.logger.info("=" * 70)
    ctx.logger.info("TEST CLIENT STARTED")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"Test Client Address: {ctx.agent.address}")
    ctx.logger.info(f"Find Supplier Agent Address: {FIND_SUPPLIER_AGENT_ADDRESS}")
    ctx.logger.info("=" * 70)

    # Wait a moment for agent to be ready
    await asyncio.sleep(2)

    # Test queries to try
    test_queries = [
        "I need a coffee supplier",
        "Find me a pizza restaurant",
        "Looking for sustainable chocolate",
    ]

    ctx.logger.info("\n🧪 Starting Find Supplier Agent Tests\n")

    for idx, query in enumerate(test_queries, 1):
        ctx.logger.info(f"\n{'=' * 70}")
        ctx.logger.info(f"TEST {idx}/{len(test_queries)}: {query}")
        ctx.logger.info(f"{'=' * 70}")

        # Create test request
        request = FindSupplierRequest(
            request_id=str(uuid4()),
            user_query=query,
            business_category="general",  # Agent will extract category
            timestamp=datetime.now(UTC).isoformat(),
        )

        # Store request ID for tracking
        pending_requests = ctx.storage.get("pending_requests") or {}
        pending_requests[request.request_id] = {
            "query": query,
            "sent_at": datetime.now(UTC).isoformat(),
            "received": False,
        }
        ctx.storage.set("pending_requests", pending_requests)

        ctx.logger.info(f"📤 Sending FindSupplierRequest...")
        ctx.logger.info(f"Request ID: {request.request_id}")
        ctx.logger.info(f"Query: {request.user_query}")

        # Send request to find_supplier agent
        await ctx.send(FIND_SUPPLIER_AGENT_ADDRESS, request)

        ctx.logger.info(f"✅ Request sent! Waiting for response...\n")

        # Wait between requests
        await asyncio.sleep(3)

    ctx.logger.info("\n" + "=" * 70)
    ctx.logger.info("All test requests sent! Waiting for responses...")
    ctx.logger.info("Press Ctrl+C to stop after receiving all responses")
    ctx.logger.info("=" * 70)


@test_protocol.on_message(model=FindSupplierResponse)
async def handle_find_supplier_response(
    ctx: Context, sender: str, msg: FindSupplierResponse
):
    """Handle response from find_supplier agent"""
    ctx.logger.info("\n" + "=" * 70)
    ctx.logger.info("📨 RECEIVED FIND SUPPLIER RESPONSE")
    ctx.logger.info("=" * 70)
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Success: {msg.success}")
    ctx.logger.info(f"Category: {msg.search_category}")
    ctx.logger.info(f"Total Results: {msg.total_results_found}")

    if msg.success and msg.best_supplier:
        ctx.logger.info("\n✅ BEST SUPPLIER FOUND:")
        ctx.logger.info(f"  Company: {msg.best_supplier.company_name}")
        ctx.logger.info(f"  Location: {msg.best_supplier.location}")
        ctx.logger.info(f"  Industry: {msg.best_supplier.industry}")
        ctx.logger.info(f"  Description: {msg.best_supplier.description}")
        ctx.logger.info(f"  Profile URL: {msg.best_supplier.b_corp_profile_url}")
        ctx.logger.info(f"\n  Summary: {msg.search_summary}")
    else:
        ctx.logger.error(f"\n❌ SEARCH FAILED:")
        ctx.logger.error(f"  Error: {msg.error_message}")

    # Mark as received
    pending_requests = ctx.storage.get("pending_requests") or {}
    if msg.request_id in pending_requests:
        pending_requests[msg.request_id]["received"] = True
        pending_requests[msg.request_id]["received_at"] = datetime.now(UTC).isoformat()
        ctx.storage.set("pending_requests", pending_requests)

    # Check if all responses received
    all_received = all(req["received"] for req in pending_requests.values())

    ctx.logger.info("\n" + "=" * 70)
    ctx.logger.info(
        f"Responses: {sum(1 for r in pending_requests.values() if r['received'])}/{len(pending_requests)}"
    )

    if all_received:
        ctx.logger.info("\n✅ ALL TEST RESPONSES RECEIVED!")
        ctx.logger.info("=" * 70)
        ctx.logger.info("\nTest Summary:")
        for req_id, req_data in pending_requests.items():
            ctx.logger.info(f"  • {req_data['query']}: ✅")
        ctx.logger.info("\n🎉 All tests completed successfully!")
        ctx.logger.info("Press Ctrl+C to exit")

    ctx.logger.info("=" * 70 + "\n")


test_client.include(test_protocol, publish_manifest=True)

if __name__ == "__main__":
    print(
        """
╔════════════════════════════════════════════════════════════════════╗
║                  FIND SUPPLIER AGENT TEST SUITE                    ║
╚════════════════════════════════════════════════════════════════════╝

This script will test the find_supplier agent by sending test queries.

Prerequisites:
1. ✅ find_supplier_agent.py must be running (Terminal 1)
2. ✅ FIND_SUPPLIER_AGENT_ADDRESS must be set in .env

Test Queries:
  • "I need a coffee supplier"
  • "Find me a pizza restaurant"  
  • "Looking for sustainable chocolate"

Starting test client...
    """
    )

    test_client.run()
