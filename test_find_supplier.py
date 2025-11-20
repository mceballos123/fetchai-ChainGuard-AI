from uagents import Agent, Context, Protocol
from models.find_supplier import FindSupplierRequest, FindSupplierResponse
from uuid import uuid4
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

test_client = Agent(
    name="test_find_supplier_client",
    seed=os.getenv("DUMMY_FIND_SUPPLIER_AGENT_SEED"),
    port=8007,
    endpoint=["http://localhost:8007/submit"],
)

FIND_SUPPLIER_AGENT_ADDRESS = os.getenv("FIND_SUPPLIER_ADDRESS")

if not FIND_SUPPLIER_AGENT_ADDRESS:
    print("Error: FIND_SUPPLIER_AGENT_ADDRESS not set in .env")
    print("\nPlease:")
    print("1. Start find_supplier_agent.py first")
    print("2. Copy its address from logs")
    print("3. Add to .env file: FIND_SUPPLIER_AGENT_ADDRESS=agent1q...")
    exit(1)

test_protocol = Protocol(name="test_find_supplier_protocol", version="1.0")


@test_client.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info("Test client started")
    ctx.logger.info(f"Client: {ctx.agent.address}")
    ctx.logger.info(f"Target: {FIND_SUPPLIER_AGENT_ADDRESS}")

    await asyncio.sleep(2)

    test_queries = ["I need a tire supplier"]

    for idx, query in enumerate(test_queries, 1):
        ctx.logger.info(f"\nTest {idx}/{len(test_queries)}: {query}")

        request = FindSupplierRequest(
            request_id=str(uuid4()),
            user_query=query,
            business_category="general",
        )

        pending_requests = ctx.storage.get("pending_requests") or {}
        pending_requests[request.request_id] = {
            "query": query,
            "received": False,
        }
        ctx.storage.set("pending_requests", pending_requests)

        ctx.logger.info(f"Sending request {request.request_id}")
        await ctx.send(FIND_SUPPLIER_AGENT_ADDRESS, request)

        await asyncio.sleep(3)

    ctx.logger.info("\nAll requests sent. Waiting for responses...")


@test_protocol.on_message(model=FindSupplierResponse)
async def handle_find_supplier_response(
    ctx: Context, sender: str, msg: FindSupplierResponse
):
    ctx.logger.info(f"\nReceived response for {msg.request_id}")
    ctx.logger.info(f"Success: {msg.success}")

    if msg.success and msg.best_supplier:
        ctx.logger.info(f"\nSupplier: {msg.best_supplier.company_name}")
        ctx.logger.info(f"Location: {msg.best_supplier.location}")
        ctx.logger.info(f"Industry: {msg.best_supplier.industry}")
        ctx.logger.info(f"Summary: {msg.search_summary}")
    else:
        ctx.logger.error(f"Error: {msg.error_message}")

    pending_requests = ctx.storage.get("pending_requests") or {}
    if msg.request_id in pending_requests:
        pending_requests[msg.request_id]["received"] = True
        ctx.storage.set("pending_requests", pending_requests)

    all_received = all(req["received"] for req in pending_requests.values())
    completed = sum(1 for r in pending_requests.values() if r["received"])
    ctx.logger.info(f"\nResponses: {completed}/{len(pending_requests)}")

    if all_received:
        ctx.logger.info("\nAll tests completed! Press Ctrl+C to exit")


test_client.include(test_protocol, publish_manifest=True)

if __name__ == "__main__":
    test_client.run()
