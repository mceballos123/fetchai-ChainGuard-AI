"""
Test script to simulate communication between find_supplier_agent and compliance_agent.

Flow:
1. Find Supplier Agent searches B Corp directory
2. Returns supplier name + B Corp URL
3. Compliance Agent receives the supplier info
4. Scrapes B Corp webpage (Governance, Workers, Community, Environment, Customers)
5. Converts to embeddings and stores in Pinecone
6. Runs RAG analysis
7. Returns compliance score (>= 60 passes)
"""

from uagents import Agent, Context, Protocol
from models.find_supplier import FindSupplierRequest, FindSupplierResponse
from models.compliance import ComplianceRequest, ComplianceResponse
from uuid import uuid4
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Test orchestrator agent
test_orchestrator = Agent(
    name="test_orchestrator",
    seed=os.getenv("DUMMY_SUPPLIER_TO_COMPLIANCE_SEED"),
    port=8010,
    endpoint=["http://localhost:8010/submit"],
)

# Get agent addresses from environment
FIND_SUPPLIER_AGENT_ADDRESS = os.getenv("FIND_SUPPLIER_ADDRESS")
COMPLIANCE_AGENT_ADDRESS = os.getenv("COMPLIANCE_AGENT_ADDRESS")

if not FIND_SUPPLIER_AGENT_ADDRESS:
    print("Error: FIND_SUPPLIER_ADDRESS not set in .env")
    exit(1)

if not COMPLIANCE_AGENT_ADDRESS:
    print("Error: COMPLIANCE_AGENT_ADDRESS not set in .env")
    exit(1)

test_protocol = Protocol(name="test_orchestrator_protocol", version="1.0")


@test_orchestrator.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info("=" * 80)
    ctx.logger.info("TEST ORCHESTRATOR: SUPPLIER TO COMPLIANCE FLOW")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"Orchestrator Address: {ctx.agent.address}")
    ctx.logger.info(f"Find Supplier Agent: {FIND_SUPPLIER_AGENT_ADDRESS}")
    ctx.logger.info(f"Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
    ctx.logger.info("=" * 80)

    await asyncio.sleep(2)

    # Initialize storage with proper defaults
    ctx.storage.set("current_request_id", "")
    ctx.storage.set("supplier_found", False)
    ctx.storage.set("compliance_checked", False)
    ctx.storage.set(
        "found_supplier", {}
    )  # Initialize as empty dict to prevent corruption

    # Step 1: Send request to find_supplier_agent
    test_query = "I need a coffee supplier"
    request_id = str(uuid4())

    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("STEP 1: FINDING SUPPLIER")
    ctx.logger.info(f"{'=' * 80}")
    ctx.logger.info(f"Query: {test_query}")
    ctx.logger.info(f"Request ID: {request_id}")

    ctx.storage.set("current_request_id", request_id)

    find_supplier_request = FindSupplierRequest(
        request_id=request_id,
        user_query=test_query,
        business_category="general",
    )

    ctx.logger.info("Sending to Find Supplier Agent...")
    await ctx.send(FIND_SUPPLIER_AGENT_ADDRESS, find_supplier_request)


@test_protocol.on_message(model=FindSupplierResponse)
async def handle_find_supplier_response(
    ctx: Context, sender: str, msg: FindSupplierResponse
):
    """Handle response from find_supplier_agent"""
    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("STEP 2: SUPPLIER FOUND")
    ctx.logger.info(f"{'=' * 80}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Success: {msg.success}")

    if msg.success and msg.best_supplier:
        supplier = msg.best_supplier
        ctx.logger.info(f"✓ Supplier: {supplier.company_name}")
        ctx.logger.info(f"✓ Location: {supplier.location}")
        ctx.logger.info(f"✓ Industry: {supplier.industry}")
        ctx.logger.info(f"✓ B Corp URL: {supplier.b_corp_profile_url}")

        ctx.storage.set("supplier_found", True)

        # Store supplier data as a dictionary to prevent JSON corruption
        supplier_dict = {
            "company_name": supplier.company_name,
            "location": supplier.location,
            "industry": supplier.industry,
            "b_corp_profile_url": supplier.b_corp_profile_url,
            "description": supplier.description,
        }
        ctx.storage.set("found_supplier", supplier_dict)

        # Step 3: Send to compliance agent
        ctx.logger.info(f"\n{'=' * 80}")
        ctx.logger.info("STEP 3: CHECKING COMPLIANCE")
        ctx.logger.info(f"{'=' * 80}")

        compliance_request = ComplianceRequest(
            request_id=msg.request_id,
            supplier_name=supplier.company_name,
            industry=supplier.industry,
            company_values="ethical sourcing, sustainability, fair trade",
            b_corp_profile_url=supplier.b_corp_profile_url,
            timestamp="",
        )

        ctx.logger.info(f"Sending compliance request for: {supplier.company_name}")
        ctx.logger.info(f"B Corp URL: {supplier.b_corp_profile_url}")
        ctx.logger.info("This will scrape the B Corp page and analyze compliance...")

        await ctx.send(COMPLIANCE_AGENT_ADDRESS, compliance_request)
    else:
        ctx.logger.error(f"✗ Supplier search failed: {msg.error_message}")
        ctx.logger.info("\nTest failed at Step 1. Press Ctrl+C to exit")


@test_protocol.on_message(model=ComplianceResponse)
async def handle_compliance_response(
    ctx: Context, sender: str, msg: ComplianceResponse
):
    """Handle response from compliance_agent"""
    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("STEP 4: COMPLIANCE RESULTS")
    ctx.logger.info(f"{'=' * 80}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier: {msg.supplier_name}")
    ctx.logger.info(f"Compliance Score: {msg.compliance_score}/100")

    # Check if passed (>= 60)
    passed = msg.compliance_score >= 60

    if passed:
        ctx.logger.info(f"✓ STATUS: PASSED (Score >= 60)")
    else:
        ctx.logger.info(f"✗ STATUS: FAILED (Score < 60)")

    ctx.logger.info(f"\nEthics Info: {msg.ethics_info[:200]}...")
    ctx.logger.info(f"\nSustainability Info: {msg.sustainability_info[:200]}...")
    ctx.logger.info(f"\nViolations: {msg.violations}")

    ctx.storage.set("compliance_checked", True)

    # Summary
    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("TEST SUMMARY")
    ctx.logger.info(f"{'=' * 80}")

    supplier_found = ctx.storage.get("supplier_found")
    compliance_checked = ctx.storage.get("compliance_checked")
    found_supplier = ctx.storage.get("found_supplier")

    if supplier_found and found_supplier:
        supplier_name = (
            found_supplier.get("company_name", "Unknown")
            if isinstance(found_supplier, dict)
            else "Unknown"
        )
        ctx.logger.info(f"✓ Step 1: Supplier Found - {supplier_name}")
    else:
        ctx.logger.info(f"✗ Step 1: Supplier Not Found")

    if compliance_checked:
        ctx.logger.info(
            f"✓ Step 2: Compliance Checked - Score: {msg.compliance_score}/100"
        )
        if passed:
            ctx.logger.info(f"✓ Step 3: Compliance PASSED")
        else:
            ctx.logger.info(f"✗ Step 3: Compliance FAILED")
    else:
        ctx.logger.info(f"✗ Step 2: Compliance Not Checked")

    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("TEST COMPLETED SUCCESSFULLY!")
    ctx.logger.info(f"{'=' * 80}")
    ctx.logger.info("Press Ctrl+C to exit")


test_orchestrator.include(test_protocol, publish_manifest=True)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TEST: FIND SUPPLIER → COMPLIANCE AGENT")
    print("=" * 80)
    print("\nThis test simulates:")
    print("1. Finding a supplier from B Corp directory")
    print("2. Scraping the B Corp webpage for compliance data")
    print("3. Converting to embeddings and storing in Pinecone")
    print("4. Running RAG analysis for compliance")
    print("5. Checking if score >= 60 (passes)")
    print("\n" + "=" * 80)
    print("\nStarting test orchestrator...")
    print("Make sure find_supplier_agent.py and compliance_agent.py are running!")
    print("=" * 80 + "\n")

    test_orchestrator.run()
