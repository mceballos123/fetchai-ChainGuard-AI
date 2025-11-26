"""
Test script to simulate communication between find_supplier_agent, compliance_agent, and risk_agent.

Flow:
1. Find Supplier Agent searches B Corp directory
2. Returns supplier name + B Corp URL
3. Compliance Agent & Risk Agent receive supplier info (IN PARALLEL)
4. Both agents scrape B Corp webpage (Governance, Workers, Community, Environment, Customers)
5. Both convert to embeddings and store in Pinecone
6. Both run RAG analysis
7. Compliance returns score (>= 60 passes)
8. Risk returns score (>= 60 passes)
9. Supplier is approved ONLY if BOTH compliance AND risk pass
"""

from uagents import Agent, Context, Protocol
from models.find_supplier import FindSupplierRequest, FindSupplierResponse
from models.compliance import ComplianceRequest, ComplianceResponse
from models.risk import RiskRequest, RiskResponse
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
RISK_AGENT_ADDRESS = os.getenv("RISK_AGENT_ADDRESS")

if not FIND_SUPPLIER_AGENT_ADDRESS:
    print("Error: FIND_SUPPLIER_ADDRESS not set in .env")
    exit(1)

if not COMPLIANCE_AGENT_ADDRESS:
    print("Error: COMPLIANCE_AGENT_ADDRESS not set in .env")
    exit(1)

if not RISK_AGENT_ADDRESS:
    print("Error: RISK_AGENT_ADDRESS not set in .env")
    exit(1)

test_protocol = Protocol(name="test_orchestrator_protocol", version="1.0")


@test_orchestrator.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info("=" * 80)
    ctx.logger.info("TEST ORCHESTRATOR: SUPPLIER → COMPLIANCE + RISK (PARALLEL)")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"Orchestrator Address: {ctx.agent.address}")
    ctx.logger.info(f"Find Supplier Agent: {FIND_SUPPLIER_AGENT_ADDRESS}")
    ctx.logger.info(f"Compliance Agent: {COMPLIANCE_AGENT_ADDRESS}")
    ctx.logger.info(f"Risk Agent: {RISK_AGENT_ADDRESS}")
    ctx.logger.info("=" * 80)

    await asyncio.sleep(2)

    # Initialize storage with proper defaults
    ctx.storage.set("current_request_id", "")
    ctx.storage.set("supplier_found", False)
    ctx.storage.set("compliance_checked", False)
    ctx.storage.set("risk_checked", False)
    ctx.storage.set(
        "found_supplier", {}
    )  # Initialize as empty dict to prevent corruption
    ctx.storage.set("compliance_score", 0.0)
    ctx.storage.set("risk_score", 0.0)

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

        # Step 3: Send to BOTH compliance and risk agents (IN PARALLEL)
        ctx.logger.info(f"\n{'=' * 80}")
        ctx.logger.info("STEP 3: CHECKING COMPLIANCE & RISK (PARALLEL)")
        ctx.logger.info(f"{'=' * 80}")

        # Compliance Request
        compliance_request = ComplianceRequest(
            request_id=msg.request_id,
            supplier_name=supplier.company_name,
            industry=supplier.industry,
            company_values="ethical sourcing, sustainability, fair trade",
            b_corp_profile_url=supplier.b_corp_profile_url,
            timestamp="",
        )

        # Risk Request
        risk_request = RiskRequest(
            request_id=msg.request_id,
            supplier_name=supplier.company_name,
            industry=supplier.industry,
            b_corp_profile_url=supplier.b_corp_profile_url,
            timestamp="",
        )

        ctx.logger.info(f"Sending parallel requests for: {supplier.company_name}")
        ctx.logger.info(f"B Corp URL: {supplier.b_corp_profile_url}")
        ctx.logger.info("→ Compliance Agent: Will scrape & analyze compliance...")
        ctx.logger.info("→ Risk Agent: Will scrape & analyze risk...")

        # Send both requests in parallel
        await ctx.send(COMPLIANCE_AGENT_ADDRESS, compliance_request)
        await ctx.send(RISK_AGENT_ADDRESS, risk_request)
    else:
        ctx.logger.error(f"✗ Supplier search failed: {msg.error_message}")
        ctx.logger.info("\nTest failed at Step 1. Press Ctrl+C to exit")


@test_protocol.on_message(model=ComplianceResponse)
async def handle_compliance_response(
    ctx: Context, sender: str, msg: ComplianceResponse
):
    """Handle response from compliance_agent"""
    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("STEP 4A: COMPLIANCE RESULTS")
    ctx.logger.info(f"{'=' * 80}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier: {msg.supplier_name}")
    ctx.logger.info(f"Compliance Score: {msg.compliance_score}/100")

    # Check if passed (>= 60)
    compliance_passed = msg.compliance_score >= 60

    if compliance_passed:
        ctx.logger.info(f"✓ COMPLIANCE STATUS: PASSED (Score >= 60)")
    else:
        ctx.logger.info(f"✗ COMPLIANCE STATUS: FAILED (Score < 60)")

    ctx.logger.info(f"\nEthics Info: {msg.ethics_info[:200]}...")
    ctx.logger.info(f"\nSustainability Info: {msg.sustainability_info[:200]}...")
    ctx.logger.info(f"\nViolations: {msg.violations}")

    ctx.storage.set("compliance_checked", True)
    ctx.storage.set("compliance_score", msg.compliance_score)

    # Check if we have both results
    await check_final_results(ctx)


@test_protocol.on_message(model=RiskResponse)
async def handle_risk_response(ctx: Context, sender: str, msg: RiskResponse):
    """Handle response from risk_agent"""
    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("STEP 4B: RISK RESULTS")
    ctx.logger.info(f"{'=' * 80}")
    ctx.logger.info(f"Request ID: {msg.request_id}")
    ctx.logger.info(f"Supplier: {msg.supplier_name}")
    ctx.logger.info(f"Risk Score: {msg.risk_score}/100")

    # Check if passed (>= 60)
    risk_passed = msg.risk_score >= 60

    if risk_passed:
        ctx.logger.info(f"✓ RISK STATUS: PASSED (Score >= 60)")
    else:
        ctx.logger.info(f"✗ RISK STATUS: FAILED (Score < 60)")

    ctx.logger.info(f"\nRisk Details: {msg.risk_details[:200]}...")
    ctx.logger.info(f"\nRisk Factors: {msg.risk_factors}")

    ctx.storage.set("risk_checked", True)
    ctx.storage.set("risk_score", msg.risk_score)

    # Check if we have both results
    await check_final_results(ctx)


async def check_final_results(ctx: Context):
    """Check if both compliance and risk results are received, then display final summary"""
    compliance_checked = ctx.storage.get("compliance_checked")
    risk_checked = ctx.storage.get("risk_checked")

    # Only show summary when BOTH results are in
    if not (compliance_checked and risk_checked):
        ctx.logger.info("Waiting for other agent response...")
        return

    # Both results are in - show final summary
    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("FINAL SUMMARY - PARALLEL EVALUATION")
    ctx.logger.info(f"{'=' * 80}")

    supplier_found = ctx.storage.get("supplier_found")
    found_supplier = ctx.storage.get("found_supplier")
    compliance_score = ctx.storage.get("compliance_score")
    risk_score = ctx.storage.get("risk_score")

    if supplier_found and found_supplier:
        supplier_name = (
            found_supplier.get("company_name", "Unknown")
            if isinstance(found_supplier, dict)
            else "Unknown"
        )
        ctx.logger.info(f"✓ Step 1: Supplier Found - {supplier_name}")
    else:
        ctx.logger.info(f"✗ Step 1: Supplier Not Found")

    # Check individual results
    compliance_passed = compliance_score >= 60
    risk_passed = risk_score >= 60

    if compliance_checked:
        if compliance_passed:
            ctx.logger.info(f"✓ Step 2A: Compliance PASSED - Score: {compliance_score}/100")
        else:
            ctx.logger.info(f"✗ Step 2A: Compliance FAILED - Score: {compliance_score}/100")
    else:
        ctx.logger.info(f"✗ Step 2A: Compliance Not Checked")

    if risk_checked:
        if risk_passed:
            ctx.logger.info(f"✓ Step 2B: Risk PASSED - Score: {risk_score}/100")
        else:
            ctx.logger.info(f"✗ Step 2B: Risk FAILED - Score: {risk_score}/100")
    else:
        ctx.logger.info(f"✗ Step 2B: Risk Not Checked")

    # Final decision: BOTH must pass
    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("FINAL DECISION")
    ctx.logger.info(f"{'=' * 80}")

    if compliance_passed and risk_passed:
        ctx.logger.info(f"✓✓ SUPPLIER APPROVED ✓✓")
        ctx.logger.info(f"Both Compliance (>= 60) and Risk (>= 60) passed!")
    elif compliance_passed and not risk_passed:
        ctx.logger.info(f"✗ SUPPLIER REJECTED")
        ctx.logger.info(f"Compliance passed but Risk failed (< 60)")
    elif not compliance_passed and risk_passed:
        ctx.logger.info(f"✗ SUPPLIER REJECTED")
        ctx.logger.info(f"Risk passed but Compliance failed (< 60)")
    else:
        ctx.logger.info(f"✗ SUPPLIER REJECTED")
        ctx.logger.info(f"Both Compliance and Risk failed (< 60)")

    ctx.logger.info(f"\n{'=' * 80}")
    ctx.logger.info("TEST COMPLETED SUCCESSFULLY!")
    ctx.logger.info(f"{'=' * 80}")
    ctx.logger.info("Press Ctrl+C to exit")


test_orchestrator.include(test_protocol, publish_manifest=True)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TEST: FIND SUPPLIER → COMPLIANCE + RISK AGENTS (PARALLEL)")
    print("=" * 80)
    print("\nThis test simulates:")
    print("1. Finding a supplier from B Corp directory")
    print("2. Sending supplier to BOTH Compliance & Risk agents (in parallel)")
    print("3. Both agents scrape the B Corp webpage")
    print("4. Both convert to embeddings and store in Pinecone")
    print("5. Both run RAG analysis")
    print("6. Compliance checks ethics/sustainability (>= 60 passes)")
    print("7. Risk checks operational/supply chain risk (>= 60 passes)")
    print("8. Supplier APPROVED only if BOTH pass (>= 60)")
    print("\n" + "=" * 80)
    print("\nStarting test orchestrator...")
    print("Make sure these agents are running:")
    print("  - find_supplier_agent.py")
    print("  - compliance_agent.py")
    print("  - risk_agent.py")
    print("=" * 80 + "\n")

    test_orchestrator.run()
