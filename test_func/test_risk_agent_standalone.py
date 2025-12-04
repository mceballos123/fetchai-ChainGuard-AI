"""
Standalone test for Risk Agent
Tests the risk management workflow using the EXACT same logic as the actual agent

Usage:
    python -m test_func.test_risk_agent_standalone
"""

import asyncio
from typing import Dict, Any
from datetime import datetime
import uuid

# Import the actual models used by the agent
from models.risk import RiskRequest, RiskResponse


class MockContext:
    """Mock Context that mimics uagents Context"""

    def __init__(self):
        self.logs = []
        self._storage = {}

    class Storage:
        def __init__(self, parent):
            self.parent = parent

        def get(self, key: str, default=None):
            return self.parent._storage.get(key, default)

        def set(self, key: str, value):
            self.parent._storage[key] = value

    class Logger:
        def __init__(self, parent):
            self.parent = parent

        def info(self, msg):
            print(f"[INFO] {msg}")
            self.parent.logs.append(("INFO", msg))

        def error(self, msg):
            print(f"[ERROR] {msg}")
            self.parent.logs.append(("ERROR", msg))

        def warning(self, msg):
            print(f"[WARNING] {msg}")
            self.parent.logs.append(("WARNING", msg))

    @property
    def logger(self):
        return self.Logger(self)

    @property
    def storage(self):
        return self.Storage(self)


class TestRiskAgent:
    """Test Risk Agent using EXACT same logic as actual agent"""

    def __init__(self):
        self.ctx = MockContext()
        self.rag_system = None
        self.workflow = None

    async def initialize(self):
        """Initialize RAG system and workflow (mimics agent startup)"""
        from langgraph_logic.risk_langgraph import (
            RiskRAGSystem,
            build_risk_workflow,
        )

        print("🔧 Initializing Risk Agent (like agent startup)...")

        # Initialize RAG system
        if self.rag_system is None:
            self.rag_system = RiskRAGSystem()

        success = await self.rag_system.initialize(self.ctx)

        if success:
            self.ctx.logger.info("Risk Agent ready with RAG system!")

            # Build LangGraph workflow
            if self.workflow is None:
                self.workflow = build_risk_workflow(self.rag_system, self.ctx)
                self.ctx.logger.info("LangGraph risk workflow built!")
        else:
            self.ctx.logger.warning("Risk Agent running in fallback mode (no RAG)")

        # Initialize state storage (like agent does)
        self.ctx.storage.set("supplier_history", [])
        self.ctx.storage.set("current_supplier", None)
        self.ctx.storage.set("processed_request_ids", [])

        print("✅ Risk Agent initialized\n")

    async def handle_risk_request(self, msg: RiskRequest) -> RiskResponse:
        """
        Handle risk request using EXACT same logic as actual agent
        (Mirrors: agents/supplier_search/risk_agent.py::handle_risk_request)
        """
        sender = "test_orchestrator"

        # === LOGGING (same as actual agent) ===
        self.ctx.logger.info("")
        self.ctx.logger.info("=" * 70)
        self.ctx.logger.info("📥 RISK AGENT: RECEIVED REQUEST")
        self.ctx.logger.info("=" * 70)
        self.ctx.logger.info(f"From: {sender}")
        self.ctx.logger.info(f"Request ID: {msg.request_id}")
        self.ctx.logger.info(f"Supplier Name: {msg.supplier_name}")
        self.ctx.logger.info(f"Industry: {msg.industry}")
        self.ctx.logger.info(f"B Corp Profile URL: {msg.b_corp_profile_url}")
        self.ctx.logger.info("=" * 70)

        # === STATE MANAGEMENT: Update current supplier state ===
        current_supplier_info = {
            "request_id": msg.request_id,
            "supplier_name": msg.supplier_name,
            "industry": msg.industry,
            "timestamp": msg.timestamp,
            "sender": sender,
        }
        self.ctx.storage.set("current_supplier", current_supplier_info)

        try:
            # === USING LANGGRAPH WORKFLOW WITH RAG (same as actual agent) ===
            from langgraph_logic.state_schemas import SupplierWorkflowState

            if self.workflow is None:
                self.ctx.logger.error("❌ Risk workflow not initialized!")
                raise RuntimeError("Risk workflow not ready")

            self.ctx.logger.info("🔄 Starting LangGraph risk workflow...")
            self.ctx.logger.info(f"Will scrape B Corp profile for risk analysis")

            # Create workflow state from request (EXACT same as actual agent)
            workflow_state = SupplierWorkflowState(
                request_id=msg.request_id,
                timestamp=msg.timestamp,
                user_input="",
                business_type="",
                company_values="",
                industry=msg.industry,
                product_needed="",
                supplier_name=msg.supplier_name,
                supplier_location=None,
                supplier_country=None,
                b_corp_profile_url=msg.b_corp_profile_url,
                retrieved_documents=None,
                rag_context=None,
                compliance_score=None,
                ethics_info=None,
                sustainability_info=None,
                violations=None,
                financial_score=None,
                financial_info=None,
                risk_score=None,
                risk_details=None,
                risk_factors=None,
                current_step="risk_check",
                error_message=None,
                should_continue=True,
                messages=[],
            )

            # Invoke workflow (same as actual agent)
            result = self.workflow.invoke(workflow_state)

            self.ctx.logger.info(f"Result variable on line 173: {result}")

            self.ctx.logger.info("✅ LangGraph workflow completed")


            # Extract results (same as actual agent)
            risk_score = result.get("risk_score", 70.0)
            risk_details = result.get("risk_details", "Analysis complete")
            risk_factors = result.get("risk_factors", [])

            self.ctx.logger.info(f"Risk Analysis Results:")
            self.ctx.logger.info(f"  - Score: {risk_score}/100")
            self.ctx.logger.info(f"  - Risk Factors: {len(risk_factors)}")
            self.ctx.logger.info(
                f"  - Status: {'APPROVED' if risk_score >= 60 else 'REJECTED'}"
            )

            # Build response (same as actual agent)
            response = RiskResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                risk_score=risk_score,
                risk_details=risk_details,
                risk_factors=risk_factors if risk_factors else [],
                timestamp=datetime.now().isoformat(),
            )
            self.ctx.logger.info(f"Response variable on line 196: {response}")

            # Update state with results (same as actual agent)
            current_step = "risk_approved" if risk_score >= 60 else "risk_rejected"
            current_supplier_info["risk_score"] = response.risk_score
            current_supplier_info["risk_details"] = response.risk_details
            current_supplier_info["risk_factors"] = response.risk_factors
            current_supplier_info["current_step"] = current_step
            self.ctx.storage.set("current_supplier", current_supplier_info)

            # Add to history
            supplier_history = self.ctx.storage.get("supplier_history") or []
            supplier_history.append(current_supplier_info)
            self.ctx.storage.set("supplier_history", supplier_history)

            self.ctx.logger.info("")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info("📤 RISK AGENT: SENDING RESPONSE")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info(f"To: {sender}")
            self.ctx.logger.info(f"Request ID: {msg.request_id}")
            self.ctx.logger.info(f"Risk Score: {response.risk_score}/100")
            self.ctx.logger.info(f"Risk Factors: {len(response.risk_factors)}")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info("")

            return response

        except Exception as e:
            self.ctx.logger.error(f"❌ Error in risk analysis: {e}")
            import traceback

            traceback.print_exc()

            # Return error response
            return RiskResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                risk_score=0.0,
                risk_details=f"Error: {str(e)}",
                risk_factors=["analysis_error"],
                timestamp=datetime.now().isoformat(),
            )


async def run_tests():
    """Run all risk agent tests using EXACT agent logic"""
    tester = TestRiskAgent()

    # Initialize agent (like startup event)
    await tester.initialize()

    # Test 1: Real B Corp Company - NOMAD COFFEE SL (Spain)
    print("\n" + "#" * 80)
    print("TEST 1: NOMAD COFFEE SL (Real B Corp from Spain)")
    print("#" * 80)
    request1 = RiskRequest(
        request_id=str(uuid.uuid4()),
        supplier_name="NOMAD COFFEE SL",
        industry="Coffee",
        b_corp_profile_url="https://www.bcorporation.net/en-us/find-a-b-corp/company/nomad-coffee-sl/",
        timestamp=datetime.now().isoformat(),
    )
    response1 = await tester.handle_risk_request(request1)
    print(
        f"\n✅ Test 1 Result: Score={response1.risk_score}/100, Status={'PASS' if response1.risk_score >= 60 else 'FAIL'}"
    )
    print(f"   Risk Factors: {len(response1.risk_factors)}")

    await asyncio.sleep(3)

    # Test 2: Another real B Corp (uncomment to test)
    # Find more B Corps at: https://www.bcorporation.net/en-us/find-a-b-corp/
    # request2 = RiskRequest(
    #     request_id=str(uuid.uuid4()),
    #     supplier_name="Your Company Name",
    #     industry="Your Industry",
    #     b_corp_profile_url="https://www.bcorporation.net/en-us/find-a-b-corp/company/YOUR-COMPANY/",
    #     timestamp=datetime.now().isoformat(),
    # )
    # response2 = await tester.handle_risk_request(request2)
    # print(f"\n✅ Test 2 Result: Score={response2.risk_score}/100")

    print("\n" + "=" * 80)
    print("🎉 ALL RISK AGENT TESTS COMPLETED")
    print("=" * 80)
    print(
        f"\nProcessed {len(tester.ctx.storage.get('supplier_history', []))} suppliers total"
    )


if __name__ == "__main__":
    print("Starting Risk Agent Standalone Tests...")
    asyncio.run(run_tests())
