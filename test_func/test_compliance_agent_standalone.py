"""
Standalone test for Compliance Agent
Tests the compliance analysis workflow using the EXACT same logic as the actual agent

Usage:
    python -m test_func.test_compliance_agent_standalone
"""

import asyncio
from typing import Dict, Any
from datetime import datetime
import uuid

# Import the actual models used by the agent
from models.compliance import ComplianceRequest, ComplianceResponse


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


class TestComplianceAgent:
    """Test Compliance Agent using EXACT same logic as actual agent"""

    def __init__(self):
        self.ctx = MockContext()
        self.rag_system = None
        self.workflow = None

    async def initialize(self):
        """Initialize RAG system and workflow (mimics agent startup)"""
        from langgraph_logic.compliance_langgraph import (
            ComplianceRAGSystem,
            build_compliance_workflow,
        )

        print("🔧 Initializing Compliance Agent (like agent startup)...")

        # Initialize RAG system
        if self.rag_system is None:
            self.rag_system = ComplianceRAGSystem()

        success = await self.rag_system.initialize(self.ctx)

        if success:
            self.ctx.logger.info("Compliance Agent ready with RAG system!")

            # Build LangGraph workflow
            if self.workflow is None:
                self.workflow = build_compliance_workflow(self.rag_system, self.ctx)
                self.ctx.logger.info("LangGraph compliance workflow built!")
        else:
            self.ctx.logger.warning(
                "Compliance Agent running in fallback mode (no RAG)"
            )

        # Initialize state storage (like agent does)
        self.ctx.storage.set("supplier_history", [])
        self.ctx.storage.set("current_supplier", None)
        self.ctx.storage.set("previous_supplier", None)
        self.ctx.storage.set("processed_request_ids", [])

        print("✅ Compliance Agent initialized\n")

    async def handle_compliance_request(
        self, msg: ComplianceRequest
    ) -> ComplianceResponse:
        """
        Handle compliance request using EXACT same logic as actual agent
        (Mirrors: agents/supplier_search/compliance_agent.py::handle_compliance_request)
        """
        sender = "test_orchestrator"

        # === LOGGING (same as actual agent) ===
        self.ctx.logger.info("")
        self.ctx.logger.info("=" * 70)
        self.ctx.logger.info("📥 COMPLIANCE AGENT: RECEIVED REQUEST")
        self.ctx.logger.info("=" * 70)
        self.ctx.logger.info(f"From: {sender}")
        self.ctx.logger.info(f"Request ID: {msg.request_id}")
        self.ctx.logger.info(f"Supplier Name: {msg.supplier_name}")
        self.ctx.logger.info(f"Industry: {msg.industry}")
        self.ctx.logger.info(f"Company Values: {msg.company_values[:50]}...")
        self.ctx.logger.info(f"B Corp Profile URL: {msg.b_corp_profile_url}")
        self.ctx.logger.info("=" * 70)

        # === STATE MANAGEMENT: Move current to previous ===
        previous_supplier = self.ctx.storage.get("current_supplier")
        if previous_supplier:
            self.ctx.storage.set("previous_supplier", previous_supplier)

        # === STATE MANAGEMENT: Set new current supplier ===
        current_supplier_info = {
            "request_id": msg.request_id,
            "supplier_name": msg.supplier_name,
            "industry": msg.industry,
            "company_values": msg.company_values,
            "timestamp": msg.timestamp,
            "sender": sender,
        }
        self.ctx.storage.set("current_supplier", current_supplier_info)

        try:
            # === USING LANGGRAPH WORKFLOW WITH RAG (same as actual agent) ===
            from langgraph_logic.state_schemas import SupplierWorkflowState

            if self.workflow is None:
                self.ctx.logger.error("❌ Compliance workflow not initialized!")
                raise RuntimeError("Compliance workflow not ready")

            self.ctx.logger.info("🔄 Starting LangGraph compliance workflow...")

            # Create workflow state from request (EXACT same as actual agent)
            workflow_state = SupplierWorkflowState(
                request_id=msg.request_id,
                timestamp=msg.timestamp,
                user_input="",
                business_type="",
                company_values=msg.company_values,
                industry=msg.industry,
                product_needed="",
                supplier_name=msg.supplier_name,
                supplier_location=None,
                supplier_country=None,
                b_corp_profile_url=(
                    msg.b_corp_profile_url if msg.b_corp_profile_url else None
                ),
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
                current_step="compliance_check",
                error_message=None,
                should_continue=True,
                messages=[],
            )

            # Invoke workflow (same as actual agent)
            result = self.workflow.invoke(workflow_state)

            self.ctx.logger.info(f"Result variable on line 186: {result}")

            self.ctx.logger.info("✅ LangGraph workflow completed")

            # Extract results (same as actual agent)
            compliance_score = result.get("compliance_score", 75.0)
            sustainability_info = result.get("sustainability_info", "")
            ethics_info = result.get("ethics_info", "")
            violations = result.get("violations", [])

            self.ctx.logger.info(f"Compliance Analysis Results:")
            self.ctx.logger.info(f"  - Score: {compliance_score}/100")
            self.ctx.logger.info(f"  - Violations: {len(violations)}")
            self.ctx.logger.info(
                f"  - Status: {'APPROVED' if compliance_score >= 75 else 'REJECTED'}"
            )

            # Build response (same as actual agent)
            response = ComplianceResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                compliance_score=compliance_score,
                violations=violations if violations else [],
                sustainability_info=sustainability_info,
                ethics_info=ethics_info,
                timestamp=datetime.now().isoformat(),
            )

            # Update state with results (same as actual agent)
            current_step = (
                "compliance_approved"
                if compliance_score >= 75
                else "compliance_rejected"
            )
            current_supplier_info["compliance_score"] = response.compliance_score
            current_supplier_info["violations"] = response.violations
            current_supplier_info["sustainability_info"] = response.sustainability_info
            current_supplier_info["ethics_info"] = response.ethics_info
            current_supplier_info["current_step"] = current_step
            self.ctx.logger.info(f"Response variabvle on line 223:{response}")

            self.ctx.logger.info(f"Current supplier info(this is on line 226): {current_supplier_info}")
            self.ctx.storage.set("current_supplier", current_supplier_info)

            self.ctx.logger.info("")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info("📤 COMPLIANCE AGENT: SENDING RESPONSE")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info(f"To: {sender}")
            self.ctx.logger.info(f"Request ID: {msg.request_id}")
            self.ctx.logger.info(f"Compliance Score: {response.compliance_score}/100")
            self.ctx.logger.info(f"Violations: {len(response.violations)}")
            self.ctx.logger.info("=" * 70)
            self.ctx.logger.info("")

            return response

        except Exception as e:
            self.ctx.logger.error(f"❌ Error in compliance analysis: {e}")
            import traceback

            traceback.print_exc()

            # Return error response
            return ComplianceResponse(
                request_id=msg.request_id,
                supplier_name=msg.supplier_name,
                compliance_score=0.0,
                violations=["analysis_error"],
                sustainability_info=f"Error: {str(e)}",
                ethics_info="",
                timestamp=datetime.now().isoformat(),
            )


async def run_tests():
    """Run all compliance agent tests using EXACT agent logic"""
    tester = TestComplianceAgent()

    # Initialize agent (like startup event)
    await tester.initialize()

    # Test 1: Real B Corp Company - NOMAD COFFEE SL (Spain)
    print("\n" + "#" * 80)
    print("TEST 1: NOMAD COFFEE SL (Real B Corp from Spain)")
    print("#" * 80)
    request1 = ComplianceRequest(
        request_id=str(uuid.uuid4()),
        supplier_name="NOMAD COFFEE SL",
        industry="Coffee",
        company_values="Sustainable specialty coffee roaster",
        b_corp_profile_url="https://www.bcorporation.net/en-us/find-a-b-corp/company/nomad-coffee-sl/",
        timestamp=datetime.now().isoformat(),
    )
    response1 = await tester.handle_compliance_request(request1)
    print(f"Response 1: {response1}")
    print(
        f"\n✅ Test 1 Result: Score={response1.compliance_score}/100, Status={'PASS' if response1.compliance_score >= 75 else 'FAIL'}"
    )
    print(f"   Violations: {len(response1.violations)}")

    await asyncio.sleep(3)
    # Test 2: Another real B Corp (uncomment to test)
    # Find more B Corps at: https://www.bcorporation.net/en-us/find-a-b-corp/
    # request2 = ComplianceRequest(
    #     request_id=str(uuid.uuid4()),
    #     supplier_name="Your Company Name",
    #     industry="Your Industry",
    #     company_values="Company description",
    #     b_corp_profile_url="https://www.bcorporation.net/en-us/find-a-b-corp/company/YOUR-COMPANY/",
    #     timestamp=datetime.now().isoformat(),
    # )
    # response2 = await tester.handle_compliance_request(request2)
    # print(f"\n✅ Test 2 Result: Score={response2.compliance_score}/100")

    print("\n" + "=" * 80)
    print("🎉 ALL COMPLIANCE AGENT TESTS COMPLETED")
    print("=" * 80)
    print(
        f"\nProcessed {len(tester.ctx.storage.get('supplier_history', []))} suppliers total"
    )


if __name__ == "__main__":
    print("Starting Compliance Agent Standalone Tests...")
    asyncio.run(run_tests())
