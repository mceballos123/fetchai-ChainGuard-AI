"""
Integrated test for all three analysis agents (Financial, Compliance, Risk)
Tests all agents working together on the same supplier

Usage:
    python -m test_func.test_all_agents_integrated
"""

import asyncio
from typing import Dict, Any
from datetime import datetime


class MockContext:
    """Mock Context for testing"""

    def __init__(self):
        self.logs = []

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


class IntegratedAgentTest:
    """Test all three agents on the same supplier"""

    def __init__(self):
        self.ctx = MockContext()

    async def run_full_analysis(
        self,
        supplier_name: str,
        supplier_country: str,
        industry: str,
        company_values: str,
        b_corp_url: str,
    ):
        """
        Run complete analysis with all three agents

        Args:
            supplier_name: Name of the supplier
            supplier_country: Country where supplier operates
            industry: Supplier industry
            company_values: Company values/description
            b_corp_url: B Corp profile URL
        """
        print("\n" + "=" * 100)
        print("🔬 INTEGRATED AGENT TEST - FULL SUPPLIER ANALYSIS")
        print("=" * 100)
        print(f"Supplier: {supplier_name}")
        print(f"Country: {supplier_country}")
        print(f"Industry: {industry}")
        print(f"B Corp URL: {b_corp_url}")
        print("=" * 100 + "\n")

        results = {
            "supplier_name": supplier_name,
            "supplier_country": supplier_country,
            "industry": industry,
            "financial": None,
            "compliance": None,
            "risk": None,
        }

        # ========================================================================
        # AGENT 1: FINANCIAL ANALYSIS
        # ========================================================================
        print("\n" + "█" * 100)
        print("🏦 AGENT 1: FINANCIAL ANALYSIS")
        print("█" * 100 + "\n")

        try:
            from langgraph_logic.financial_langgraph import (
                FinancialRAGSystem,
                build_financial_workflow,
            )
            from langgraph_logic.state_schemas import SupplierWorkflowState

            print("Initializing Financial RAG System...")
            financial_rag = FinancialRAGSystem()
            await financial_rag.initialize(self.ctx)

            print("Building Financial Workflow...")
            financial_workflow = build_financial_workflow(financial_rag, self.ctx)

            print("Running Financial Analysis...\n")
            financial_state: SupplierWorkflowState = {
                "supplier_name": supplier_name,
                "supplier_country": supplier_country,
                "industry": industry,
                "b_corp_profile_url": b_corp_url,
                "current_step": "start",
                "error_message": "",
                "should_continue": True,
                "financial_score": 0.0,
                "financial_info": "",
                "risk_factors": [],
            }

            financial_result = financial_workflow.invoke(financial_state)
            results["financial"] = financial_result

            print("\n✅ Financial Analysis Complete")
            print(f"   Score: {financial_result.get('financial_score', 0)}/100")
            print(f"   Status: {financial_result.get('current_step', 'unknown')}")

        except Exception as e:
            print(f"\n❌ Financial Analysis Failed: {e}")
            import traceback

            traceback.print_exc()

        await asyncio.sleep(2)

        # ========================================================================
        # AGENT 2: COMPLIANCE ANALYSIS
        # ========================================================================
        print("\n" + "█" * 100)
        print("✅ AGENT 2: COMPLIANCE ANALYSIS")
        print("█" * 100 + "\n")

        try:
            from langgraph_logic.compliance_langgraph import (
                ComplianceRAGSystem,
                build_compliance_workflow,
            )

            print("Initializing Compliance RAG System...")
            compliance_rag = ComplianceRAGSystem()
            await compliance_rag.initialize(self.ctx)

            print("Building Compliance Workflow...")
            compliance_workflow = build_compliance_workflow(compliance_rag, self.ctx)

            print("Running Compliance Analysis...\n")
            compliance_state: SupplierWorkflowState = {
                "supplier_name": supplier_name,
                "industry": industry,
                "company_values": company_values,
                "b_corp_profile_url": b_corp_url,
                "current_step": "start",
                "error_message": "",
                "should_continue": True,
                "compliance_score": 0.0,
                "compliance_info": "",
                "violations": [],
                "sustainability_info": "",
                "ethics_info": "",
            }

            compliance_result = compliance_workflow.invoke(compliance_state)
            results["compliance"] = compliance_result

            print("\n✅ Compliance Analysis Complete")
            print(f"   Score: {compliance_result.get('compliance_score', 0)}/100")
            print(f"   Status: {compliance_result.get('current_step', 'unknown')}")

        except Exception as e:
            print(f"\n❌ Compliance Analysis Failed: {e}")
            import traceback

            traceback.print_exc()

        await asyncio.sleep(2)

        # ========================================================================
        # AGENT 3: RISK ANALYSIS
        # ========================================================================
        print("\n" + "█" * 100)
        print("⚠️  AGENT 3: RISK ANALYSIS")
        print("█" * 100 + "\n")

        try:
            from langgraph_logic.risk_langgraph import (
                RiskRAGSystem,
                build_risk_workflow,
            )

            print("Initializing Risk RAG System...")
            risk_rag = RiskRAGSystem()
            await risk_rag.initialize(self.ctx)

            print("Building Risk Workflow...")
            risk_workflow = build_risk_workflow(risk_rag, self.ctx)

            print("Running Risk Analysis...\n")
            risk_state: SupplierWorkflowState = {
                "supplier_name": supplier_name,
                "industry": industry,
                "b_corp_profile_url": b_corp_url,
                "current_step": "start",
                "error_message": "",
                "should_continue": True,
                "risk_score": 0.0,
                "risk_info": "",
                "risk_factors": [],
            }

            risk_result = risk_workflow.invoke(risk_state)
            results["risk"] = risk_result

            print("\n✅ Risk Analysis Complete")
            print(f"   Score: {risk_result.get('risk_score', 0)}/100")
            print(f"   Status: {risk_result.get('current_step', 'unknown')}")

        except Exception as e:
            print(f"\n❌ Risk Analysis Failed: {e}")
            import traceback

            traceback.print_exc()

        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================
        print("\n" + "=" * 100)
        print("📊 FINAL INTEGRATED ANALYSIS SUMMARY")
        print("=" * 100)
        print(f"\nSupplier: {supplier_name}")
        print(f"Country: {supplier_country}")
        print(f"Industry: {industry}\n")

        print("-" * 100)
        print("SCORES:")
        print("-" * 100)
        if results["financial"]:
            print(
                f"🏦 Financial Score:   {results['financial'].get('financial_score', 0):.1f}/100"
            )
        if results["compliance"]:
            print(
                f"✅ Compliance Score:  {results['compliance'].get('compliance_score', 0):.1f}/100"
            )
        if results["risk"]:
            print(
                f"⚠️  Risk Score:        {results['risk'].get('risk_score', 0):.1f}/100"
            )

        # Calculate overall score (average of all three)
        scores = []
        if results["financial"]:
            scores.append(results["financial"].get("financial_score", 0))
        if results["compliance"]:
            scores.append(results["compliance"].get("compliance_score", 0))
        if results["risk"]:
            scores.append(results["risk"].get("risk_score", 0))

        if scores:
            overall_score = sum(scores) / len(scores)
            print(f"\n{'🎯 OVERALL SCORE:':<20} {overall_score:.1f}/100")

            # Pass/Fail determination (threshold: 60)
            threshold = 60
            passed = all(score >= threshold for score in scores)
            print(
                f"{'Status:':<20} {'✅ PASS' if passed else '❌ FAIL'} (Threshold: {threshold})"
            )

        print("\n" + "-" * 100)
        print("RISK FACTORS:")
        print("-" * 100)
        all_risk_factors = set()
        if results["financial"]:
            all_risk_factors.update(results["financial"].get("risk_factors", []))
        if results["compliance"]:
            all_risk_factors.update(results["compliance"].get("violations", []))
        if results["risk"]:
            all_risk_factors.update(results["risk"].get("risk_factors", []))

        if all_risk_factors:
            for factor in all_risk_factors:
                print(f"  • {factor}")
        else:
            print("  • No significant risk factors identified")

        print("=" * 100 + "\n")

        return results


async def run_integrated_tests():
    """Run integrated tests on real suppliers"""
    tester = IntegratedAgentTest()

    # Test 1: NOMAD COFFEE SL (Spain) - Real B Corp
    print("\n" + "#" * 100)
    print("INTEGRATED TEST 1: NOMAD COFFEE SL (Spain)")
    print("#" * 100)
    await tester.run_full_analysis(
        supplier_name="NOMAD COFFEE SL",
        supplier_country="Spain",
        industry="Coffee",
        company_values="Sustainable specialty coffee roaster committed to quality and transparency",
        b_corp_url="https://www.bcorporation.net/en-us/find-a-b-corp/company/nomad-coffee-sl/",
    )

    await asyncio.sleep(3)

    # Test 2: You can add more tests here
    # Find B Corps at: https://www.bcorporation.net/en-us/find-a-b-corp/

    print("\n" + "=" * 100)
    print("🎉 ALL INTEGRATED TESTS COMPLETED")
    print("=" * 100)


if __name__ == "__main__":
    print("Starting Integrated Agent Tests...")
    print("This will test Financial, Compliance, and Risk agents together\n")
    asyncio.run(run_integrated_tests())
