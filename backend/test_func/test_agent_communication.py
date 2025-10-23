"""
Test utility functions for verifying communication between Orchestrator and Compliance agents.

This module provides helper functions to test and verify:
1. Message transmission between agents
2. Request/Response format validation
3. Connection stability
4. Message trace tracking
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CommunicationTestResult:
    """Data class to hold communication test results"""

    test_name: str
    status: str  # "PASSED", "FAILED", "WARNING"
    timestamp: str
    duration_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None


class AgentCommunicationTester:
    """Helper class for testing inter-agent communication"""

    def __init__(self):
        self.test_results: List[CommunicationTestResult] = []
        self.message_log: List[Dict[str, Any]] = []

    def log_message(
        self,
        direction: str,
        agent_from: str,
        agent_to: str,
        message_type: str,
        message_id: str,
        payload: Dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Log a message exchange between agents.

        Args:
            direction: "SENT" or "RECEIVED"
            agent_from: Source agent name
            agent_to: Destination agent name
            message_type: Type of message (ComplianceRequest, ComplianceResponse, etc.)
            message_id: Unique message identifier
            payload: Message payload
            timestamp: Optional timestamp (uses current time if not provided)
        """
        entry = {
            "direction": direction,
            "agent_from": agent_from,
            "agent_to": agent_to,
            "message_type": message_type,
            "message_id": message_id,
            "payload": payload,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }
        self.message_log.append(entry)

    def verify_request_response_pair(
        self, request_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that a request has a corresponding response.

        Args:
            request_id: The request ID to verify

        Returns:
            Tuple of (success, details)
        """
        request_msg = None
        response_msg = None

        for msg in self.message_log:
            if msg["message_id"] == request_id:
                if msg["message_type"] == "ComplianceRequest":
                    request_msg = msg
                elif msg["message_type"] == "ComplianceResponse":
                    response_msg = msg

        success = request_msg is not None and response_msg is not None
        details = {
            "request_found": request_msg is not None,
            "response_found": response_msg is not None,
            "request": request_msg,
            "response": response_msg,
        }

        if success:
            # Calculate response time
            req_time = datetime.fromisoformat(request_msg["timestamp"])
            resp_time = datetime.fromisoformat(response_msg["timestamp"])
            details["response_time_ms"] = (resp_time - req_time).total_seconds() * 1000

        return success, details

    def verify_message_format(
        self, message: Dict[str, Any], message_type: str
    ) -> Tuple[bool, List[str]]:
        """
        Verify that a message has the correct format for its type.

        Args:
            message: The message to verify
            message_type: Expected message type

        Returns:
            Tuple of (valid, errors)
        """
        errors = []

        if message_type == "ComplianceRequest":
            required_fields = [
                "request_id",
                "supplier_name",
                "industry",
                "company_values",
            ]
            for field in required_fields:
                if field not in message.get("payload", {}):
                    errors.append(f"Missing required field: {field}")

            # Validate field types
            if not isinstance(message.get("payload", {}).get("request_id"), str):
                errors.append("request_id must be a string")
            if not isinstance(message.get("payload", {}).get("supplier_name"), str):
                errors.append("supplier_name must be a string")

        elif message_type == "ComplianceResponse":
            required_fields = [
                "request_id",
                "supplier_name",
                "compliance_score",
                "violations",
            ]
            for field in required_fields:
                if field not in message.get("payload", {}):
                    errors.append(f"Missing required field: {field}")

            # Validate field types
            payload = message.get("payload", {})
            if not isinstance(payload.get("compliance_score"), (int, float)):
                errors.append("compliance_score must be a number")
            if not isinstance(payload.get("violations"), list):
                errors.append("violations must be a list")

        return len(errors) == 0, errors

    def verify_agent_state(
        self, ctx_storage: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify agent state consistency.

        Args:
            ctx_storage: The context storage dictionary from agent

        Returns:
            Tuple of (valid, details)
        """
        details = {
            "has_active_sessions": "active_sessions" in ctx_storage,
            "has_message_trace": "message_trace" in ctx_storage
            or "request_trace" in ctx_storage,
            "sessions_count": 0,
            "trace_entries": 0,
        }

        if "active_sessions" in ctx_storage:
            details["sessions_count"] = len(ctx_storage["active_sessions"])

        if "message_trace" in ctx_storage:
            trace = ctx_storage["message_trace"]
            details["trace_entries"] = sum(
                len(v) if isinstance(v, list) else 0 for v in trace.values()
            )
        elif "request_trace" in ctx_storage:
            trace = ctx_storage["request_trace"]
            details["trace_entries"] = sum(
                len(v) if isinstance(v, list) else 0 for v in trace.values()
            )

        valid = details["has_active_sessions"] and details["has_message_trace"]
        return valid, details

    async def test_orchestrator_startup(self, ctx) -> CommunicationTestResult:
        """Test orchestrator agent startup and initialization"""
        start_time = datetime.utcnow()

        try:
            # Check if required storage keys are initialized
            active_sessions = ctx.storage.get("active_sessions")
            message_trace = ctx.storage.get("message_trace")

            success = active_sessions is not None and message_trace is not None

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = CommunicationTestResult(
                test_name="Orchestrator Startup",
                status="PASSED" if success else "FAILED",
                timestamp=start_time.isoformat(),
                duration_ms=duration,
                details={
                    "active_sessions_initialized": active_sessions is not None,
                    "message_trace_initialized": message_trace is not None,
                    "agent_address": ctx.agent.address,
                },
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            result = CommunicationTestResult(
                test_name="Orchestrator Startup",
                status="FAILED",
                timestamp=start_time.isoformat(),
                duration_ms=duration,
                details={},
                error=str(e),
            )
            self.test_results.append(result)
            return result

    async def test_compliance_agent_startup(self, ctx) -> CommunicationTestResult:
        """Test compliance agent startup and initialization"""
        start_time = datetime.utcnow()

        try:
            # Check if required storage keys are initialized
            supplier_history = ctx.storage.get("supplier_history")
            request_trace = ctx.storage.get("request_trace")

            success = supplier_history is not None and request_trace is not None

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = CommunicationTestResult(
                test_name="Compliance Agent Startup",
                status="PASSED" if success else "FAILED",
                timestamp=start_time.isoformat(),
                duration_ms=duration,
                details={
                    "supplier_history_initialized": supplier_history is not None,
                    "request_trace_initialized": request_trace is not None,
                    "agent_address": ctx.agent.address,
                },
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            result = CommunicationTestResult(
                test_name="Compliance Agent Startup",
                status="FAILED",
                timestamp=start_time.isoformat(),
                duration_ms=duration,
                details={},
                error=str(e),
            )
            self.test_results.append(result)
            return result

    async def test_request_transmission(
        self, request_id: str, supplier_name: str
    ) -> CommunicationTestResult:
        """Test successful transmission of compliance request"""
        start_time = datetime.utcnow()

        try:
            # This would be called after actual transmission
            request_found = any(
                msg["message_id"] == request_id
                and msg["message_type"] == "ComplianceRequest"
                for msg in self.message_log
            )

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = CommunicationTestResult(
                test_name="Request Transmission",
                status="PASSED" if request_found else "FAILED",
                timestamp=start_time.isoformat(),
                duration_ms=duration,
                details={
                    "request_id": request_id,
                    "supplier_name": supplier_name,
                    "request_found": request_found,
                },
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            result = CommunicationTestResult(
                test_name="Request Transmission",
                status="FAILED",
                timestamp=start_time.isoformat(),
                duration_ms=duration,
                details={"request_id": request_id},
                error=str(e),
            )
            self.test_results.append(result)
            return result

    async def test_response_transmission(
        self, request_id: str
    ) -> CommunicationTestResult:
        """Test successful transmission of compliance response"""
        start_time = datetime.utcnow()

        try:
            response_found = any(
                msg["message_id"] == request_id
                and msg["message_type"] == "ComplianceResponse"
                for msg in self.message_log
            )

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = CommunicationTestResult(
                test_name="Response Transmission",
                status="PASSED" if response_found else "FAILED",
                timestamp=start_time.isoformat(),
                duration_ms=duration,
                details={
                    "request_id": request_id,
                    "response_found": response_found,
                },
            )

            self.test_results.append(result)
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            result = CommunicationTestResult(
                test_name="Response Transmission",
                status="FAILED",
                timestamp=start_time.isoformat(),
                duration_ms=duration,
                details={"request_id": request_id},
                error=str(e),
            )
            self.test_results.append(result)
            return result

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a summary report of all tests"""
        passed = sum(1 for r in self.test_results if r.status == "PASSED")
        failed = sum(1 for r in self.test_results if r.status == "FAILED")
        warnings = sum(1 for r in self.test_results if r.status == "WARNING")

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_tests": len(self.test_results),
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "success_rate": (
                    (passed / len(self.test_results) * 100) if self.test_results else 0
                ),
            },
            "test_results": [asdict(r) for r in self.test_results],
            "message_log_entries": len(self.message_log),
        }

        return report

    def export_report_to_json(self, filepath: str) -> None:
        """Export test report to JSON file"""
        report = self.generate_test_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)


# Convenience functions for direct use


def verify_compliance_request_format(
    request_data: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Verify compliance request format.

    Args:
        request_data: The request data to verify

    Returns:
        Tuple of (valid, errors)
    """
    errors = []
    required_fields = ["request_id", "supplier_name", "industry", "company_values"]

    for field in required_fields:
        if field not in request_data:
            errors.append(f"Missing required field: {field}")

    # Type validation
    if not isinstance(request_data.get("request_id"), str):
        errors.append("request_id must be a string")
    if not isinstance(request_data.get("supplier_name"), str):
        errors.append("supplier_name must be a string")
    if not isinstance(request_data.get("industry"), str):
        errors.append("industry must be a string")
    if not isinstance(request_data.get("company_values"), str):
        errors.append("company_values must be a string")

    return len(errors) == 0, errors


def verify_compliance_response_format(
    response_data: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Verify compliance response format.

    Args:
        response_data: The response data to verify

    Returns:
        Tuple of (valid, errors)
    """
    errors = []
    required_fields = ["request_id", "supplier_name", "compliance_score", "violations"]

    for field in required_fields:
        if field not in response_data:
            errors.append(f"Missing required field: {field}")

    # Type validation
    if not isinstance(response_data.get("request_id"), str):
        errors.append("request_id must be a string")
    if not isinstance(response_data.get("supplier_name"), str):
        errors.append("supplier_name must be a string")
    if not isinstance(response_data.get("compliance_score"), (int, float)):
        errors.append("compliance_score must be a number")
    if not isinstance(response_data.get("violations"), list):
        errors.append("violations must be a list")

    # Validate compliance_score is between 0-100
    score = response_data.get("compliance_score")
    if isinstance(score, (int, float)) and not (0 <= score <= 100):
        errors.append("compliance_score must be between 0 and 100")

    return len(errors) == 0, errors
