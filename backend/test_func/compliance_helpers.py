"""
Helper functions for Compliance Agent communication and verification.

This module provides utilities for the compliance agent to verify connection,
log requests/responses, and validate responses without cluttering the main agent file.
"""

from typing import Dict, Any, Tuple
from datetime import datetime
from uagents import Context

from backend.models.compliance import ComplianceResponse
from backend.models.financial import FinancialResponse
from .test_agent_communication import verify_compliance_response_format


def verify_compliance_connection(ctx: Context) -> Dict[str, Any]:
    """
    Verify that the compliance agent connection is established.

    Returns:
        Dictionary with connection status and details
    """
    try:
        # Check if storage is initialized
        supplier_history = ctx.storage.get("supplier_history")
        request_trace = ctx.storage.get("request_trace")

        # Note: rag_system is checked in the actual agent
        is_connected = supplier_history is not None and request_trace is not None

        return {
            "status": "CONNECTED" if is_connected else "INITIALIZING",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_address": ctx.agent.address,
            "storage_initialized": supplier_history is not None,
            "trace_initialized": request_trace is not None,
            "message": (
                "Compliance Agent ready to process requests"
                if is_connected
                else "Compliance Agent initializing"
            ),
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "message": "Failed to verify compliance connection",
        }


def log_request_reception(
    ctx: Context,
    request_id: str,
    supplier_name: str,
    sender: str,
) -> None:
    """
    Log incoming request from orchestrator.

    Args:
        ctx: Context object
        request_id: Unique request identifier
        supplier_name: Name of supplier being checked
        sender: Address of sender (orchestrator)
    """
    try:
        request_trace = ctx.storage.get("request_trace") or {
            "received_requests": [],
            "sent_responses": [],
        }

        trace_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "supplier_name": supplier_name,
            "sender": sender,
            "status": "RECEIVED",
        }

        request_trace["received_requests"].append(trace_entry)
        ctx.storage.set("request_trace", request_trace)
        ctx.logger.info(f"✓ Request logged: {request_id} from {sender}")
    except Exception as e:
        ctx.logger.error(f"Error logging request reception: {e}")


def log_response_transmission(
    ctx: Context,
    request_id: str,
    supplier_name: str,
    compliance_score: float,
    sender: str,
) -> None:
    """
    Log outgoing response to orchestrator.

    Args:
        ctx: Context object
        request_id: Unique request identifier
        supplier_name: Name of supplier
        compliance_score: Compliance score calculated
        sender: Address of recipient (orchestrator)
    """
    try:
        request_trace = ctx.storage.get("request_trace") or {
            "received_requests": [],
            "sent_responses": [],
        }

        trace_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "supplier_name": supplier_name,
            "compliance_score": compliance_score,
            "recipient": sender,
            "status": "SENT",
        }

        request_trace["sent_responses"].append(trace_entry)
        ctx.storage.set("request_trace", request_trace)
        ctx.logger.info(f"✓ Response logged: {request_id} to {sender}")
    except Exception as e:
        ctx.logger.error(f"Error logging response transmission: {e}")


def validate_response_before_sending(
    ctx: Context, response: ComplianceResponse
) -> Tuple[bool, str]:
    """
    Validate a compliance response before sending to orchestrator.

    Args:
        ctx: Context object
        response: ComplianceResponse to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        response_data = {
            "request_id": response.request_id,
            "supplier_name": response.supplier_name,
            "compliance_score": response.compliance_score,
            "violations": response.violations,
        }

        is_valid, errors = verify_compliance_response_format(response_data)

        if not is_valid:
            error_msg = ", ".join(errors)
            ctx.logger.warning(f"Response validation failed: {error_msg}")
            return False, error_msg

        ctx.logger.info("Response validation passed")
        return True, ""
    except Exception as e:
        ctx.logger.error(f"Error validating response: {e}")
        return False, str(e)


def validate_financial_response_before_sending(
    ctx: Context, response: FinancialResponse
) -> Tuple[bool, str]:
    """
    Validate a financial response before sending to orchestrator.
    This is separate from compliance responses - financial responses
    do NOT include compliance scores.

    Args:
        ctx: Context object
        response: FinancialResponse to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check that required fields are present
        if not hasattr(response, "request_id") or not response.request_id:
            return False, "Missing or invalid request_id"

        if not hasattr(response, "supplier_name") or not response.supplier_name:
            return False, "Missing or invalid supplier_name"

        if not hasattr(response, "financial_score") or response.financial_score is None:
            return False, "Missing or invalid financial_score"

        # Validate financial score is in valid range
        if not (0 <= response.financial_score <= 100):
            return False, f"Financial score out of range: {response.financial_score}"

        ctx.logger.info("Response validation passed")
        return True, ""
    except Exception as e:
        ctx.logger.error(f"Error validating financial response: {e}")
        return False, str(e)
