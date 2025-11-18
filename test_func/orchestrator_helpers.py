"""
Helper functions for Orchestrator Agent communication and verification.

This module provides utilities for the orchestrator agent to verify connection,
log messages, and validate requests without cluttering the main agent file.
"""

from typing import Dict, Any, Tuple
from datetime import datetime
from uagents import Context

from models.compliance import ComplianceRequest
from .test_agent_communication import verify_compliance_request_format


def verify_orchestrator_connection(ctx: Context) -> Dict[str, Any]:
    """
    Verify that the orchestrator agent connection is established.

    Returns:
        Dictionary with connection status and details
    """
    try:
        # Check if storage is initialized
        active_sessions = ctx.storage.get("active_sessions")
        message_trace = ctx.storage.get("message_trace")

        is_connected = active_sessions is not None and message_trace is not None

        return {
            "status": "CONNECTED" if is_connected else "DISCONNECTED",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_address": ctx.agent.address,
            "storage_initialized": active_sessions is not None,
            "trace_initialized": message_trace is not None,
            "message": (
                "Orchestrator ready to forward requests to Compliance Agent"
                if is_connected
                else "Connection verification failed"
            ),
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "message": "Failed to verify orchestrator connection",
        }


def log_message_transmission(
    ctx: Context,
    direction: str,
    message_type: str,
    message_id: str,
    data: Dict[str, Any],
) -> None:
    """
    Log message transmission between agents.

    Args:
        ctx: Context object
        direction: "SENT" or "RECEIVED"
        message_type: Type of message
        message_id: Unique message identifier
        data: Message data
    """
    try:
        message_trace = ctx.storage.get("message_trace") or {
            "received_from_asi": [],
            "sent_to_compliance": [],
            "received_from_compliance": [],
            "sent_to_asi": [],
        }

        trace_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": message_type,
            "message_id": message_id,
            "data": data,
        }

        if direction == "SENT" and message_type == "ComplianceRequest":
            message_trace["sent_to_compliance"].append(trace_entry)
        elif direction == "RECEIVED" and message_type == "ComplianceResponse":
            message_trace["received_from_compliance"].append(trace_entry)
        elif direction == "RECEIVED" and message_type == "ChatMessage":
            message_trace["received_from_asi"].append(trace_entry)
        elif direction == "SENT" and message_type == "ChatMessage":
            message_trace["sent_to_asi"].append(trace_entry)

        ctx.storage.set("message_trace", message_trace)
        ctx.logger.info(f"✓ Message logged: {direction} {message_type}")
    except Exception as e:
        ctx.logger.error(f"Error logging message transmission: {e}")


def validate_request_before_sending(
    ctx: Context, request: ComplianceRequest
) -> Tuple[bool, str]:
    """
    Validate a compliance request before sending to compliance agent.

    Args:
        ctx: Context object
        request: ComplianceRequest to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        request_data = {
            "request_id": request.request_id,
            "supplier_name": request.supplier_name,
            "industry": request.industry,
            "company_values": request.company_values,
        }

        is_valid, errors = verify_compliance_request_format(request_data)

        if not is_valid:
            error_msg = ", ".join(errors)
            ctx.logger.warning(f"Request validation failed: {error_msg}")
            return False, error_msg

        ctx.logger.info("✓ Request validation passed")
        return True, ""
    except Exception as e:
        ctx.logger.error(f"Error validating request: {e}")
        return False, str(e)
