from uagents import Model
from pydantic import Field
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC
from enum import Enum



class ClientMessage(Model):
    """Client's business requirements for finding a supplier"""
    client_message: str = Field(..., description="Client's message")

class SupplierSearchRequest(Model):
    """Main request model for supplier search"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(default="", description="Request timestamp")
    client_requirements: ClientMessage
    
    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


"""
    Prompt message for the client to search for a supplier:

    Client --> interacts --> ASI:1
    My name is [name], I'm a busines owner of [business name], Optional[business type], I'm looking for a supplier for my business[product or services needed], that matches the values that my business holds, find my a supplier that matches with witht he values that my business hold

    ASI:1 --> looks and findsa the find_supplier agent and calls the orchestrator agent to pass the request ot the find_supplier feature

"""