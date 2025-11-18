"""
State schemas for LangGraph workflows in the Find Supplier system.

This module defines TypedDict schemas that track state as it flows through
the workflow nodes (Find Supplier -> Compliance -> Financial -> Risk).
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class SupplierInputState(TypedDict):
    """Initial input from user via ASI:1"""

    user_input: str  # Full user prompt
    business_type: str  # e.g., "coffee shop"
    company_values: str  # e.g., "treat workers well"
    industry: str  # e.g., "food & beverage"
    product_needed: str  # e.g., "coffee beans"


class SupplierWorkflowState(TypedDict):
    """Main state that flows through the entire workflow"""

    # Request tracking
    request_id: str
    timestamp: str

    # User input information
    user_input: str
    business_type: str
    company_values: str
    industry: str
    product_needed: str

    # Supplier search results (from Pinecone vector DB)
    supplier_name: Optional[str]
    supplier_location: Optional[str]
    supplier_country: Optional[str]

    # RAG context from LlamaIndex + Pinecone (EPA, BBB documents)
    retrieved_documents: Optional[List[str]]
    rag_context: Optional[str]

    # Compliance agent results
    compliance_score: Optional[float]  # 0-100, combined ethics + sustainability
    ethics_info: Optional[str]
    sustainability_info: Optional[str]
    violations: Optional[List[str]]

    # Financial agent results - RAG from financial_files/
    financial_score: Optional[float]  # 0-100, higher = lower risk (better)
    financial_info: Optional[str]  # Summary: tariffs, inflation, economic risks

    # Risk management agent results - RAG from risk_management_files/
    risk_score: Optional[float]  # 0-100, higher = better (lower risk)
    risk_details: Optional[str]  # Summary: capacity, disasters, logistics
    risk_factors: Optional[List[str]]  # List of identified risk factors

    # Workflow control
    current_step: str  # Track which node we're in
    error_message: Optional[str]
    should_continue: bool  # For conditional routing

    # Message history for LangGraph
    messages: Annotated[List[BaseMessage], operator.add]


class SupplierOutputState(TypedDict):
    """Final output returned to user via ASI:1"""

    request_id: str
    supplier_name: str
    supplier_location: str
    supplier_country: str
    compliance_score: float
    financial_score: float
    overall_message: str  # Human-readable summary
    passed_compliance: bool  # True if compliance >= 75
    passed_financial: bool  # True if financial >= 70 (lower risk)


class CompliancePrivateState(TypedDict):
    """Private state used only in compliance node"""

    rag_query: str
    document_chunks: List[str]
    llm_prompt: str
