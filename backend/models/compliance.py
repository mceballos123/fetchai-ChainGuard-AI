from uagents import Model
from pydantic import Field
from typing import List


class ComplianceRequest(Model):
    request_id: str
    supplier_name: str
    industry: str
    company_values: str
    timestamp: str = ""


class ComplianceResponse(Model):
    request_id: str
    supplier_name: str
    compliance_score: float
    violations: List[str] = []
    sustainability_info: str = ""
    ethics_info: str = ""
    timestamp: str = ""


"""
    Prompt message for the compliance agent to check the compliance of the supplier:
    Documents --> RAG + Llama index  --> Compliance

    RAG + Llama index sorts the information from the documents and passes the important information to the compliance agent

    The compliance agent analyzes the information and passes the information like the compliance score and the violations found

    Compliance --> passes the information --> find supplier agent --> passes the info to the user(if it goes through)

"""
