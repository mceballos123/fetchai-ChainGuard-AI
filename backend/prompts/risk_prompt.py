def risk_prompt(supplier_name, industry):
    return """
            YOU TASK IS TO:
            Analyze the risk management of the supplier {supplier_name} in the industry {industry} and score it from 0-100:
            
            Score it from 0-100:
            - 70-100: Excellent risk management, minimal vulnerabilities
            - 60-69: Good risk management, acceptable with minor considerations (APPROVED)
            - 50-59: Moderate risk, requires careful evaluation (NEEDS REVIEW)
            - 40-49: Significant risks requiring mitigation plans
            - Below 40: High risk, major concerns present
            
            Evaluate these risk factors:
            - Capacity constraints
            - Natural disaster exposure (location vulnerabilities including weather event that could impact the supplier)
            - Logistics and accessibility (transportation, infrastructure)
            
            Provide a balanced assessment:
            1. Risk score (0-100)
            2. Risk summary (2-3 sentences): highlight both strengths and concerns
            3. Risk factors: list specific concerns, or "minimal risks identified" , if there is minimal risk
            include it in the list even if the supplier is approved.
            
            Response format:
            RISK_SCORE: [number]
            RISK_DETAILS: [balanced 2-3 sentence summary]
            RISK_FACTORS: [specific issues or "minimal risks identified even if the supplier is approved"]
    
    """