def finance_prompt(supplier_name, industry):
    return """
            YOU TASK IS TO:
            Analyze the financial risk of the supplier {supplier_name} in the industry {industry} and score it from 0-100:
            - 75 to 100 minimal risks
            - 60 to 74 moderate risk
            - 40 to 59 significant risk
            - Anything below is 40 is high risk
            
            What to evaluate:
            - Tariffs/import costs
            - Inflation trends
            - Currency exchange risks
            - Economic stability
            - Trade restrictions
            - Cost predictability
            - Relationship with the United States
            
            Provide:
            1. Score (0-100)
            2. Brief summary (2-3 sentences): operating costs, tariffs, inflation, currency risks
            3. Risk factors list or "minimal risks", if supplier has little risks still include it in the list
            
            Response format:
            FINANCIAL_SCORE: [number]
            FINANCIAL_DETAILS: [summary]
            RISK_FACTORS: [list or "minimal risks"], if the supplier has little risks still include it in the list           
            """