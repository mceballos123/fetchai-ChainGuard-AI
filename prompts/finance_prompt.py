def finance_prompt(supplier_name, industry, country):
    return f"""
            YOUR TASK IS TO:
            Analyze the financial risk for supplier {supplier_name} operating in {country} within the {industry} industry.
            
            The country was extracted from the B Corp Headquarters section (e.g., "Catalonia, Spain" -> "Spain").
            
            Score from 0-100 based on the US trade relationship with {country}. Evaluate tariffs and inflation fairly - 
            yes, some countries have strict tariffs with the United States, but be balanced. Don't subtract 40 points 
            just because there are some trade restrictions. Be fair when evaluating the financial score.
            
            SCORING GUIDANCE:
            - 75 to 100: minimal financial risks (low tariffs, stable inflation)
            - 60 to 74: moderate risk (some tariffs, manageable inflation)
            - 40 to 59: significant risk (high tariffs or inflation concerns)
            - Below 40: high risk (severe tariffs, economic instability)
            
            WHAT TO EVALUATE FOR {country}:
            - Current tariff rates in US-{country} trade relationship
            - Trade agreements (USMCA, FTA, etc.)
            - Inflation trends in {country}
            - Trade restrictions or exemptions
            - Currency and economic stability
            - Impact on supply chain costs
            
            PROVIDE:
            1. Score (0-100) - based on tariff severity and inflation impact
            2. Brief summary (2-3 sentences): describe specific tariffs, inflation concerns, and cost implications for {country}
            3. Risk factors list - specific tariff percentages, trade restrictions, or "minimal risks"
            
            RESPONSE FORMAT:
            FINANCIAL_SCORE: [number]
            FINANCIAL_DETAILS: [summary focusing on tariffs and inflation in {country}]
            RISK_FACTORS: [specific tariff rates, trade restrictions, or "minimal risks"]
            """
