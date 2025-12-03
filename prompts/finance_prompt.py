def finance_prompt(supplier_name, industry, country):
    return f"""
            YOUR TASK IS TO:
            Analyze the financial risk for supplier {supplier_name} operating in {country} within the {industry} industry.
            Score from 0-100 based on tariffs and inflation data from Trade War Tracker, do the best you can to evualte the score with the information that is provided to you, since we're based in american when your're avualting the score of companies in america evualte them based on inflations and not tariffs:
            
            - 75 to 100: minimal financial risks (low tariffs, stable inflation)
            - 60 to 74: moderate risk (some tariffs, manageable inflation)
            - 40 to 59: significant risk (high tariffs or inflation concerns)
            - Below 40: high risk (severe tariffs, economic instability)
            
            What to evaluate based on the Trade War Tracker data:
            - Current tariff rates affecting {country}
            - Trade war timeline events involving {country}
            - Tariff increases or decreases
            - Inflation trends in {country}
            - Trade restrictions or exemptions
            - Currency and economic stability
            - Impact on supply chain costs
            
            Provide:
            1. Score (0-100) - based on tariff severity and inflation impact
            2. Brief summary (2-3 sentences): describe specific tariffs, inflation concerns, and cost implications for {country}
            3. Risk factors list - specific tariff percentages, trade restrictions, or "minimal risks"
            
            Response format:
            FINANCIAL_SCORE: [number]
            FINANCIAL_DETAILS: [summary focusing on tariffs and inflation in {country}]
            RISK_FACTORS: [specific tariff rates, trade restrictions, or "minimal risks"]
            """