def finance_prompt(supplier_name: str, industry: str, supplier_country: str, user_country: str = "United States"):
    """
    Generate financial analysis prompt based on trade relationship between user's country and supplier's country.
    
    Args:
        supplier_name: Name of the supplier
        industry: Industry/sector of the supplier
        supplier_country: Country where supplier operates (extracted from B Corp Headquarters)
        user_country: Country where user/buyer is located (default: United States)
    
    Returns:
        Prompt string for financial risk analysis
    """
    # Check if same country (domestic trade)
    is_domestic = supplier_country.lower() == user_country.lower()
    
    if is_domestic:
        return f"""
            YOUR TASK IS TO:
            Analyze the financial risk for DOMESTIC supplier {supplier_name} operating in {supplier_country} within the {industry} industry.
            
            This is a DOMESTIC transaction - both buyer and supplier are in {user_country}.
            No import tariffs or international trade considerations apply.
            
            SCORING GUIDANCE (Domestic):
            - 85 to 100: excellent domestic supplier (stable, established)
            - 70 to 84: good domestic supplier (minor concerns)
            - 60 to 69: acceptable domestic supplier (some financial risks)
            - Below 60: concerning domestic supplier (financial instability)
            
            WHAT TO EVALUATE:
            - Economic stability within {user_country}
            - Industry health in {industry} sector
            - Domestic supply chain reliability
            - Local currency stability
            - Regional economic factors
            
            PROVIDE:
            1. Score (0-100) - based on domestic financial health
            2. Brief summary (2-3 sentences): describe domestic trade advantages and any concerns
            3. Risk factors list - specific domestic considerations or "minimal risks (domestic)"
            
            RESPONSE FORMAT:
            FINANCIAL_SCORE: [number]
            FINANCIAL_DETAILS: [summary for domestic trade in {user_country}]
            RISK_FACTORS: [domestic considerations or "minimal risks"]
            """
    else:
        return f"""
            YOUR TASK IS TO:
            Analyze the financial risk for supplier {supplier_name} operating in {supplier_country} within the {industry} industry.
            
            The BUYER is located in: {user_country}
            The SUPPLIER is located in: {supplier_country}
            
            Evaluate the trade relationship between {user_country} and {supplier_country}.
            
            SCORING GUIDANCE:
            - 75 to 100: minimal financial risks (free trade agreements, low tariffs, stable currencies)
            - 60 to 74: moderate risk (some tariffs, manageable trade barriers)
            - 40 to 59: significant risk (high tariffs, trade tensions)
            - Below 40: high risk (severe tariffs, sanctions, economic instability)
            
            WHAT TO EVALUATE FOR {user_country}-{supplier_country} TRADE:
            - Current tariff rates between {user_country} and {supplier_country}
            - Trade agreements (FTA, customs unions, trade blocs)
            - Currency exchange stability ({user_country} currency vs {supplier_country} currency)
            - Trade restrictions, sanctions, or exemptions
            - Inflation trends in {supplier_country}
            - Shipping/logistics costs between countries
            - Import/export regulations
            
            SPECIAL CONSIDERATIONS:
            - EU-EU trade: No tariffs within European Union
            - USMCA: US-Canada-Mexico free trade
            - Same trade bloc: Generally favorable
            - Sanctioned countries: High risk
            
            PROVIDE:
            1. Score (0-100) - based on {user_country}-{supplier_country} trade relationship
            2. Brief summary (2-3 sentences): describe tariffs, trade agreements, and cost implications
            3. Risk factors list - specific trade considerations or "minimal risks"
            
            RESPONSE FORMAT:
            FINANCIAL_SCORE: [number]
            FINANCIAL_DETAILS: [summary for {user_country}-{supplier_country} trade]
            RISK_FACTORS: [specific trade factors or "minimal risks"]
            """
