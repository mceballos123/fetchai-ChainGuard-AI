def analyze_demand_data_prompt(demand_forecast_prompt: str, supplier_name: str, product_category: str, data_summary: str):
    return f"""
{demand_forecast_prompt}

---

SUPPLIER: {supplier_name}
PRODUCT CATEGORY: {product_category.upper()}

{data_summary}

---

Based on the above data and your role as a DEMAND FORECAST AGENT, provide your analysis in the following EXACT format:

DELAY_STATUS: [ON TIME or MONITOR or DELAYS DETECTED]
DELAY_DETAILS: [Your analysis of delays based on availability and sales patterns]

SHIPPING_STATUS: [NORMAL or MODERATE VOLUME or HIGH VOLUME]
SHIPPING_DETAILS: [Your analysis of shipping/logistics demand pressure]

QUALITY_STATUS: [STANDARD or GOOD or PREMIUM]
QUALITY_DETAILS: [Your analysis of quality based on price and product performance]

OVERALL_PERFORMANCE: [EXCELLENT or GOOD or FAIR or POOR]

ALERTS: [List any alerts separated by semicolons, or "None" if no alerts]

REASONING: [Brief explanation of your analysis]
""" 