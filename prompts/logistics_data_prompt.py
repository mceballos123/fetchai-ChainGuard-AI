def analyze_logistics_data_prompt(logistics_prompt: str, supplier_name: str, product_category: str, estimated_inventory:str,data_summary: str):
    return f"""
{logistics_prompt}

---

SUPPLIER: {supplier_name}
PRODUCT CATEGORY: {product_category.upper()}
ESTIMATED INVENTORY UNITS: {estimated_inventory:,}

{data_summary}

---

Based on the above data and your role as a LOGISTICS MONITORING AGENT, provide your analysis in the following EXACT format:

CURRENT_INVENTORY_LEVEL: [LOW or MEDIUM or HIGH or CRITICAL]
INVENTORY_DETAILS: [Your analysis based on lead times and product availability]

PREDICTED_INVENTORY_LEVEL: [LOW or MEDIUM or HIGH or STABLE]
INVENTORY_FORECAST: [Your forecast based on shipping patterns and lead times]

RESTOCKING_STATUS: [ON_TIME or MONITOR or DELAYED]
RESTOCKING_DETAILS: [Your analysis of supplier lead times and shipping schedules]

OVERALL_LOGISTICS_STATUS: [HEALTHY or WARNING or CRITICAL]

ALERTS: [List any alerts separated by semicolons, or "None" if no alerts]

REASONING: [Brief explanation of your analysis]
"""