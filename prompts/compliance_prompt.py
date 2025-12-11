def compliance_prompt(company_values, industry, supplier_data):
    return f"""Analyze the following supplier's compliance and sustainability practices based on their B Corporation certification data.

COMPANY CONTEXT:
- Values: {company_values}
- Industry: {industry}

SUPPLIER DATA:
{supplier_data}

SCORING GUIDELINES:
- 75-100: Strong evidence of excellent practices
- 60-74: Good practices with room for improvement
- 40-59: Basic compliance, meets minimum standards
- 0-39: Limited evidence or concerns identified

REQUIRED OUTPUT:
1. ETHICS_SCORE: Rate 0-100 based on governance, workers, and customer practices
2. SUSTAINABILITY_SCORE: Rate 0-100 based on environmental and community impact
3. COMBINED_SCORE: Average of ethics and sustainability scores
4. ETHICS_INFO: Brief 2-3 sentence summary of ethical practices
5. SUSTAINABILITY_INFO: Brief 2-3 sentence summary of sustainability efforts
6. VIOLATIONS: List any concerns or areas needing improvement (empty list if none found)

Base your assessment only on the provided data. If information is limited, provide a fair evaluation based on available evidence."""