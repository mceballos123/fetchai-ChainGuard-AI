def compliance_prompt(company_values, industry, supplier_list):
    return f"""
            The data that is presented to you is scraped from the B corp page website do not use any other data besides what's shown there
            and do not make up any data. The goal is to get a good assemnet based on what provided to you. If there is not enough data make 
            what you can from the data available to you. Just because there isn't enough data doesn't mean you can't make a good assessment.
            using the scraped data from the B Corp page:
            When evulating be fair with the information that is provided to you. Do not make up any data from the page that is not provided to you.
            Additonaly yes I know that the data is not perfect and doesn't contain alot of information but make the best assessment you can with the data available to you. At the same time don't be too hard on the supplier since we have little information to work with. Keep this in mind this page is preapporved supplier but at the same time still evulaute the compliance of the supplier

            YOU TASK IS TO:
            SELECT THE BEST SUPPLIER AND ANALYZE COMPLIANCE - Single Query

            Your company Values: {company_values}
            Industry: {industry}

            Select from the available suppliers:
            {supplier_list}

            Identify the single best supplier matching your company values
            the analyze that supplier's compliance

            Score the selected suppliers compliance from 0-100: 
            - 75+ strong verified evidence
            - 60-74 decent practices
            - 40-59 basic
            - <40 poor

            Provide:
            1. Selected supplier name (exact match from list)
            2. Ethics score (0-100)
            3. Sustainability score (0-100)
            4. Combined score (average)
            5. Ethics summary (2-3 sentences)
            6. Sustainability summary (2-3 sentences)
            7. Violations

            Response format:
            SELECTED_SUPPLIER: [name]
            ETHICS_SCORE: [number]
            SUSTAINABILITY_SCORE: [number]
            COMBINED_SCORE: [number]
            ETHICS_INFO: [summary]
            SUSTAINABILITY_INFO: [summary]
            VIOLATIONS: [list], if supplier has little violations still include it in the list"""