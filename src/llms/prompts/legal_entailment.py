PROMPT_DICT = {
    "lte.pr.ma.1": {
        "system": """You are an intelligent legal assistant that can identify the paragraph(s) from a historical case law that entails or supports the decision of a new legal case.
        """,
        "input": """Your task is to identify which paragraph(s) from the provided historical case law texts most closely supports the judgment in a current case. Please read the paragraphs from previous case laws and the decision from the ongoing case provided below. Start by stating which paragraph(s) you consider most applicable to the current case's decision. Then, follow up with your reasoning as to why these paragraph(s) are pertinent.\nHere are the provided paragraphs from previous case laws:\n{docs}\n\nDecision of the ongoing case law: {query}\n\nPlease begin your response by identifying the relevant paragraph ID(s), and then elaborate on how they relate to the decision in the current case.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.2": {
        "system": """As an advanced legal assistant with a keen understanding of case law, your task is to determine which paragraphs from historical legal cases most strongly support or entail the decision in a current legal case.
        """,
        "input": """Your task is to recommend which historical case law paragraph(s) should be referenced in their argument, based on a recent case decision.\nBelow are the historical case law paragraphs followed by the decision made in the recent case:\n{docs}\n\nLatest case decision: {query}\n\nLead with your recommendation of the relevant paragraph ID(s). Then discuss why these paragraph(s) are applicable, in light of the recent case decision.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.3": {
        "system": """You are an intelligent legal assistant tasked with identifying relevant precedents from historical case law to support the decision in a current legal case.
        """,
        "input": """You are to present a legal opinion on which historical case law paragraphs serves as precedent for a decision in a new case.\nProvided are the excerpts and the decision from the new case:\n{docs}\n\nDecision in the new case: {query}\n\nCommence your legal opinion by selecting the appropriate paragraph ID(s). Proceed to expound on the relevance or irrelevance of the selected paragraph(s) to the decision in question.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.4": {
        "system": """You are an intelligent legal assistant tasked with identifying relevant precedents from historical case law to support the decision in a current legal case.
        """,
        "input": """Your objective is to determine which paragraphs from the provided historical case law texts best align with the judgment in the ongoing case. Review the paragraphs from past cases and the decision from the current case provided below. Begin by identifying the relevant paragraph(s) and then provide a detailed explanation of their relevance to the decision in the current case. Here are the paragraphs from previous case laws:\n{docs}\n\nDecision of the ongoing case law: {query}\n\nStart your response by indicating the relevant paragraph ID(s), followed by a comprehensive explanation of their significance to the current case's decision.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.5": {
        "system": """You are an adept legal assistant entrusted with the task of matching relevant paragraphs from historical legal cases to the decision of a current legal case.
        """,
        "input": """Your objective is to pinpoint which paragraphs from the provided historical case law texts best substantiate the decision in the ongoing case. Analyze the paragraphs from previous cases and the decision from the current case provided below. Start by identifying the relevant paragraph(s) and then elucidate their significance to the decision in the current case. Evaluate the paragraphs from previous case laws and the decision of the ongoing case: \n{docs}\n\nDecision of the ongoing case law: {query}\n\nStart your response by identifying the relevant paragraph ID(s), followed by an in-depth explanation of their relevance to the current case's decision.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.6": {
        "system": """In your capacity as a proficient legal assistant, your task is to extract pertinent paragraphs from historical case law that support the decision in a current legal case.
        """,
        "input": """Your goal is to determine which paragraphs from the provided historical case law texts most strongly support the judgment in the ongoing case. Assess the paragraphs from previous cases and the decision from the current case provided below. Commence by identifying the relevant paragraph(s) and then provide a comprehensive explanation of their significance to the decision in the current case. Evaluate the paragraphs from previous case laws and the decision of the ongoing case:\n{docs}\n\nDecision of the ongoing case law: {query}\n\nBegin your response by identifying the relevant paragraph ID(s), followed by a thorough explanation of their applicability to the current case's decision.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.7": {
        "input": """Given the provided paragraphs from prior case law and the ruling from the ongoing case, identify the paragraph ID(s) that offer the most significant legal precedent.\nThe paragraphs are as follows:\n{docs}\n\nCurrent case ruling: {query}\n\nBegin by highlighting which paragraph ID(s) you find most influential in reaching the conclusion for the case at hand. Afterwards, explain the connection between these paragraph(s) and the ruling, detailing their legal significance.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.8": {
        "input": """Examine the paragraphs from prior case law below and articulate their relevance to the current case decision.\nHere are the historical legal paragraphs:\n{docs}\n\nDecision of the current case: {query}\n\nIndicate which case paragraph ID(s) you believe are most critical to the decision of the current case and provide justification for why these parts matter.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.9": {
        "input": """Appraise the relevance of each historical paragraph provided to the resolution of a current legal dispute.\nDetailed case law paragraphs:\n{docs}\n\nResolution of the current dispute: {query}\n\nIndicate which past case paragraph ID(s) appear directly applicable to the dispute's resolution and dissect the bearing they have on the outcome.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.10": {
        "input": """Inspect the paragraphs from antecedent cases alongside the ruling at hand. Do any paragraphs inform the ruling, or are they unrelated?\nParagraphs from previous cases:\n{docs}\n\nRuling at hand: {query}\n\nIdentify influencing paragraph ID(s), then delve into the reasons supporting your assessment.""",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.11": {
        "input": "Sift through the historical case law paragraphs and the recent case decision to establish notable legal connections.\nProvided paragraphs:\n{docs}\n\nRecent case decision: {query}\n\nSelect the legal paragraph ID(s) that bear the greatest correlation with the recent decision and rationalize the impact these paragraph(s) had on the adjudication.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.12": {
        "input": "### Decision of New Case (Q):\n{query}\n\n### Relevant Case (R) Paragraphs:\n{docs}\n\n### Task: Identify the paragraph ID(s) from the relevant case (R) that provides a legal argumentation entailing the decision of the new case (Q). Discuss the legal arguments and reasoning involved.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.13": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: From the provided paragraphs in the relevant case, determine which paragraph ID(s) entails the decision of the new case. Explain your reasoning with reference to legal principles, facts, or logic used in both cases.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.14": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: From the provided paragraphs in the relevant case, determine which paragraph ID(s) entails the decision of the new case. Explain your reasoning with reference to legal principles, facts, or logic used in both cases.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.pr.ma.15": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: Analyze each paragraph from the relevant case and determine which paragraph ID(s) entails the decision of the new case. Support your answer with logical and legal analysis.",
        "response": "\n\nThe paragraph ID(s) is:"
    },

    "lte.p.ma.1": {
        "input": "Question:\nDetermine which legal paragraph(s) among the provided options logically entail or support the given legal statement. \nParagraphs:\n{docs}\n\nStatement: {query}\n\n Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.2": {
        "input": "Question:\nGiven a legal statement, your task is to identify the specific legal paragraph(s) from the provided list of paragraphs that substantiate or logically imply the given statement. \nParagraphs:\n{docs}\n\nStatement: {query}\n\nOnly respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.3": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nAscertain the legal paragraph(s) within the given set of paragraphs that best corresponds to, supports, or logically entails the presented legal statement. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.4": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nEvaluate the list of legal paragraphs and determine which one(s) can be considered as entailing or providing valid support for the given legal statement. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.5": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nExamine the legal statement and assess the provided legal paragraphs to identify the paragraph(s) that most strongly entail or support the given statement. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.6": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nYour goal is to find the legal paragraph(s) within the given options that, when considered, logically lead to or support the presented legal statement. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.7": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nGiven a legal statement, sift through the provided legal paragraphs to pinpoint the paragraph(s) that best encapsulate or logically entail the essence of the given statement. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.8": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nAnalyzing the legal context, identify the paragraph(s) from the list that can be considered as entailing or providing legal basis for the presented statement. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.9": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: From the provided paragraphs in the relevant case, determine which paragraph(s) entails the decision of the new case. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.10": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: From the provided paragraphs in the relevant case, determine which paragraph(s) entails the decision of the new case. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.11": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: Analyze each paragraph from the relevant case and determine which paragraph(s) entails the decision of the new case. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.12": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: Identify the paragraph(s) from the relevant case that provides a legal argumentation entailing the decision of the new case. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.13": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: Compare each paragraph in the relevant case to the decision of the new case and identify the paragraph(s) that entails the decision. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.14": {
        "input": "### Decision of New Case (Q):\n{query}\n\n### Relevant Case (R) Paragraphs:\n{docs}\n\n### Task: Extract the key concepts from the decision of the new case (Q) and identify which paragraph from the relevant case (R) contains these concepts, thereby entailing the decision. Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },
    "lte.p.ma.15": {
        "input": "### Decision of New Case (Q):\n{query}\n\n### Relevant Case (R) Paragraphs:\n{docs}\n\n### Task: Analyze the precedents set in the paragraphs of the relevant case (R) and identify which one entails the decision of the new case (Q). Only respond with the paragraph ID(s), do not say any word or explain.",
        "response": "\n\nThe paragraph ID(s) is:"
    },

    "lte.pr.sa.1": {
        "system": """You are an intelligent legal assistant that can identify the paragraph from a historical case law that entails or supports the decision of a new legal case.
        """,
        "input": """Your task is to identify which paragraph from the provided historical case law texts most closely supports the judgment in a current case. Please read the paragraphs from previous case laws and the decision from the ongoing case provided below. Determine strictly one paragraph that is most relevant to the present case's decision. Start by stating the paragraph you consider most applicable to the current case's decision. Then, follow up with your reasoning as to why that paragraph are pertinent.\nHere are the provided paragraphs from previous case laws:\n{docs}\n\nDecision of the ongoing case law: {query}\n\nPlease begin your response by identifying the relevant paragraph ID and then elaborate on how that paragraph relate to the decision in the current case.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.2": {
        "system": """As an advanced legal assistant with a keen understanding of case law, your task is to determine which paragraph from historical legal cases most strongly supports or entails the decision in a current legal case.
        """,
        "input": """Your task is to recommend one historical case law paragraph that should be referenced in their argument, based on a recent case decision.\nBelow are the historical case law paragraphs followed by the decision made in the recent case:\n{docs}\n\nLatest case decision: {query}\n\nLead with your recommendation of strictly one relevant paragraph ID. Then discuss why that paragraph are applicable, or why none of them fit, in light of the recent case decision.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.3": {
        "system": """You are an intelligent legal assistant tasked with identifying relevant precedent from historical case law to support the decision in a current legal case.
        """,
        "input": """You are to present a legal opinion on one historical case law paragraph that serves as a precedent for a decision in a new case.\nProvided are the excerpts and the decision from the new case:\n{docs}\n\nDecision in the new case: {query}\n\nCommence your legal opinion by selecting strictly one appropriate paragraph ID. Proceed to expound on the relevance or irrelevance of the selected paragraph to the decision in question.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.4": {
        "system": """You are an intelligent legal assistant tasked with identifying relevant precedent from historical case law to support the decision in a current legal case.
        """,
        "input": """Your objective is to determine one paragraph from the provided historical case law texts best align with the judgment in the ongoing case. Review the paragraphs from past cases and the decision from the current case provided below. Begin by identifying strictly one relevant paragraph and then provide a detailed explanation of its relevance to the decision in the current case. Here are the paragraphs from previous case laws:\n{docs}\n\nDecision of the ongoing case law: {query}\n\nStart your response by indicating the relevant paragraph ID, followed by a comprehensive explanation of its significance to the current case's decision.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.5": {
        "system": """You are an adept legal assistant entrusted with the task of matching relevant paragraph from historical legal cases to the decision of a current legal case.
        """,
        "input": """Your objective is to pinpoint one paragraph from the provided historical case law texts best substantiate the decision in the ongoing case. Analyze the paragraphs from previous cases and the decision from the current case provided below. Start by identifying strictly one relevant paragraph and then elucidate its significance to the decision in the current case. Evaluate the paragraphs from previous case laws and the decision of the ongoing case: \n{docs}\n\nDecision of the ongoing case law: {query}\n\nStart your response by identifying the relevant paragraph ID, followed by an in-depth explanation of its relevance to the current case's decision.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.6": {
        "system": """In your capacity as a proficient legal assistant, your task is to extract pertinent paragraph from historical case law that support the decision in a current legal case.
        """,
        "input": """Your goal is to determine one paragraphs from the provided historical case law texts most strongly support the judgment in the ongoing case. Assess the paragraphs from previous cases and the decision from the current case provided below. Commence by identifying strictly one relevant paragraph and then provide a comprehensive explanation of its significance to the decision in the current case. Evaluate the paragraphs from previous case laws and the decision of the ongoing case:\n{docs}\n\nDecision of the ongoing case law: {query}\n\nBegin your response by identifying the relevant paragraph ID, followed by a thorough explanation of its applicability to the current case's decision.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.7": {
        "input": """Given the provided paragraphs from prior case law and the ruling from the ongoing case, identify one paragraph ID that offer the most significant legal precedent.\nThe paragraphs are as follows:\n{docs}\n\nCurrent case ruling: {query}\n\nBegin by highlighting strictly one paragraph you find most influential in reaching the conclusion for the case at hand. Afterwards, explain the connection between that paragraph and the ruling, detailing its legal significance.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.8": {
        "input": """Examine the paragraphs from prior case law below and articulate their relevance to the current case decision.\nHere are the historical legal paragraphs:\n{docs}\n\nDecision of the current case: {query}\n\nIndicate strictly one paragraph ID you believe are most critical to the decision of the current case and provide justification for why that paragraph matter.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.9": {
        "input": """Appraise the relevance of each historical paragraph provided to the resolution of a current legal dispute.\nDetailed case law paragraphs:\n{docs}\n\nResolution of the current dispute: {query}\n\nIndicate strictly one past case paragraph ID appear directly applicable to the dispute's resolution and dissect the bearing it have on the outcome.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.10": {
        "input": """Inspect the paragraphs from antecedent cases alongside the ruling at hand. Do any paragraphs inform the ruling, or are they unrelated?\nParagraphs from previous cases:\n{docs}\n\nRuling at hand: {query}\n\nIdentify strictly one influencing paragraph ID, then delve into the reasons supporting your assessment.""",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.11": {
        "input": "Sift through the historical case law paragraphs and the recent case decision to establish notable legal connections.\nProvided paragraphs:\n{docs}\n\nRecent case decision: {query}\n\nSelect strictly one legal paragraph ID that bear the greatest correlation with the recent decision and rationalize the impact that paragraph had on the adjudication.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.12": {
        "input": "### Decision of New Case (Q):\n{query}\n\n### Relevant Case (R) Paragraphs:\n{docs}\n\n### Task: Identify strictly one paragraph ID from the relevant case (R) that provides a legal argumentation entailing the decision of the new case (Q). Discuss the legal arguments and reasoning involved.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.13": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: From the provided paragraphs in the relevant case, determine stricly one paragraph that entails the decision of the new case. Explain your reasoning with reference to legal principles, facts, or logic used in both cases.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.14": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: From the provided paragraphs in the relevant case, determine stricly one paragraph that the decision of the new case. Explain your reasoning with reference to legal principles, facts, or logic used in both cases.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.pr.sa.15": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: Analyze each paragraph from the relevant case and determine stricly one paragraph that entails the decision of the new case. Support your answer with logical and legal analysis.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.1": {
        "input": "Question:\nDetermine strictly one legal paragraph among the provided options logically entail or support the given legal statement. \nParagraphs:\n{docs}\n\nStatement: {query}\n\n Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.2": {
        "input": "Question:\nGiven a legal statement, your task is to identify strictly one specific legal paragraph from the provided list of paragraphs that substantiates or logically implies the given statement. \nParagraphs:\n{docs}\n\nStatement: {query}\n\nOnly respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.3": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nAscertain strictly one legal paragraph within the given set of paragraphs that best corresponds to, supports, or logically entails the presented legal statement. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.4": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nEvaluate the list of legal paragraphs and determine strictly one paragraph that can be considered as entailing or providing valid support for the given legal statement. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.5": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nExamine the legal statement and assess the provided legal paragraphs to identify strictly one paragraph that most strongly entails or supports the given statement. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.6": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nYour goal is to find strictly one legal paragraph within the given options that, when considered, logically leads to or supports the presented legal statement. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.7": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nGiven a legal statement, sift through the provided legal paragraphs to pinpoint strictly one paragraph that best encapsulates or logically entails the essence of the given statement. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.8": {
        "input": "Paragraphs:\n{docs}\n\nStatement:\n{query}\n\nQuestion:\nAnalyzing the legal context, identify strictly one paragraph from the list that can be considered as entailing or providing legal basis for the presented statement. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.9": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: From the provided paragraphs in the relevant case, determine strictly one paragraph that entails the decision of the new case. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.10": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: From the provided paragraphs in the relevant case, determine strictly one paragraph that entails the decision of the new case. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.11": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: Analyze each paragraph from the relevant case and determine strictly one paragraph that entails the decision of the new case. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.12": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: Identify strictly one paragraph that from the relevant case that provides a legal argumentation entailing the decision of the new case. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.13": {
        "input": "### Decision of New Case :\n{query}\n\n### Relevant Case Paragraphs:\n{docs}\n\n### Task: Compare each paragraph in the relevant case to the decision of the new case and identify strictly one paragraph that entails the decision. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.14": {
        "input": "### Decision of New Case (Q):\n{query}\n\n### Relevant Case (R) Paragraphs:\n{docs}\n\n### Task: Extract the key concepts from the decision of the new case (Q) and identify which paragraph from the relevant case (R) contains these concepts, thereby entailing the decision. Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    },
    "lte.p.sa.15": {
        "input": "### Decision of New Case (Q):\n{query}\n\n### Relevant Case (R) Paragraphs:\n{docs}\n\n### Task: Analyze the precedents set in the paragraphs of the relevant case (R) and identify which one entails the decision of the new case (Q). Only respond with the paragraph ID, do not say any word or explain.",
        "response": "\n\nThe paragraph ID is:"
    }
}
