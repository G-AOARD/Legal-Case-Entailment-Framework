PROMPT_DICT = {
    "monoT5": {
        "instruction": "",
        "input": "{input}",
        "response": " Relevancy:"
    }
}

INSTRUCTION_DICT = {
    "monoT5": {
        "instruction": "",
        "input": "Query: {query} doc: {doc}",
        "answers": ["false", "true"]
    }
}