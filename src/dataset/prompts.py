
SIMPLE_CONTINUATION="""Write a continuation of similar length to the passage provided.
Keep the tone and style consistent with the provided passage.

Passage: {}

Only output the continuation, do not include any other details.

Continuation:
"""

REPHRASE_PROMPT="""Rephrase the following passage: {}

Only output the rephrased-passage, do not include any other details.

Rephrased passage:
"""

REPHRASE_WITH_CONTEXT_PROMPT = """Passage Preceding: {}
Passage Proceeding: {}

Rephrase the following passage: {}

Only output the rephrased-passage, do not include any other details.

Rephrased passage:
"""

PROMPT_NAMES = ["rephrase", "rephrase_with_context", "continuation"]

def get_prompt(prompt_type: str) -> str:
    assert prompt_type in PROMPT_NAMES, f"Invalid prompt type: {prompt_type}"
    if prompt_type == "rephrase":
        return REPHRASE_PROMPT
    elif prompt_type == "rephrase_with_context":
        return REPHRASE_WITH_CONTEXT_PROMPT
    elif prompt_type == "continuation":
        return SIMPLE_CONTINUATION