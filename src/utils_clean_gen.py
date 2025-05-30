from typing import Union

def split_on_string(string: str, generation: str, index_to_pick: int) -> str:
    assert index_to_pick in [0, 1]

    generation_copy = generation
    if string in generation:
        generation = generation.split(string)[index_to_pick]
        # if it is empty, try the other index, otherwise revert to the original
        if generation == "":
            generation = generation_copy
            generation = generation.split(string)[abs(index_to_pick-1)]
        if generation == "":
            generation = generation_copy

    return generation

def clean_segment_strings(generation: str, strings_to_remove_segment: list):
    generation_copy = generation
    generation = generation.split("\n")
    index = [i for i, gen in enumerate(generation) if any(string in gen.lower() for string in strings_to_remove_segment)]
    if len(index) > 0:
        index = index[0]
        generation = "\n".join(generation[:index])
    else:
        generation = "\n".join(generation)
    
    if generation == "":
        return generation_copy
    else:
        return generation

def clean_generation(generation: Union[str, list[str]], is_reddit: bool = False) -> str:
    if isinstance(generation, list):
        result = []
        for g in generation:
            try:
                result.append(clean_generation(g))
            except:
                result.append(None)
        return result

    # split on obvious phrases added in predictable locations:
    strings_to_remove_and_index = [
        ("[Note: I rephrased", 0),
        ("# Rephrased passage:", 0),
        ("# Passage Preceding", 0),
        ("Rephrase the following passage", 0),
        ("Rephrased passage:", 0),
        ("This rephrased passage condenses the original passage", 0),
        ("Please let me know if you have any other questions.", 0),
        ("The rephrased passage is a", 0),
        ("[1] refers to the original", 0),
        ("Sure, here is the rephrased passage:\n\n", 1),
        ("Sure, here is the continuation:\n\n", 0),
        ("To rephrase the given passage, we can say:", 1),
        ("Only output the continuation, do not include any other details.", 1),
        ("Only output the continuation, do not include any other details.", 1),
        ("\n\n ", 1),
        ("\n ", 1),
        ("\n", 1),
    ]

    for string, index in strings_to_remove_and_index:
        generation = split_on_string(string, generation, index)
    # split on newlines, and remove all segments that contain the following strings:
    strings_to_remove_segment = [
        "note:", 
        "please note that the rephrased passage", 
        "rephrased passage",
        "alternatively:",
    ]
    generation = clean_segment_strings(generation, strings_to_remove_segment)

    # remove things that don't end with punctuation, closed parenthesis, and other legal things...
    if not is_reddit:
        generation_copy = generation
        if generation[-1] not in [".", "?", "!", ")", "]", "$", '"']:
            generation = generation.rsplit(".", 1)[0] + "."
            if generation[-2] in "0123456789":
                generation = generation_copy

    return generation

