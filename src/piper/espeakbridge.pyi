def initialize(data_dir: str) -> None: ...
"""Initialize espeak-ng."""

def set_voice(voice: str) -> None: ...
"""Set the espeak-ng voice by name."""

def get_phonemes(text: str) -> list[tuple[str, str, bool, int]]:
    """
    Convert input text to a list of (phonemes, terminator, end_of_sentence, byte_cursor) tuples.

    Returns a list where each item is:
        phonemes: str - IPA phonemes for a clause
        terminator: str - punctuation mark indicating clause type (".", "?", "!", ",", ":", ";")
        end_of_sentence: bool - True if the clause ends a sentence
        byte_cursor: int - UTF-8 byte index until which the input text has been consumed.
            For clauses subsequent to the first, this is offset by the UTF-8 length of the first byte
            of the current clause, which had already been enqueued along with the previous clause.
    """
    ...
