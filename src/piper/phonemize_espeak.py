"""Phonemization with espeak-ng."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Union

_DIR = Path(__file__).parent
ESPEAK_DATA_DIR = _DIR / "espeak-ng-data"


@dataclass
class EspeakClause:
    text: str
    phonemes: list[str]

    def __eq__(self, other):
        if not isinstance(other, EspeakClause):
            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with other types"
            )
        # e.g. different subclasses
        if type(self) != type(other):
            return False
        return self.text == other.text and self.phonemes == other.phonemes


@dataclass
class RawPhonemeEspeakClause(EspeakClause):
    content: str


@dataclass
class EspeakSentence:
    def __init__(self, clauses: list[EspeakClause] | None = None):
        self.clauses = [] if clauses is None else clauses

    def __eq__(self, other):
        if not isinstance(other, EspeakSentence):
            raise TypeError(
                f"Cannot compare {self.__class__.__name__} with other types"
            )
        # e.g. different subclasses
        if type(self) != type(other):
            return False
        return self.clauses == other.clauses

    def __repr__(self):
        return f"{self.__class__.__name__}({self.clauses!r})"


class EspeakPhonemizer:
    """Phonemizer that uses espeak-ng."""

    def __init__(self, espeak_data_dir: Union[str, Path] = ESPEAK_DATA_DIR) -> None:
        """Initialize phonemizer."""
        from . import espeakbridge  # avoid circular import

        espeakbridge.initialize(str(espeak_data_dir))

    def phonemize(self, voice: str, text: str) -> list[EspeakSentence]:
        """Text to phonemes grouped by sentence."""
        from . import espeakbridge  # avoid circular import

        espeakbridge.set_voice(voice)

        sentences: list[EspeakSentence] = []
        sentence = EspeakSentence()
        encoded = text.encode("utf-8")

        # Next UTF-8 char after clause is pre-loaded into espeak after each clause,
        # so we need to amend end_byte_index by its byte length
        enqueued_next_char = ""
        byte_index = 0

        clause_phonemes = espeakbridge.get_phonemes(text)

        for (
            phonemes_str,
            terminator_str,
            end_of_sentence,
            end_byte_index,
        ) in clause_phonemes:
            input_chunk = encoded[byte_index:end_byte_index].decode("utf-8")
            byte_index = end_byte_index
            clause_str = enqueued_next_char + input_chunk[:-1]

            enqueued_next_char = input_chunk[-1] if input_chunk else ""

            # Filter out (lang) switch (flags).
            # These surround words from languages other than the current voice.
            phonemes_str = re.sub(r"\([^)]+\)", "", phonemes_str)

            # Keep punctuation even though it's not technically a phoneme
            phonemes_str += terminator_str
            if terminator_str in (",", ":", ";"):
                # Not a sentence boundary
                phonemes_str += " "

            # Decompose phonemes into UTF-8 codepoints.
            # This separates accent characters into separate "phonemes".
            phonemes = list(unicodedata.normalize("NFD", phonemes_str))

            clause = EspeakClause(
                text=clause_str,
                phonemes=phonemes,
            )

            sentence.clauses.append(clause)

            if end_of_sentence:
                sentences.append(sentence)
                sentence = EspeakSentence()

        if sentence.clauses:
            sentences.append(sentence)

        if sentences:
            sentences[-1].clauses[-1].text += enqueued_next_char

        return sentences
