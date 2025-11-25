"""Phonemization with espeak-ng."""

import re
import unicodedata
from pathlib import Path
from typing import Union

_DIR = Path(__file__).parent
ESPEAK_DATA_DIR = _DIR / "espeak-ng-data"


class EspeakPhonemizer:
    """Phonemizer that uses espeak-ng."""

    def __init__(self, espeak_data_dir: Union[str, Path] = ESPEAK_DATA_DIR) -> None:
        """Initialize phonemizer."""
        from . import espeakbridge  # avoid circular import

        espeakbridge.initialize(str(espeak_data_dir))

    def phonemize(self, voice: str, text: str) -> list[list[str]]:
        """Text to phonemes grouped by sentence."""
        phonemized = self._phonemize_clausewise(voice, text)
        out: list[list[str]] = []

        for sentence in phonemized:
            sentence_phonemes: list[str] = []
            for clause in sentence:
                _, clause_phonemes = clause
                sentence_phonemes.extend(clause_phonemes)
            out.append(sentence_phonemes)

        return out

    def _phonemize_clausewise(
        self, voice: str, text: str
    ) -> list[list[tuple[str, list[str]]]]:
        """Text to phonemes grouped by sentence and clause, including input text for each clause."""
        from . import espeakbridge  # avoid circular import

        espeakbridge.set_voice(voice)

        all_phonemes: list[list[tuple[str, list[str]]]] = []
        sentence_phonemes: list[tuple[str, list[str]]] = []

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
            byte_cursor,
        ) in clause_phonemes:
            input_chunk = encoded[byte_index:byte_cursor].decode("utf-8")
            byte_index = byte_cursor
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
            sentence_phonemes.append(
                (clause_str, list(unicodedata.normalize("NFD", phonemes_str)))
            )

            if end_of_sentence:
                all_phonemes.append(sentence_phonemes)
                sentence_phonemes = []

        if sentence_phonemes:
            all_phonemes.append(sentence_phonemes)

        # patch last clause to include any enqueued char
        if all_phonemes and all_phonemes[-1] and all_phonemes[-1][-1]:
            all_phonemes[-1][-1] = (
                all_phonemes[-1][-1][0] + enqueued_next_char,
                all_phonemes[-1][-1][1],
            )

        return all_phonemes
