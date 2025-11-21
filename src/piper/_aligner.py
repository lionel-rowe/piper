from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Sequence

import regex
from unicode_segment.sentence import SentenceSegmenter
from unicode_segment.word import WordSegmenter

from piper.config import PhonemeType
from piper.phonemize_espeak import (
    EspeakClause,
    EspeakPhonemizer,
    EspeakSentence,
    RawPhonemeEspeakClause,
)
from piper.tashkeel import TashkeelDiacritizer

from .phoneme_ids import phonemes_to_ids
from .phonemize_espeak import (
    EspeakClause,
    EspeakPhonemizer,
    EspeakSentence,
    RawPhonemeEspeakClause,
)

if TYPE_CHECKING:
    from .voice import PiperVoice


_ESPEAK_PHONEMIZER: EspeakPhonemizer | None = None
_ESPEAK_PHONEMIZER_LOCK = threading.Lock()


@dataclass
class PhonemeAlignment:
    phoneme: str
    phoneme_ids: Sequence[int]
    num_samples: int


@dataclass
class WordAlignment:
    char_index: int
    substring: str
    num_samples: int


word_segmenter = WordSegmenter()


_WORD_LIKE_PATTERN = regex.compile(r"[\p{L}\p{N}\p{S}]", regex.V1)
"""
Matches any "word-like" character (letters, numbers, symbols, i.e. anything that's likely to be semantically important).
"""
_WRAPPING_NON_WORD_LIKE_PATTERN = regex.compile(
    rf"^[^{_WORD_LIKE_PATTERN.pattern}]+|[^{_WORD_LIKE_PATTERN.pattern}]+$",
    regex.V1,
)
"""Matches any non-word-like characters to be stripped from the start or end of a string."""
_NON_PHONEME_PATTERN = regex.compile(
    rf"^\^?[^{_WORD_LIKE_PATTERN.pattern}]*|[^{_WORD_LIKE_PATTERN.pattern}]*\$?$",
    regex.V1,
)
"""Matches non-semantic characters to be stripped from phonemes."""


_SPACE_PATTERN = regex.compile(r"[\p{space}]", regex.V1)
"""Matches a single space character."""

_PHONEME_BLOCK_PATTERN = regex.compile(
    r"\[\[(?<content>.*?)\]\]\p{space}*", regex.V1 | regex.DOTALL
)
"""Matches phoneme blocks like `[[<content>]]` in the text, along with any trailing space."""


class BlockType(Enum):
    TEXT = "text"
    """Represents a span of normal text, e.g. "hello" """
    PHONEME = "phoneme_block"
    """Represents a double-bracket-wrapped block of raw phonemes, e.g. "[[ həˈloʊ ]]" """


def is_word_like(text: str) -> bool:
    return _WORD_LIKE_PATTERN.search(text) is not None


def split_source_words(sentence: str) -> list[str]:
    words: list[str] = []

    for _, word in word_segmenter.segment(sentence):
        if not words or (
            is_word_like(word) and (len(words) > 1 or is_word_like(words[0]))
        ):
            words.append(word)
        else:
            words[-1] += word

    return words


class Aligner:
    tashkeel_diacritizier: TashkeelDiacritizer | None = None

    def __init__(self, voice: PiperVoice):
        self.voice = voice

    def get_phonemes_from_sentences(
        self, sentences: list[EspeakSentence]
    ) -> list[list[str]]:
        text_part_phonemes: list[list[str]] = []
        for sentence in sentences:
            text_part_phonemes.append(
                [c for clause in sentence.clauses for c in clause.phonemes]
            )

        return text_part_phonemes

    def phonemize_text(self, text: str) -> list[EspeakSentence]:
        """
        Text to phonemes grouped by sentence > clause, with the original text chunk that produced them.

        :param text: Text to phonemize.
        :return: List of sentences, each containing list of clauses with their phonemes.
        """
        sentences = [s[1] for s in SentenceSegmenter().segment(text)]
        return [self.phonemize_sentence(s) for s in sentences]

    def phonemize_sentence(self, sentence: str) -> EspeakSentence:
        global _ESPEAK_PHONEMIZER

        if self.voice.config.phoneme_type == PhonemeType.TEXT:
            # Phonemes = codepoints
            codepoints = list(unicodedata.normalize("NFD", sentence))
            return EspeakSentence([EspeakClause(sentence, codepoints)])

        if self.voice.config.phoneme_type != PhonemeType.ESPEAK:
            raise ValueError(
                f"Unexpected phoneme type: {self.voice.config.phoneme_type}"
            )

        matches = list(_PHONEME_BLOCK_PATTERN.finditer(sentence))
        blocks: list[tuple[BlockType, str, str]] = []
        last_index = 0
        for match in matches:
            if match.start() > last_index:
                text = sentence[last_index : match.start()]
                blocks.append((BlockType.TEXT, text, text))
            blocks.append((BlockType.PHONEME, match.group(0), match.group("content")))

            last_index = match.end()
        if last_index < len(sentence):
            text = sentence[last_index:]
            blocks.append((BlockType.TEXT, text, text))

        out = EspeakSentence()

        for i, block in enumerate(blocks):
            (kind, raw_text, content) = block

            append_space = False
            next_part = blocks[i + 1] if i + 1 < len(blocks) else None
            if next_part:
                append_space = (
                    (_SPACE_PATTERN.match(raw_text, -1) is not None)
                    or (
                        kind == BlockType.PHONEME
                        and _SPACE_PATTERN.match(content, -1) is not None
                    )
                    or (_SPACE_PATTERN.match(next_part[2]) is not None)
                    or (
                        next_part[0] == BlockType.PHONEME
                        and _SPACE_PATTERN.match(next_part[1]) is not None
                    )
                )

            if kind == BlockType.PHONEME:
                phonemes: list[str] = []
                phonemes.extend(unicodedata.normalize("NFD", content.strip()))

                if append_space:
                    phonemes.append(" ")

                out.clauses.append(
                    RawPhonemeEspeakClause(
                        text=raw_text, phonemes=phonemes, content=content
                    )
                )

                continue

            if not raw_text:
                continue

            # Arabic diacritization
            if (self.voice.config.espeak_voice == "ar") and self.voice.use_tashkeel:
                if self.tashkeel_diacritizier is None:
                    self.tashkeel_diacritizier = TashkeelDiacritizer()

                content = self.tashkeel_diacritizier(
                    raw_text, taskeen_threshold=self.voice.taskeen_threshold
                )

            with _ESPEAK_PHONEMIZER_LOCK:
                if _ESPEAK_PHONEMIZER is None:
                    _ESPEAK_PHONEMIZER = EspeakPhonemizer(self.voice.espeak_data_dir)

                # Generally there should only be a single sentence output given the
                # input should be a single sentence; however, we need to handle the
                # case where espeak's sentence segmenting gives a different result from
                # our segmentation. In any such cases, we simply flatten the
                # multi-"sentence" output into clauses of a single output sentence.
                sentences = _ESPEAK_PHONEMIZER.phonemize(
                    self.voice.config.espeak_voice, content
                )

                for s in sentences:
                    if content != raw_text:
                        # If text_part has been modified (e.g. by tashkeel),
                        # intra-sentence clause boundaries may have changed,
                        # so we just emit the full sentence as one clause.
                        phonemes = [p for clause in s.clauses for p in clause.phonemes]
                        out.clauses.append(EspeakClause(raw_text, phonemes))

                    else:
                        out.clauses.extend(s.clauses)

                if out.clauses and append_space:
                    out.clauses[-1].phonemes.append(" ")

        return out

    def phonemize_word(self, word: str) -> list[str]:
        """
        Only used for re-phonemization when word boundaries couldn't be matched first time around.
        """
        global _ESPEAK_PHONEMIZER
        with _ESPEAK_PHONEMIZER_LOCK:
            if _ESPEAK_PHONEMIZER is None:
                _ESPEAK_PHONEMIZER = EspeakPhonemizer(self.voice.espeak_data_dir)

            # Generally there should only be a single sentence output given the
            # input should be a single word; however, we need to handle the
            # case where espeak's sentence segmenting gives a different result from
            # our segmentation. In any such cases, we simply flatten the
            # multi-"sentence" output into a single phoneme list.
            sentences = _ESPEAK_PHONEMIZER.phonemize(
                self.voice.config.espeak_voice, word
            )

            return [
                p for s in sentences for clause in s.clauses for p in clause.phonemes
            ]

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """
        Phonemes to ids.

        :param phonemes: List of phonemes.
        :return: List of phoneme ids.
        """
        return phonemes_to_ids(phonemes, self.voice.config.phoneme_id_map)

    char_index = 0

    def create_word_alignment(
        self, text_part: str, phoneme_alignments: list[PhonemeAlignment]
    ) -> WordAlignment:
        stripped = _WRAPPING_NON_WORD_LIKE_PATTERN.sub("", text_part)
        stripped_index = text_part.find(stripped)

        start_char_index = self.char_index + stripped_index
        self.char_index += len(text_part)
        num_samples = sum(pa.num_samples for pa in phoneme_alignments)
        return WordAlignment(
            char_index=start_char_index,
            substring=stripped,
            num_samples=num_samples,
        )

    def distribute_alignments(
        self, words: list[str], alignments: list[PhonemeAlignment]
    ):
        """
        Divide the number of samples equally between the words.
        """
        if not words:
            return
        elif len(words) == 1:
            yield self.create_word_alignment(words[0], alignments)
        else:
            total_samples = sum(pa.num_samples for pa in alignments)
            samples_per_word = int_divide_to_n_parts(total_samples, len(words))

            for w_idx, word in enumerate(words):
                dummy_phoneme_alignment = PhonemeAlignment(
                    phoneme="_",
                    phoneme_ids=[-1],
                    num_samples=samples_per_word[w_idx],
                )
                yield self.create_word_alignment(word, [dummy_phoneme_alignment])

    def align(
        self,
        phoneme_alignments: list[PhonemeAlignment],
        espeak_sentence: EspeakSentence,
    ) -> Iterable[WordAlignment]:
        self.char_index = 0

        if not phoneme_alignments or not espeak_sentence.clauses:
            return

        clauses_with_aligmnents: list[tuple[EspeakClause, list[PhonemeAlignment]]] = []
        phoneme_idx = 0
        for clause in espeak_sentence.clauses:
            clause_phoneme_alignments: list[PhonemeAlignment] = []
            for p in clause.phonemes:
                while phoneme_idx < len(phoneme_alignments):
                    next_alignment = phoneme_alignments[phoneme_idx]
                    phoneme_idx += 1

                    clause_phoneme_alignments.append(next_alignment)
                    if next_alignment.phoneme == p:
                        break
            clauses_with_aligmnents.append((clause, clause_phoneme_alignments))

        clauses_with_aligmnents[-1][1].extend(phoneme_alignments[phoneme_idx:])

        for clause, alignments_for_clause in clauses_with_aligmnents:
            words = split_source_words(clause.text)
            phoneme_words: list[list[PhonemeAlignment]] = []

            prev_is_space = False
            for alignment in alignments_for_clause:
                if not phoneme_words or prev_is_space:
                    phoneme_words.append([])
                group = phoneme_words[-1]
                prev_is_space = (
                    alignment.phoneme == " " if alignment is not None else False
                )
                group.append(alignment)

            # First attempt (happy path): words match up 1:1 with phoneme_words
            if len(words) == len(phoneme_words):
                for w_idx, word in enumerate(words):
                    group = phoneme_words[w_idx]
                    yield self.create_word_alignment(word, group)
            else:
                # Second attempt (backup): try re-phonemizing each word to see if we can get a better match,
                # distributing equally among words in cases that couldn't be resolved

                first_pass = [
                    "".join(x.phoneme for x in word) for word in phoneme_words
                ]
                assert sum(len(w) for w in first_pass) == len(
                    alignments_for_clause
                ), "Invariant: first_pass includes all of alignments_for_clause, 1 phoneme char per alignment"

                alignments_by_first_pass_word: list[list[PhonemeAlignment]] = []
                i = 0
                for w in first_pass:
                    g: list[PhonemeAlignment] = []
                    alignments_by_first_pass_word.append(g)
                    for ws in range(len(w)):
                        g.append(alignments_for_clause[i])
                        i += 1

                print(alignments_by_first_pass_word)

                second_pass = ["".join(self.phonemize_word(word)) for word in words]

                snap_points = snap_needles_within_haystacks(second_pass, first_pass)
                assert len(snap_points) == len(
                    words
                ), "Invariant: snapped_indexes corresponds 1:1 to words"

                print(f"first_pass={first_pass}")
                print(f"second_pass={second_pass}")
                print(f"snap_points={snap_points}")

                for ws, gs in group_words_with_alignments(
                    words, alignments_by_first_pass_word, snap_points
                ):
                    yield from self.distribute_alignments(ws, gs)


def int_divide_to_n_parts(total: int, n_parts: int) -> list[int]:
    """
    Divide integer n into num_parts parts as equally as possible.
    For example, dividing 10 into 3 parts would result in [4, 3, 3].
    """
    base = total // n_parts
    remainder = total % n_parts
    parts = [base] * n_parts
    for i in range(remainder):
        parts[i] += 1
    return parts


class Position(Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"
    WHOLE = "whole"


def snap_needles_within_haystacks(
    needles: list[str],
    haystacks: list[str],
) -> list[int | None]:
    """
    Snaps segments from the needles to their found indexes in the haystacks.

    Output has same length as, and corresponds to items of, `needles`.

    Output items are the index_in_list of the haystacks where the needle was found, or None if not found.
    Snapping is:
    - To previous item if previous position is END or MIDDLE
    - To next item if next position is START or MIDDLE
    - To nothing (`None`) otherwise
    """
    aligned = exact_needles_within_haystacks(needles, haystacks)
    result: list[int | None] = []

    for i, alignment in enumerate(aligned):
        if alignment is not None:
            # Found alignment
            result.append(alignment[0])
        else:
            # Not found, need to snap
            snapped = None

            # Find the last non-None alignment before this position
            prev_alignment = None
            for j in range(i - 1, -1, -1):
                if aligned[j] is not None:
                    prev_alignment = aligned[j]
                    break

            # Check if previous is START or MIDDLE - snap to previous
            if prev_alignment is not None:
                prev_idx, prev_pos = prev_alignment
                if prev_pos in (Position.START, Position.MIDDLE):
                    snapped = prev_idx

            # Check next item if we didn't snap to previous - snap if next is not WHOLE
            if snapped is None:
                for j in range(i + 1, len(aligned)):
                    a = aligned[j]
                    if a is not None:
                        next_idx, next_pos = a
                        if next_pos != Position.WHOLE:
                            snapped = next_idx
                        break

            result.append(snapped)

    can_front_fill = 0 not in result
    can_back_fill = (len(result) - 1) not in result

    if can_front_fill:
        for i, x in enumerate(result):
            if x is None:
                result[i] = 0
            else:
                break

    if can_back_fill:
        for i in range(len(result) - 1, -1, -1):
            x = result[i]
            if x is None:
                result[i] = len(result) - 1
            else:
                break

    return result


def exact_needles_within_haystacks(
    needles: list[str],
    haystacks: list[str],
) -> list[tuple[int, Position] | None]:
    """
    Aligns segments from the needles to their found indexes in the haystacks.

    Output tuples are (index_in_list, index_within_item).

    - `needles` always has more items than `haystacks`.
    - `needles` items don't always match `haystacks` items exactly.
    - Matching can only be done forward (both within and between items).
    """
    result: list[tuple[int, Position] | None] = []
    haystacks_cursor = 0
    prev_idx = -1
    prev_pos = 0

    # normalize both
    needles = [_NON_PHONEME_PATTERN.sub("", needle) for needle in needles]
    haystacks = [_NON_PHONEME_PATTERN.sub("", haystack) for haystack in haystacks]

    for needle in needles:
        found = False
        print(f"needle='{needle}'")
        for i in range(haystacks_cursor, len(haystacks)):
            haystack = haystacks[i]

            pos = haystack.find(needle, prev_pos if i == prev_idx else 0)

            print(f"haystack='{haystack}', pos={pos}")

            if pos >= 0:
                if needle == haystack:
                    position = Position.WHOLE
                else:
                    if pos == 0:
                        position = Position.START
                    elif pos + len(needle) == len(haystack):
                        position = Position.END
                    else:
                        position = Position.MIDDLE

                result.append((i, position))
                prev_pos = pos + len(needle) if i == prev_idx else 0
                prev_idx = i
                haystacks_cursor = i
                found = True
                break
        if not found:
            result.append(None)

    print(result)

    return result


def group_words_with_alignments(
    words: list[str],
    alignments: list[list[PhonemeAlignment]],
    snapped_indexes: list[int | None],
):
    result: list[tuple[list[str], list[PhonemeAlignment]]] = []
    current_group_words = []
    current_timing_index = None

    for i, (word, snap_idx) in enumerate(zip(words, snapped_indexes)):
        if current_timing_index is None:
            # First word
            current_timing_index = snap_idx
            current_group_words = [word]
        elif snap_idx == current_timing_index:
            # Same timing group
            current_group_words.append(word)
        else:
            # New timing group - save previous group
            result.append((current_group_words, alignments[current_timing_index]))
            current_timing_index = snap_idx
            current_group_words = [word]

    # Add the last group
    if current_group_words and current_timing_index is not None:
        result.append((current_group_words, alignments[current_timing_index]))

    return result
