from dataclasses import dataclass
from typing import Iterable, Sequence

import regex
from unicode_segment import WordSegmenter

from piper.const import BOS, EOS
from piper.phonemize_espeak import EspeakSentence


@dataclass
class PhonemeAlignment:
    phoneme: str
    phoneme_ids: Sequence[int]
    num_samples: int


@dataclass
class CharAlignment:
    char_index: int
    substring: str
    num_samples: int


def is_word_like(s: str) -> bool:
    return regex.search(r"[\p{L}\p{M}\p{N}]", s) is not None


word_segmenter = WordSegmenter()


def split_source_words(sentence: str) -> list[str]:
    words: list[str] = []

    for _, word in word_segmenter.segment(sentence):
        if is_word_like(word) or not words:
            words.append(word)
        else:
            words[-1] += word

    return words


def align(
    phoneme_alignments: list[PhonemeAlignment], espeak_sentence: EspeakSentence
) -> Iterable[CharAlignment]:
    char_idx = 0
    phoneme_idx = 0

    def create_char_alignment(
        text_part: str, phoneme_alignments: list[PhonemeAlignment]
    ) -> CharAlignment:
        nonlocal char_idx
        start_char_index = char_idx
        char_idx += len(text_part)
        num_samples = sum(pa.num_samples for pa in phoneme_alignments)
        return CharAlignment(
            char_index=start_char_index,
            substring=text_part,
            num_samples=num_samples,
        )

    for clause in espeak_sentence.clauses:
        is_raw_phoneme_clause = clause.text.startswith("[[")
        if is_raw_phoneme_clause:
            # unimplemented
            raise NotImplementedError("Raw phoneme clause alignment not implemented")

            # Raw phoneme clause, just append whole clause
            yield create_char_alignment(
                clause.text,
                [
                    # TODO
                ],
            )
            continue

        words = split_source_words(clause.text)
        phonemes_groups: list[list[PhonemeAlignment]] = []
        prev_is_space = False
        for _ in range(len(clause.phonemes)):
            if not phonemes_groups or prev_is_space:
                phonemes_groups.append([])

            group = phonemes_groups[-1]

            alignment = None
            while alignment == None or alignment.phoneme == BOS:
                if phoneme_idx >= len(phoneme_alignments):
                    break
                alignment = phoneme_alignments[phoneme_idx]
                group.append(alignment)
                phoneme_idx += 1

            prev_is_space = alignment.phoneme == " "

        if (
            phonemes_groups
            and phoneme_idx < len(phoneme_alignments)
            and phoneme_alignments[phoneme_idx].phoneme == EOS
        ):
            phonemes_groups[-1].append(phoneme_alignments[phoneme_idx])
            phoneme_idx += 1

        if len(words) == len(phonemes_groups):
            for w_idx, word in enumerate(words):
                group = phonemes_groups[w_idx]
                yield create_char_alignment(word, group)
        else:
            # Fallback: just append whole clause if words can't be remapped to phonemes
            print(
                f"\x1b[31mWarning: couldn't align chars for clause: '{clause.text}'\x1b[0m"
            )

            yield create_char_alignment(
                clause.text, [p for phonemes in phonemes_groups for p in phonemes]
            )

        print(f"words={words}")
        print(
            f"phonemes_groups={["".join(p.phoneme for p in group) for group in phonemes_groups]}"
        )
