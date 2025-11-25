import regex
from diff_match_patch import diff_match_patch
from unicode_segment import WordSegmenter

dmp = diff_match_patch()
dmp.Match_Distance = 10

word_segmenter = WordSegmenter()

WORD_LIKE_PATTERN = regex.compile(r"[[\p{L}\p{N}\p{S}]--[`]]", regex.V1)


def is_word_like(text: str) -> bool:
    """
    Match any "word-like" character - letters, numbers, symbols, i.e. anything
    that's probably semantically important.
    """
    return WORD_LIKE_PATTERN.search(text) is not None


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


def get_matched_portion(full_text: str, pattern: str) -> tuple[int, int]:
    # fast paths:
    if not pattern:
        return 0, 0
    if full_text.startswith(pattern):
        return 0, len(pattern)

    words_full_text = word_segmenter.segment(full_text)
    words_pattern = list(word_segmenter.segment(pattern))

    segments: list[tuple[int, str]] = []

    i = 0
    # grab the same number of words as in pattern
    for segment in words_full_text:
        if i >= len(words_pattern):
            break
        segments.append(segment)
        i += 1

    if not segments:
        return 0, 0

    return segments[0][0], segments[-1][0] + len(segments[-1][1])
