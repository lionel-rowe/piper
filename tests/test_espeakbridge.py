from pathlib import Path

import pytest

from piper import espeakbridge

# [1, 2, 3, 4] bytes respectively in UTF-8
chars = ["B", "Б", "文", "🦄"]


@pytest.mark.parametrize("char", chars)
def test_get_phonemes_cursor_offset(char: str):
    espeakbridge.set_voice("en-us")

    prefix = "A! "
    suffix = " c"
    text = prefix + char + suffix

    result = espeakbridge.get_phonemes(text)
    print(result)

    # contains two clauses
    assert len(result) == 2
    # cursor after first clause is offset after the first character of the next clause
    assert result[0][3] == len(text[0 : len(prefix + char)].encode("utf-8"))
    # cursor after last clause is the full byte length of the text
    assert result[1][3] == len(text.encode("utf-8"))
