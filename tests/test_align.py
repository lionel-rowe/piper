from math import ceil, floor
from pathlib import Path

from piper._aligner import (
    Aligner,
    PhonemeAlignment,
    WordAlignment,
    int_divide_to_n_parts,
    split_source_words,
)
from piper.config import PhonemeType, PiperConfig
from piper.phonemize_espeak import EspeakClause, EspeakSentence, RawPhonemeEspeakClause
from piper.voice import PiperVoice
from tests._phoneme_alignment_utils import Phoneme

_DIR = Path(__file__).parent
_TESTS_DIR = _DIR
_TEST_VOICE = _TESTS_DIR / "test_voice.onnx"
voice = PiperVoice.load(_TEST_VOICE)
aligner = Aligner(voice)


def test_split_source_words():
    text = "Hello, world! Hey there."
    words = split_source_words(text)
    assert words == ["Hello, ", "world! ", "Hey ", "there."]

    text = " ¡¿ wrapping punctuation ?! "
    words = split_source_words(text)
    assert words == [" ¡¿ wrapping ", "punctuation ?! "]

    text = "支持多语言处理。"
    words = split_source_words(text)
    assert words == ["支", "持", "多", "语", "言", "处", "理。"]


def test_phonemize_sentence() -> None:
    """Test [[ phonemes block ]] with phonemize_sentence."""
    phonemes = aligner.phonemize_sentence("Hello, world!")

    assert phonemes == EspeakSentence(
        [
            EspeakClause("Hello, ", list("həlˈoʊ, ")),
            EspeakClause("world!", list("wˈɜːld!")),
        ]
    )


def test_raw_phonemes_phonemize_sentence() -> None:
    """Test [[ phonemes block ]] with phonemize_sentence."""
    phonemes = aligner.phonemize_sentence("I am the [[ bˈætmæn ]] not [[bɹˈuːs wˈe‍ɪn]]")

    assert phonemes == EspeakSentence(
        [
            EspeakClause("I am the ", list("aɪɐm ðə ")),
            RawPhonemeEspeakClause("[[ bˈætmæn ]] ", list("bˈætmæn "), " bˈætmæn "),
            EspeakClause("not ", list("nˈɑːt ")),
            RawPhonemeEspeakClause(
                "[[bɹˈuːs wˈe‍ɪn]]", list("bɹˈuːs wˈe‍ɪn"), "bɹˈuːs wˈe‍ɪn"
            ),
        ],
    )


def test_phonemize_text() -> None:
    """Test [[ phonemes block ]] with phonemize_sentence."""
    phonemes = aligner.phonemize_text("Hello, world. Hey there!")

    assert phonemes == [
        EspeakSentence(
            [
                EspeakClause("Hello, ", list("həlˈoʊ, ")),
                EspeakClause("world. ", list("wˈɜːld.")),
            ]
        ),
        EspeakSentence(
            [
                EspeakClause("Hey there!", list("hˈeɪ ðˈɛɹ!")),
            ]
        ),
    ]


def test_align_basic():
    espeak_sentence = EspeakSentence(
        [
            EspeakClause("Hello, ", list("həlˈoʊ, ")),
            EspeakClause("world!", list("wˈɜːld!")),
        ]
    )

    phoneme_alignments = [
        PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
        PhonemeAlignment(phoneme="h", phoneme_ids=[-1], num_samples=Phoneme.h),
        PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="o", phoneme_ids=[-1], num_samples=Phoneme.o),
        PhonemeAlignment(phoneme="ʊ", phoneme_ids=[-1], num_samples=Phoneme.ʊ),
        PhonemeAlignment(phoneme=",", phoneme_ids=[-1], num_samples=Phoneme.COMMA),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="w", phoneme_ids=[-1], num_samples=Phoneme.w),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɜ", phoneme_ids=[-1], num_samples=Phoneme.ɜ),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="d", phoneme_ids=[-1], num_samples=Phoneme.d),
        PhonemeAlignment(phoneme="!", phoneme_ids=[-1], num_samples=Phoneme.BANG),
        PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
    ]

    aligned = list(aligner.align(phoneme_alignments, espeak_sentence))

    assert aligned == [
        WordAlignment(
            char_index=0,
            substring="Hello",
            num_samples=Phoneme.BOS
            + Phoneme.h
            + Phoneme.ə
            + Phoneme.l
            + Phoneme.STRESS
            + Phoneme.o
            + Phoneme.ʊ
            + Phoneme.COMMA
            + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=7,
            substring="world",
            num_samples=Phoneme.w
            + Phoneme.STRESS
            + Phoneme.ɜ
            + Phoneme.LONG
            + Phoneme.l
            + Phoneme.d
            + Phoneme.BANG
            + Phoneme.EOS,
        ),
    ]


def test_align_simple_word_alignment():
    espeak_sentence = EspeakSentence(
        [
            EspeakClause(
                "Hello\xa0world!",
                list("həlˈoʊ wˈɜːld."),
            ),
        ]
    )

    phoneme_alignments = [
        PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
        PhonemeAlignment(phoneme="h", phoneme_ids=[-1], num_samples=Phoneme.h),
        PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="o", phoneme_ids=[-1], num_samples=Phoneme.o),
        PhonemeAlignment(phoneme="ʊ", phoneme_ids=[-1], num_samples=Phoneme.ʊ),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="w", phoneme_ids=[-1], num_samples=Phoneme.w),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɜ", phoneme_ids=[-1], num_samples=Phoneme.ɜ),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="d", phoneme_ids=[-1], num_samples=Phoneme.d),
        PhonemeAlignment(phoneme="!", phoneme_ids=[-1], num_samples=Phoneme.BANG),
        PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
    ]

    aligned = list(aligner.align(phoneme_alignments, espeak_sentence))

    assert aligned == [
        WordAlignment(
            char_index=0,
            substring="Hello",
            num_samples=Phoneme.BOS
            + Phoneme.h
            + Phoneme.ə
            + Phoneme.l
            + Phoneme.STRESS
            + Phoneme.o
            + Phoneme.ʊ
            + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=6,
            substring="world",
            num_samples=Phoneme.w
            + Phoneme.STRESS
            + Phoneme.ɜ
            + Phoneme.LONG
            + Phoneme.l
            + Phoneme.d
            + Phoneme.BANG
            + Phoneme.EOS,
        ),
    ]


def test_align_with_raw_phonetic_input():
    espeak_sentence = EspeakSentence(
        [
            # # Actual: 'aɪɐm ðə' (not 'aɪ ɐm ðə')
            # # But we add a space before "ɐm" here because re-adding synthetic word boundaries is tested separately below.
            EspeakClause("I am the ", list("aɪ ɐm ðə ")),
            RawPhonemeEspeakClause("[[ bˈætmæn ]] ", list("bˈætmæn "), "bˈætmæn"),
            EspeakClause("not ", list("nˈɑːt ")),
            RawPhonemeEspeakClause(
                "[[bɹˈuːs wˈe\u200dɪn]]",
                list("bɹˈuːs wˈe\u200dɪn"),
                "bɹˈuːs wˈe\u200dɪn",
            ),
        ]
    )

    phoneme_alignments = [
        PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
        PhonemeAlignment(phoneme="a", phoneme_ids=[-1], num_samples=Phoneme.a),
        PhonemeAlignment(phoneme="ɪ", phoneme_ids=[-1], num_samples=Phoneme.ɪ),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="ɐ", phoneme_ids=[-1], num_samples=Phoneme.ɐ),
        PhonemeAlignment(phoneme="m", phoneme_ids=[-1], num_samples=Phoneme.m),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="ð", phoneme_ids=[-1], num_samples=Phoneme.ð),
        PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="b", phoneme_ids=[-1], num_samples=Phoneme.b),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="æ", phoneme_ids=[-1], num_samples=Phoneme.æ),
        PhonemeAlignment(phoneme="t", phoneme_ids=[-1], num_samples=Phoneme.t),
        PhonemeAlignment(phoneme="m", phoneme_ids=[-1], num_samples=Phoneme.m),
        PhonemeAlignment(phoneme="æ", phoneme_ids=[-1], num_samples=Phoneme.æ),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɑ", phoneme_ids=[-1], num_samples=Phoneme.ɑ),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme="t", phoneme_ids=[-1], num_samples=Phoneme.t),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="b", phoneme_ids=[-1], num_samples=Phoneme.b),
        PhonemeAlignment(phoneme="ɹ", phoneme_ids=[-1], num_samples=Phoneme.ɹ),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="u", phoneme_ids=[-1], num_samples=Phoneme.u),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme="s", phoneme_ids=[-1], num_samples=Phoneme.s),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="w", phoneme_ids=[-1], num_samples=Phoneme.w),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="e", phoneme_ids=[-1], num_samples=Phoneme.e),
        PhonemeAlignment(
            phoneme="\u200d", phoneme_ids=[-1], num_samples=Phoneme.U_200D
        ),
        PhonemeAlignment(phoneme="ɪ", phoneme_ids=[-1], num_samples=Phoneme.ɪ),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
    ]

    aligned = list(aligner.align(phoneme_alignments, espeak_sentence))

    assert aligned == [
        WordAlignment(
            char_index=0,
            substring="I",
            num_samples=Phoneme.BOS + Phoneme.a + Phoneme.ɪ + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=2,
            substring="am",
            num_samples=Phoneme.ɐ + Phoneme.m + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=5,
            substring="the",
            num_samples=Phoneme.ð + Phoneme.ə + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=12,
            substring="bˈætmæn",
            num_samples=Phoneme.SPACE
            + Phoneme.b
            + Phoneme.STRESS
            + Phoneme.æ
            + Phoneme.t
            + Phoneme.m
            + Phoneme.æ
            + Phoneme.n,
        ),
        WordAlignment(
            char_index=23,
            substring="not",
            num_samples=Phoneme.n
            + Phoneme.STRESS
            + Phoneme.ɑ
            + Phoneme.LONG
            + Phoneme.t
            + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=29,
            substring="bɹˈuːs",
            num_samples=Phoneme.b
            + Phoneme.ɹ
            + Phoneme.STRESS
            + Phoneme.u
            + Phoneme.LONG
            + Phoneme.s
            + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=36,
            substring="wˈe‍ɪn",
            num_samples=Phoneme.w
            + Phoneme.STRESS
            + Phoneme.e
            + Phoneme.U_200D
            + Phoneme.ɪ
            + Phoneme.n
            + Phoneme.EOS,
        ),
    ]


def test_backup_word_boundaries():
    espeak_sentence = EspeakSentence(
        [EspeakClause("Lord of the Rings", list("lˈɔːɹd ʌvðə ɹˈɪŋz"))]
    )

    phoneme_alignments = [
        PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɔ", phoneme_ids=[-1], num_samples=Phoneme.ɑ),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme="ɹ", phoneme_ids=[-1], num_samples=Phoneme.ɹ),
        PhonemeAlignment(phoneme="d", phoneme_ids=[-1], num_samples=Phoneme.d),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="ʌ", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme="v", phoneme_ids=[-1], num_samples=Phoneme.v),
        PhonemeAlignment(phoneme="ð", phoneme_ids=[-1], num_samples=Phoneme.ð),
        PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="ɹ", phoneme_ids=[-1], num_samples=Phoneme.ɹ),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɪ", phoneme_ids=[-1], num_samples=Phoneme.ɪ),
        PhonemeAlignment(phoneme="ŋ", phoneme_ids=[-1], num_samples=Phoneme.ŋ),
        PhonemeAlignment(phoneme="z", phoneme_ids=[-1], num_samples=Phoneme.z),
        PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
    ]

    aligned = list(aligner.align(phoneme_alignments, espeak_sentence))

    [of_samples, the_samples] = int_divide_to_n_parts(
        Phoneme.ə + Phoneme.v + Phoneme.ð + Phoneme.ə + Phoneme.SPACE,
        2,
    )

    assert aligned == [
        WordAlignment(
            char_index=0,
            substring="Lord",
            num_samples=Phoneme.BOS
            + Phoneme.l
            + Phoneme.STRESS
            + Phoneme.ɑ
            + Phoneme.LONG
            + Phoneme.ɹ
            + Phoneme.d
            + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=5,
            substring="of",
            num_samples=of_samples,
        ),
        WordAlignment(
            char_index=8,
            substring="the",
            num_samples=the_samples,
        ),
        WordAlignment(
            char_index=12,
            substring="Rings",
            num_samples=Phoneme.ɹ
            + Phoneme.STRESS
            + Phoneme.ɪ
            + Phoneme.ŋ
            + Phoneme.z
            + Phoneme.EOS,
        ),
    ]


def test_sloppy_word_boundaries():
    espeak_sentence = EspeakSentence(
        [EspeakClause("I am the Batman", list("aɪɐm ðə bˈætmæn"))]
    )

    phoneme_alignments = [
        PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
        PhonemeAlignment(phoneme="a", phoneme_ids=[-1], num_samples=Phoneme.a),
        PhonemeAlignment(phoneme="ɪ", phoneme_ids=[-1], num_samples=Phoneme.ɪ),
        PhonemeAlignment(phoneme="ɐ", phoneme_ids=[-1], num_samples=Phoneme.ɐ),
        PhonemeAlignment(phoneme="m", phoneme_ids=[-1], num_samples=Phoneme.m),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="ð", phoneme_ids=[-1], num_samples=Phoneme.ð),
        PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="b", phoneme_ids=[-1], num_samples=Phoneme.b),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="æ", phoneme_ids=[-1], num_samples=Phoneme.æ),
        PhonemeAlignment(phoneme="t", phoneme_ids=[-1], num_samples=Phoneme.t),
        PhonemeAlignment(phoneme="m", phoneme_ids=[-1], num_samples=Phoneme.m),
        PhonemeAlignment(phoneme="æ", phoneme_ids=[-1], num_samples=Phoneme.æ),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
    ]

    aligned = list(aligner.align(phoneme_alignments, espeak_sentence))

    [i_samples, am_samples] = int_divide_to_n_parts(
        Phoneme.BOS + Phoneme.a + Phoneme.ɪ + Phoneme.ɐ + Phoneme.m + Phoneme.SPACE,
        2,
    )

    assert aligned == [
        WordAlignment(
            char_index=0,
            substring="I",
            num_samples=i_samples,
        ),
        WordAlignment(
            char_index=2,
            substring="am",
            num_samples=am_samples,
        ),
        WordAlignment(
            char_index=5,
            substring="the",
            num_samples=Phoneme.ð + Phoneme.ə + Phoneme.SPACE,
        ),
        WordAlignment(
            char_index=9,
            substring="Batman",
            num_samples=Phoneme.b
            + Phoneme.STRESS
            + Phoneme.æ
            + Phoneme.t
            + Phoneme.m
            + Phoneme.æ
            + Phoneme.n
            + Phoneme.EOS,
        ),
    ]


# def test_multiple_phoneme_words_rendered_as_one():
#     espeak_sentence = EspeakSentence(
#         [EspeakClause("run-of-the-mill", list("ɹˈʌnəðəmˈɪl"))]
#     )

#     phoneme_alignments = [
#         PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
#         PhonemeAlignment(phoneme="ɹ", phoneme_ids=[-1], num_samples=Phoneme.ɹ),
#         PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
#         PhonemeAlignment(phoneme="ʌ", phoneme_ids=[-1], num_samples=Phoneme.ʌ),
#         PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
#         PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
#         PhonemeAlignment(phoneme="ð", phoneme_ids=[-1], num_samples=Phoneme.ð),
#         PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
#         PhonemeAlignment(phoneme="m", phoneme_ids=[-1], num_samples=Phoneme.m),
#         PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
#         PhonemeAlignment(phoneme="ɪ", phoneme_ids=[-1], num_samples=Phoneme.ɪ),
#         PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
#         PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
#     ]

#     aligned = list(aligner.align(phoneme_alignments, espeak_sentence))

#     [of_samples, the_samples] = int_divide_to_n_parts(
#         Phoneme.ə + Phoneme.ð + Phoneme.ə ,
#         2,
#     )

#     assert aligned == [
#         WordAlignment(
#             char_index=0,
#             substring="run",
#             num_samples=Phoneme.BOS + Phoneme.ɹ + Phoneme.STRESS + Phoneme.ʌ + Phoneme.n,
#         ),
#         WordAlignment(
#             char_index=4,
#             substring="of",
#             num_samples=of_samples,
#         ),
#         WordAlignment(
#             char_index=7,
#             substring="the",
#             num_samples=the_samples,
#         ),
#         WordAlignment(
#             char_index=11,
#             substring="mill",
#             num_samples=Phoneme.m
#             + Phoneme.STRESS
#             + Phoneme.ɪ
#             + Phoneme.l
#             + Phoneme.EOS,
#         ),
#     ]


def test_numerals():
    # You have 123 friends"
    # juː hav wˈɒnhˈʌndɹɪdən twˈɛnti θɹˈiː fɹˈɛndz

    espeak_sentence = EspeakSentence(
        [EspeakClause("You have 123 friends", list("juː hav wˈɒnhˈʌndɹɪdən twˈɛnti θɹˈiː fɹˈɛndz"))]
    )

    phoneme_alignments = [
        PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
        PhonemeAlignment(phoneme="j", phoneme_ids=[-1], num_samples=Phoneme.j),
        PhonemeAlignment(phoneme="u", phoneme_ids=[-1], num_samples=Phoneme.u),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="h", phoneme_ids=[-1], num_samples=Phoneme.h),
        PhonemeAlignment(phoneme="a", phoneme_ids=[-1], num_samples=Phoneme.a),
        PhonemeAlignment(phoneme="v", phoneme_ids=[-1], num_samples=Phoneme.v),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="w", phoneme_ids=[-1], num_samples=Phoneme.w),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɒ", phoneme_ids=[-1], num_samples=Phoneme.ɒ),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme="h", phoneme_ids=[-1], num_samples=Phoneme.h),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ʌ", phoneme_ids=[-1], num_samples=Phoneme.ʌ),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme="d", phoneme_ids=[-1], num_samples=Phoneme.d),
        PhonemeAlignment(phoneme="ɹ", phoneme_ids=[-1], num_samples=Phoneme.ɹ),
        PhonemeAlignment(phoneme="ɪ", phoneme_ids=[-1], num_samples=Phoneme.ɪ),
        PhonemeAlignment(phoneme="d", phoneme_ids=[-1], num_samples=Phoneme.d),
        PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="t", phoneme_ids=[-1], num_samples=Phoneme.t),
        PhonemeAlignment(phoneme="w", phoneme_ids=[-1], num_samples=Phoneme.w),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɛ", phoneme_ids=[-1], num_samples=Phoneme.ɛ),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme="t", phoneme_ids=[-1], num_samples=Phoneme.t),
        PhonemeAlignment(phoneme="i", phoneme_ids=[-1], num_samples=Phoneme.i),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="θ", phoneme_ids=[-1], num_samples=Phoneme.θ),
        PhonemeAlignment(phoneme="ɹ", phoneme_ids=[-1], num_samples=Phoneme.ɹ),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="i", phoneme_ids=[-1], num_samples=Phoneme.i),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="f", phoneme_ids=[-1], num_samples=Phoneme.f),
        PhonemeAlignment(phoneme="ɹ", phoneme_ids=[-1], num_samples=Phoneme.ɹ),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɛ", phoneme_ids=[-1], num_samples=Phoneme.ɛ),
        PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
        PhonemeAlignment(phoneme="d", phoneme_ids=[-1], num_samples=Phoneme.d),
        PhonemeAlignment(phoneme="z", phoneme_ids=[-1], num_samples=Phoneme.z),
        PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
    ]

    aligned = list(aligner.align(phoneme_alignments, espeak_sentence))

    assert aligned == [
    ]

    