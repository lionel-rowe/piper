# from piper._aligner import align_second_pass

# test_cases = [
#     {
#         "first_pass": [],
#         "second_pass": [],
#         # empty lists, haystack: ""
#         "expected": [],
#     },
#     {
#         "first_pass": ["hˈɛloʊ"],
#         "second_pass": ["hˈɛloʊ"],
#         # haystack: "hˈɛloʊ"
#         # single segment exact match
#         "expected": ["hˈɛloʊ"],
#     },
#     {
#         "first_pass": ["lˈɔːɹd ", "ʌv", "ðə ", "ɹˈɪŋz"],
#         "second_pass": ["lˈɔːɹd", "ʌv", "ðə", "ɹˈɪŋz"],
#         # haystack: "lˈɔːɹd ʌv ðə ɹˈɪŋz"
#         # All segments match exactly in the haystack
#         "expected": ["lˈɔːɹd ", "ʌv", "ðə ", "ɹˈɪŋz"],
#     },
#     {
#         "first_pass": ["aɪɐm ", "ðə ", "bˈætmæn"],
#         "second_pass": ["ˈaɪ", "æm", "ðə", "bˈætmæn"],
#         # haystack: "ˈaɪ æm ðə bˈætmæn"
#         # "aɪɐm " not found, "ðə " found, "bˈætmæn" found
#         "expected": [None, "ðə ", "bˈætmæn"],
#     },
#     {
#         "first_pass": ["test"],
#         "second_pass": ["tes"],
#         # haystack: "tes"
#         # "test" not found (haystack too short)
#         "expected": [None],
#     },
#     {
#         "first_pass": ["a ", "b ", "c"],
#         "second_pass": ["a", "b", "c"],
#         # haystack: "a b c"
#         # All segments with trailing space match
#         "expected": ["a ", "b ", "c"],
#     },
#     {
#         "first_pass": ["hello", " ", "world"],
#         "second_pass": ["hello", "world"],
#         # haystack: "hello world"
#         # Space character matches between words
#         "expected": ["hello", " ", "world"],
#     },
#     {
#         "first_pass": ["abc", "def", "ghi"],
#         "second_pass": ["abcdefghi"],
#         # haystack: "abcdefghi"
#         # Consecutive matches without spaces
#         "expected": ["abc", "def", "ghi"],
#     },
#     {
#         "first_pass": ["x", "y", "z"],
#         "second_pass": ["z", "y", "x"],
#         # haystack: "z y x"
#         # 'x' found at position 4, cursor moves to 5
#         # 'y' found at position 2 which is < 5, so it fails
#         # 'z' found at position 0 which is < 5, so it fails
#         "expected": ["x", None, None],
#     },
#     {
#         "first_pass": ["foo", "bar", "baz"],
#         "second_pass": ["foo", "qux", "bar", "baz"],
#         # haystack: "foo qux bar baz"
#         # 'foo' matches, then 'bar' matches (skipping 'qux'), then 'baz' matches
#         "expected": ["foo", "bar", "baz"],
#     },
#     {
#         "first_pass": ["one", "two", "three"],
#         "second_pass": ["one"],
#         # haystack: "one"
#         # Only first segment matches, others fail
#         "expected": ["one", None, None],
#     },
#     {
#         "first_pass": ["a", "a", "a"],
#         "second_pass": ["a", "a", "a"],
#         # haystack: "a a a"
#         # Each 'a' matches progressively through the string
#         "expected": ["a", "a", "a"],
#     },
#     {
#         "first_pass": ["ðə", " kwˈɪk", " bɹˈaʊn"],
#         "second_pass": ["ðə", "kwˈɪk", "bɹˈaʊn"],
#         # haystack: "ðə kwˈɪk bɹˈaʊn"
#         # Segments with leading spaces
#         "expected": ["ðə", " kwˈɪk", " bɹˈaʊn"],
#     },
#     {
#         "first_pass": ["test  ", "case"],
#         "second_pass": ["test", "case"],
#         # haystack: "test case"
#         # 'test  ' (with double space) not in haystack
#         "expected": [None, "case"],
#     },
# ]


# def test_align_second_pass():
#     """Test the align_second_pass function with various cases."""
#     for i, test_case in enumerate(test_cases):
#         result = align_second_pass(test_case["first_pass"], test_case["second_pass"])
#         expected = test_case["expected"]
#         assert result == expected, (
#             f"Test case {i} failed:\n"
#             f"  first_pass: {test_case['first_pass']}\n"
#             f"  second_pass: {test_case['second_pass']}\n"
#             f"  expected: {expected}\n"
#             f"  got: {result}"
#         )
