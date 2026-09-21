from docingest.clean import clean_page, strip_repeated_lines


def test_ligatures_and_odd_spaces_are_normalised():
    assert clean_page("ofﬁce hours") == "office hours"


def test_control_characters_are_dropped():
    assert clean_page("a\x00b\x0cc") == "abc"


def test_hyphenated_line_breaks_are_rejoined():
    assert clean_page("retriev-\nal works") == "retrieval works"


def test_a_real_hyphen_before_a_capital_is_kept():
    assert clean_page("state-of-\nThe Art") == "state-of- The Art"


def test_hard_wrapped_lines_become_one_paragraph():
    assert clean_page("the quick brown\nfox jumps over\nthe lazy dog") == "the quick brown fox jumps over the lazy dog"


def test_sentence_ends_and_blank_lines_stay_as_breaks():
    got = clean_page("First sentence.\nSecond sentence.\n\nNew paragraph here")
    assert got == "First sentence.\nSecond sentence.\n\nNew paragraph here"


def test_runaway_blank_lines_are_collapsed():
    assert clean_page("a\n\n\n\n\nb") == "a\n\nb"


def pages_with_furniture(n):
    return [f"ACME Confidential\nreal content number {i}\nPage {i + 1}" for i in range(n)]


def test_running_headers_and_page_numbers_are_removed():
    out = strip_repeated_lines(pages_with_furniture(5))
    assert out[2] == "real content number 2"


def test_lines_that_appear_on_only_a_few_pages_are_kept():
    pages = pages_with_furniture(6)
    pages[0] += "\nSee the appendix"
    assert "See the appendix" in strip_repeated_lines(pages)[0]


def test_too_few_pages_to_judge_leaves_everything_alone():
    pages = ["Header\nbody one", "Header\nbody two"]
    assert strip_repeated_lines(pages) == pages


def test_long_repeated_lines_are_treated_as_content_not_furniture():
    long_line = "This sentence is long enough that it is surely body text and not a running header " * 2
    pages = [f"{long_line}\nunique {i}" for i in range(4)]
    assert all(long_line.strip() in p for p in strip_repeated_lines(pages))
