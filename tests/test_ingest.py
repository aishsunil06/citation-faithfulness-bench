import pytest

from cfbench.ingest import clean_text, dedupe, ingest_rows, to_document
from cfbench.retrieval import Document


def test_clean_text_strips_navigation_and_reference_markers():
    raw = "Jump to content\nMain menu\n\nRevenue rose 12 percent in 2024.[3]\n\nSee also\nReferences\nExternal links"
    out = clean_text(raw)
    assert "Revenue rose 12 percent in 2024." in out
    assert "Jump to content" not in out
    assert "External links" not in out
    assert "[3]" not in out


def test_clean_text_drops_short_lines_and_collapses_whitespace():
    out = clean_text("OK\n\n\nA real sentence that is certainly long enough to keep.\n\nEdit\n")
    assert out == "A real sentence that is certainly long enough to keep."


def test_dedupe_drops_near_identical_documents_keeping_the_first():
    a = Document(url="u1", text="shared body text about storage capacity " * 20)
    b = Document(url="u2", text="shared body text about storage capacity " * 20 + "tail")
    kept, dropped = dedupe([a, b])
    assert [d.url for d in kept] == ["u1"]
    assert dropped == ["u2"]


def test_dedupe_keeps_genuinely_different_documents():
    a = Document(url="u1", text="battery storage capacity megawatt hours " * 20)
    b = Document(url="u2", text="charter aviation fleet regional hubs aircraft " * 20)
    kept, _dropped = dedupe([a, b])
    assert len(kept) == 2


def test_to_document_rejects_a_page_too_short_to_chunk():
    with pytest.raises(ValueError):
        to_document("u", "t", "too short")


# ---------------------------------------------------------------------------
# Review Focus 1: boilerplate stripping on realistic captured text.
# ---------------------------------------------------------------------------

# Shaped like a real copy-paste of an English Wikipedia article: the chrome
# above the prose, the inline citation markers, and the whole footer apparatus
# below it. If any of this survives, a retrieved "evidence" chunk can be a
# list of navigation links.
_CAPTURED_WIKIPEDIA_PAGE = """Jump to content
Main menu
Navigation
Main page
Contents
Current events
Random article

Hoover Dam

From Wikipedia, the free encyclopedia

Hoover Dam is a concrete arch-gravity dam in the Black Canyon of the Colorado
River, on the border between the U.S. states of Nevada and Arizona.[1] It was
constructed between 1931 and 1936 during the Great Depression and was dedicated
on September 30, 1935, by President Franklin D. Roosevelt.[2] Its construction
was the result of a massive effort involving thousands of workers, and cost
over one hundred lives.[citation needed]

The dam impounds Lake Mead, the largest reservoir in the United States by
volume when full, holding approximately 35.2 cubic kilometres of water.[4] The
generators of the dam's power plant have a nameplate capacity of 2080 megawatts.

Contents
1 Background
2 Construction
3 See also

See also
List of dams in the Colorado River system
References
Further reading
External links
Retrieved from "https://en.wikipedia.org/wiki/Hoover_Dam"
Categories: Dams in Nevada
Edit
"""


def test_clean_text_on_captured_page_keeps_prose_and_drops_all_chrome():
    out = clean_text(_CAPTURED_WIKIPEDIA_PAGE)

    # The facts a question would be authored against survive intact.
    assert "concrete arch-gravity dam in the Black Canyon" in out
    assert "nameplate capacity of 2080 megawatts" in out
    assert "September 30, 1935" in out

    for chrome in (
        "Jump to content",
        "Main menu",
        "Navigation",
        "Contents",
        "Current events",
        "See also",
        "References",
        "Further reading",
        "External links",
        "Retrieved from",
        "Categories:",
        "Edit",
    ):
        assert chrome not in out, f"boilerplate survived cleaning: {chrome!r}"


def test_clean_text_strips_every_inline_reference_marker():
    out = clean_text(_CAPTURED_WIKIPEDIA_PAGE)
    assert "[" not in out and "]" not in out
    assert "citation needed" not in out
    # Stripping a marker must not weld the neighbouring words together.
    assert "Arizona. It was" in out


def test_clean_text_rejoins_lines_wrapped_mid_sentence():
    # Captured prose wraps at the terminal width; a chunker that saw the raw
    # newlines would be fine, but a judge reading the snippet should not.
    out = clean_text(_CAPTURED_WIKIPEDIA_PAGE)
    assert "Colorado River" in out
    assert "\n" not in out


def test_clean_text_drops_a_long_boilerplate_line_not_just_a_short_one():
    # "Retrieved from <url>" is well over the length floor, so length alone
    # cannot catch it; the blocklist has to match on the line's prefix.
    raw = 'Retrieved from "https://en.wikipedia.org/wiki/Hoover_Dam_and_the_Colorado"'
    assert clean_text(raw) == ""


def test_clean_text_keeps_a_sentence_that_merely_mentions_a_blocklist_word():
    raw = "The contents of the reservoir fell by eleven percent during the drought."
    assert clean_text(raw) == raw


def test_to_document_accepts_the_captured_page_and_carries_metadata():
    doc = to_document(
        "https://en.wikipedia.org/wiki/Hoover_Dam", "Hoover Dam", _CAPTURED_WIKIPEDIA_PAGE
    )
    assert doc.url == "https://en.wikipedia.org/wiki/Hoover_Dam"
    assert doc.title == "Hoover Dam"
    assert len(doc.text.split()) >= 60
    assert "Jump to content" not in doc.text


def test_to_document_rejects_a_page_that_is_only_boilerplate():
    # The dangerous case: plenty of raw characters, no prose at all. Rejecting
    # on raw length would let this through.
    raw = "\n".join(["Jump to content", "Main menu", "See also", "References"] * 20)
    with pytest.raises(ValueError):
        to_document("u", "t", raw)


# ---------------------------------------------------------------------------
# Review Focus 3: near-duplicate detection.
# ---------------------------------------------------------------------------


def test_dedupe_drops_a_summary_page_that_restates_its_parent_article():
    body = (
        "Hoover Dam impounds Lake Mead, the largest reservoir in the United States by "
        "volume, holding approximately 35.2 cubic kilometres of water. The concrete "
        "arch-gravity dam spans Black Canyon on the Colorado River between Nevada and "
        "Arizona, and the generators of its power plant carry a nameplate capacity of "
        "2080 megawatts. Construction ran from 1931 to 1936 and the structure was "
        "dedicated by President Franklin Roosevelt in September 1935. "
    )
    parent = Document(url="u1", text=body + "Six companies formed a consortium to bid.")
    summary = Document(url="u2", text=body + "A brief overview follows below.")
    kept, dropped = dedupe([parent, summary])
    assert [d.url for d in kept] == ["u1"]
    assert dropped == ["u2"]


def test_dedupe_drops_every_later_copy_not_just_the_second():
    text = "wind turbine blade manufacturing plant in Pueblo Colorado " * 20
    docs = [Document(url=f"u{i}", text=text) for i in range(4)]
    kept, dropped = dedupe(docs)
    assert [d.url for d in kept] == ["u0"]
    assert dropped == ["u1", "u2", "u3"]


def test_dedupe_compares_against_kept_documents_not_dropped_ones():
    # B overlaps A and is dropped; C overlaps B but not A. Comparing C against
    # the already-dropped B would delete a page no kept document duplicates.
    a = Document(url="u1", text="alpha beta gamma delta epsilon zeta eta theta iota kappa")
    b = Document(
        url="u2",
        text="alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron rho sigma tau phi chi",
    )
    c = Document(url="u3", text="lambda mu nu xi omicron rho sigma tau phi chi")
    kept, dropped = dedupe([a, b, c], threshold=0.4)
    assert [d.url for d in kept] == ["u1", "u3"]
    assert dropped == ["u2"]


def test_dedupe_threshold_is_configurable():
    a = Document(url="u1", text="one two three four five six seven eight nine ten")
    b = Document(url="u2", text="one two three four five alpha beta gamma delta epsilon")
    # Jaccard here is 5/15 = 0.33: kept at the default, dropped at 0.3.
    assert len(dedupe([a, b])[0]) == 2
    assert [d.url for d in dedupe([a, b], threshold=0.3)[0]] == ["u1"]


def test_dedupe_drops_an_exact_duplicate_url():
    text = "geothermal plant near Steamboat Springs produced 24 megawatts " * 20
    kept, dropped = dedupe([Document(url="u1", text=text), Document(url="u1", text=text)])
    assert len(kept) == 1
    assert dropped == ["u1"]


def test_dedupe_on_an_empty_corpus_returns_nothing():
    assert dedupe([]) == ([], [])


def test_dedupe_does_not_drop_a_document_with_no_usable_tokens():
    # An all-stopword document has an empty token set; Jaccard against another
    # empty set is undefined, and treating it as 1.0 would silently delete
    # unrelated pages.
    a = Document(url="u1", text="the and of to")
    b = Document(url="u2", text="in it is as")
    kept, _dropped = dedupe([a, b])
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# ingest_rows: the whole path, as the CLI will use it.
# ---------------------------------------------------------------------------


def _long_row(url: str, filler: str) -> dict:
    return {"url": url, "title": url, "raw": f"{filler} " * 80}


def test_ingest_rows_cleans_dedupes_and_reports_dropped_urls():
    rows = [
        _long_row("u1", "lithium refinery throughput tonnes annually"),
        _long_row("u2", "lithium refinery throughput tonnes annually"),
        _long_row("u3", "regional airline turboprop fleet schedules expanded"),
    ]
    docs, dropped = ingest_rows(rows)
    assert [d.url for d in docs] == ["u1", "u3"]
    assert dropped == ["u2"]


def test_ingest_rows_skips_a_too_short_row_instead_of_raising():
    rows = [
        _long_row("u1", "copper smelter emissions permit renewed"),
        {"url": "u2", "title": "t", "raw": "too short"},
    ]
    docs, dropped = ingest_rows(rows)
    assert [d.url for d in docs] == ["u1"]
    assert dropped == ["u2"]


def test_ingest_rows_tolerates_a_missing_title():
    docs, _dropped = ingest_rows([{"url": "u1", "raw": "tidal barrage output rose steadily " * 30}])
    assert docs[0].title == ""


def test_ingest_rows_on_no_rows_returns_nothing():
    assert ingest_rows([]) == ([], [])
