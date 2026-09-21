import json

import pytest
from conftest import CountingEmbed, fake_embed, scanned_pdf, text_pdf

from docingest.pipeline import Ingester


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "docs"
    f.mkdir()
    (f / "pets.txt").write_text("cats purr when they are happy and dogs bark when they are excited")
    (f / "cars.txt").write_text("engines burn fuel to turn the wheels of a car")
    return f


def ingester(store, tmp_path, embed=None, **kw):
    return Ingester(store, embed or fake_embed, tmp_path / "state.json", ocr=lambda png: "ocr text here for testing", **kw)


def find(store, query):
    return [h.source for h in store.query(fake_embed([query], "query")[0], k=5)]


def test_first_sync_adds_everything(folder, store, tmp_path):
    r = ingester(store, tmp_path).sync(folder)
    assert sorted(r.added) == ["cars.txt", "pets.txt"]
    assert r.chunks == 2 and store.count() == 2


def test_second_sync_touches_nothing(folder, store, tmp_path):
    embed = CountingEmbed()
    ing = ingester(store, tmp_path, embed)
    ing.sync(folder)
    calls = embed.calls
    r = ing.sync(folder)
    assert sorted(r.unchanged) == ["cars.txt", "pets.txt"] and not r.added and not r.updated
    assert embed.calls == calls


def test_changed_file_is_reprocessed_and_its_old_text_disappears(folder, store, tmp_path):
    ing = ingester(store, tmp_path)
    ing.sync(folder)
    (folder / "pets.txt").write_text("hamsters run on wheels all night")
    r = ing.sync(folder)
    assert r.updated == ["pets.txt"] and r.unchanged == ["cars.txt"]
    assert store.count() == 2  # replaced, not appended
    hits = store.query(fake_embed(["cats purr"], "query")[0], k=5)
    assert all("cats" not in h.text for h in hits)


def test_shortened_file_leaves_no_stale_tail(folder, store, tmp_path):
    (folder / "long.txt").write_text("alpha beta gamma. " * 200)
    ing = ingester(store, tmp_path, size=300, overlap=30)
    ing.sync(folder)
    assert store.count() > 3  # the long file really did span several chunks
    (folder / "long.txt").write_text("alpha beta gamma.")
    ing.sync(folder)
    assert store.count() == 3  # pets.txt, cars.txt and the one short chunk, nothing left over


def test_deleted_file_is_removed_from_the_index(folder, store, tmp_path):
    ing = ingester(store, tmp_path)
    ing.sync(folder)
    (folder / "cars.txt").unlink()
    r = ing.sync(folder)
    assert r.removed == ["cars.txt"]
    assert find(store, "engines burn fuel") == ["pets.txt"]


def test_state_survives_a_restart(folder, store, tmp_path):
    ingester(store, tmp_path).sync(folder)
    r = ingester(store, tmp_path).sync(folder)
    assert len(r.unchanged) == 2


def test_changing_chunk_settings_reprocesses_everything(folder, store, tmp_path):
    ingester(store, tmp_path, size=800).sync(folder)
    r = ingester(store, tmp_path, size=400).sync(folder)
    assert len(r.added) == 2 and not r.unchanged


def test_one_bad_file_does_not_stop_the_others(folder, store, tmp_path):
    (folder / "broken.pdf").write_bytes(b"this is not a pdf")
    r = ingester(store, tmp_path).sync(folder)
    assert "broken.pdf" in r.failed
    assert sorted(r.added) == ["cars.txt", "pets.txt"]
    # and the broken file is retried next time instead of being remembered as done
    assert "broken.pdf" in ingester(store, tmp_path).sync(folder).failed


def test_an_interrupted_run_resumes_where_it_stopped(folder, store, tmp_path):
    def dies_on_pets(texts, kind="document"):
        if any("cats" in t for t in texts):
            raise KeyboardInterrupt
        return fake_embed(texts, kind)

    # files go in sorted order, so cars.txt is done before pets.txt blows up
    with pytest.raises(KeyboardInterrupt):
        ingester(store, tmp_path, dies_on_pets).sync(folder)
    saved = json.loads((tmp_path / "state.json").read_text())["files"]
    assert list(saved) == ["cars.txt"]  # the finished file was already recorded

    r = ingester(store, tmp_path).sync(folder)
    assert r.added == ["pets.txt"] and r.unchanged == ["cars.txt"]


def test_subfolders_are_indexed_with_their_relative_path(folder, store, tmp_path):
    (folder / "2024").mkdir()
    (folder / "2024" / "report.txt").write_text("annual results were strong")
    ingester(store, tmp_path).sync(folder)
    assert "2024/report.txt" in find(store, "annual results")


def test_scanned_pages_are_counted_and_indexed(tmp_path, store):
    docs = tmp_path / "scans"
    docs.mkdir()
    scanned_pdf(docs / "scan.pdf", "hello")
    r = ingester(store, tmp_path).sync(docs)
    assert (r.ocr_pages, r.chunks) == (1, 1)
    assert find(store, "ocr text here") == ["scan.pdf"]


def test_pdf_chunks_carry_page_numbers_for_citations(tmp_path, store):
    docs = tmp_path / "pdfs"
    docs.mkdir()
    text_pdf(docs / "paper.pdf", ["Introduction to whales and their songs in the deep ocean.", "Methods used for tagging dolphins near the coast."])
    ingester(store, tmp_path).sync(docs)
    hit = store.query(fake_embed(["dolphins tagging"], "query")[0], k=1)[0]
    assert (hit.source, hit.page) == ("paper.pdf", 2)
