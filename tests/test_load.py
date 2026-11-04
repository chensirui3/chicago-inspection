"""Tests for the portal client: retries and paging by row identifier (no network)."""

import pytest
import requests

from inspections import load


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


class FakeSession:
    """Returns the queued results in order (an exception instance is raised)."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(load.time, "sleep", lambda s: None)


def test_timeouts_are_retried_and_pages_continue_after_the_last_row_id(monkeypatch):
    page1 = FakeResponse('":id","sr_number"\n"row-a","SR1"\n"row-b","SR2"\n')
    page2 = FakeResponse('":id","sr_number"\n"row-c","SR3"\n')
    session = FakeSession([requests.ReadTimeout("slow portal"), page1, page2])
    monkeypatch.setattr(load, "_session", lambda: session)

    df = load.socrata_frame("service_requests_311", select="sr_number",
                            where="sr_type = 'X'", page_size=2)

    assert df["sr_number"].tolist() == ["SR1", "SR2", "SR3"]
    assert ":id" not in df.columns
    assert len(session.calls) == 3  # one timeout, then two pages
    first, second = session.calls[1], session.calls[2]
    assert first["$where"] == "(sr_type = 'X')" and first["$order"] == ":id"
    assert second["$where"] == "(sr_type = 'X') AND :id > 'row-b'"
    assert "$offset" not in second


def test_server_errors_are_retried_then_raised(monkeypatch):
    session = FakeSession([FakeResponse("busy", 503)] * 3)
    monkeypatch.setattr(load, "_session", lambda: session)
    with pytest.raises(RuntimeError, match="503"):
        load.socrata_frame("crimes", select="id", retries=3)
    assert len(session.calls) == 3


def test_a_page_cut_off_mid_response_is_retried(monkeypatch):
    cut_off = requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")
    page = FakeResponse('":id","id"\n"row-a","1"\n')
    session = FakeSession([cut_off, page])
    monkeypatch.setattr(load, "_session", lambda: session)

    df = load.socrata_frame("crimes", select="id")

    assert df["id"].tolist() == ["1"]
    assert len(session.calls) == 2


def test_repeated_timeouts_give_up_after_the_retry_budget(monkeypatch):
    session = FakeSession([requests.ConnectTimeout("down")] * 2)
    monkeypatch.setattr(load, "_session", lambda: session)
    with pytest.raises(requests.ConnectTimeout):
        load.socrata_frame("crimes", select="id", retries=2)
