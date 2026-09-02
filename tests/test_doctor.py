"""`sp500lab doctor` aggregates every check into one verdict. Offline: each stage is
stubbed, because the stages have their own tests and the doctor's job is the plumbing -
run everything, stop for nothing, fail if anything failed, and say why."""

from __future__ import annotations

import pytest

from sp500lab import doctor


def _ok():
    return True, ["fine"]


def _bad():
    return False, ["broken thing"]


def test_a_stage_that_raises_is_a_failure_with_the_reason():
    s = doctor._stage("boom", lambda: 1 / 0)
    assert s.passed is False and "ZeroDivisionError" in s.lines[0]
    assert s.seconds >= 0


def test_a_stage_reports_its_lines():
    s = doctor._stage("x", _bad)
    assert s.line().startswith("[FAIL] x") and "broken thing" in s.line()
    assert doctor._stage("y", _ok).line().startswith("[PASS] y")


def test_run_never_stops_early_and_orders_cheapest_first(monkeypatch):
    calls: list[str] = []

    def rec(name, ret=(True, [])):
        def fn(*a, **k):
            calls.append(name)
            return ret
        return fn

    monkeypatch.setattr(doctor, "stage_bronze", rec("bronze", (False, ["corrupt"])))
    monkeypatch.setattr(doctor, "stage_silver", rec("silver"))
    monkeypatch.setattr(doctor, "stage_engine", rec("engine"))
    monkeypatch.setattr(doctor, "stage_legs", rec("legs"))
    monkeypatch.setattr(doctor, "stage_features", rec("features"))
    monkeypatch.setattr(doctor, "stage_roster", rec("roster"))
    import sp500lab.backtest.panel as panel_mod
    monkeypatch.setattr(panel_mod, "build_panel", lambda: object())

    stages = doctor.run(roster=True)
    assert calls == ["bronze", "silver", "engine", "legs", "features", "roster"]
    assert [s.passed for s in stages] == [False, True, True, True, True, True]
    text = doctor.report(stages)
    assert "1 stage(s) failed: bronze" in text and "corrupt" in text


def test_fast_skips_the_two_slow_stages(monkeypatch):
    calls: list[str] = []
    for name in ("bronze", "silver", "engine", "legs", "features", "roster"):
        monkeypatch.setattr(doctor, f"stage_{name}",
                            lambda *a, _n=name, **k: calls.append(_n) or (True, []))
    import sp500lab.backtest.panel as panel_mod
    monkeypatch.setattr(panel_mod, "build_panel", lambda: object())
    doctor.run(fast=True)
    assert calls == ["silver", "engine", "legs"]


def test_an_unbuildable_panel_is_reported_and_the_rest_is_skipped(monkeypatch):
    monkeypatch.setattr(doctor, "stage_silver", lambda **k: (True, []))
    import sp500lab.backtest.panel as panel_mod

    def boom():
        raise FileNotFoundError("no silver")
    monkeypatch.setattr(panel_mod, "build_panel", boom)
    stages = doctor.run(fast=True)
    assert [s.name for s in stages] == ["silver", "panel"]
    assert stages[-1].passed is False and "no silver" in stages[-1].lines[0]


def test_report_says_when_everything_passed_and_hints_at_the_roster():
    ok = [doctor.Stage("silver", True, 0.1), doctor.Stage("engine", True, 0.1)]
    assert "Add --roster" in doctor.report(ok)
    ok.append(doctor.Stage("roster", True, 0.1))
    assert "every strategy honours the contract" in doctor.report(ok)


def test_cli_exit_code_follows_the_verdict(monkeypatch):
    from sp500lab.cli import main
    monkeypatch.setattr(doctor, "run", lambda **k: [doctor.Stage("x", True, 0.0)])
    assert main(["doctor", "--fast"]) == 0
    monkeypatch.setattr(doctor, "run", lambda **k: [doctor.Stage("x", False, 0.0)])
    assert main(["doctor", "--fast"]) == 1


@pytest.mark.parametrize("argv", [["doctor"], ["doctor", "--fast", "--roster"],
                                  ["backtest", "accept", "--strategies", "all"]])
def test_the_new_flags_parse(argv):
    from sp500lab.cli import build_parser
    args = build_parser().parse_args(argv)
    assert callable(args.func)
