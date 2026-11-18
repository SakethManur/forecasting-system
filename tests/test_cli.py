"""Tests for the training command-line entry point."""

from __future__ import annotations

import sys

import training.train as cli


def test_cli_passes_path_and_models_to_the_pipeline(monkeypatch, capsys):
    calls = {}

    def fake_train_all_states(excel_path, model_names):
        calls.update(excel_path=excel_path, model_names=model_names)
        return {"states": {"Alpha": {"best_model": "sarima"}}}

    monkeypatch.setattr(cli, "train_all_states", fake_train_all_states)
    monkeypatch.setattr(
        sys, "argv", ["train", "--excel-path", "data/x.xlsx", "--models", "sarima", "xgboost"]
    )

    cli.main()

    assert calls == {"excel_path": "data/x.xlsx", "model_names": ["sarima", "xgboost"]}
    assert '"best_model": "sarima"' in capsys.readouterr().out


def test_cli_defaults_to_every_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train"])
    args = cli.parse_args()
    assert args.excel_path is None
    assert args.models is None
