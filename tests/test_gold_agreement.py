import pandas as pd

from riskpulse.evaluation import nlp_eval


def test_author_labels_override_and_agreement(tmp_path, monkeypatch):
    test = nlp_eval.gold_test()
    first = pd.read_csv(nlp_eval.LABELED / "gold_labels.csv").set_index("gold_id")
    ids = test["gold_id"].head(4).tolist()
    check = first.loc[ids, ["event_type", "sentiment"]].copy()
    flip = {"negative": "positive", "positive": "negative", "neutral": "negative"}
    check.iloc[0, check.columns.get_loc("sentiment")] = flip[check.iloc[0]["sentiment"]]   # one disagreement
    path = tmp_path / "gold_labels_author_check.csv"
    check.reset_index().to_csv(path, index=False)
    monkeypatch.setattr(nlp_eval, "AUTHOR_CHECK", path)

    out = nlp_eval.annotator_agreement()
    assert out["n"] == 4 and out["event"]["agreement"] == 1.0 and out["event"]["changed_to_author_label"] == 0
    assert out["sentiment"]["agreement"] == 0.75 and out["sentiment"]["changed_to_author_label"] == 1
    adjudicated = nlp_eval.gold_test().set_index("gold_id")
    assert adjudicated.at[ids[0], "sentiment"] == check.iloc[0]["sentiment"]       # the author's label wins
    assert len(adjudicated) == 200
