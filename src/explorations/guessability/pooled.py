"""The guesses from every source pooled per question, on the experiment's own question ids.

Three sources answer the same OneStop questions without the paragraph:

* `"guess"` -- `guess.tsv`, the Prolific guessing data (OneStop questions only here).
* `"pilot"` -- the pilot batches, an earlier version of the questions.
* `"knowqa"` -- KnowQA's no-knowledge regime, the experiment itself.

**The anchor is the experiment's text table** (`EXPERIMENT_TEXT_COMPLETED_PATH`), the version
KnowQA's participants saw. A question is keyed by `question_key`: its `text_id_with_q` without
the level, `batch_article_paragraph_question`. OneStop's two levels share the question and its
answers, and a guesser never sees the paragraph, so a question guessed at either level is one
item here.

**How a guessing file is matched to it.** Each paragraph the file shows is matched to the
experiment's paragraph at the same level, by the overlap of their words; the match has to be
clear (`PARAGRAPH_MIN_OVERLAP`, `PARAGRAPH_MAX_RUNNER_UP`). Within a paragraph the file's
questions are then paired one-to-one with the experiment's, maximising the mean text similarity
of the question and the four answers in stored order (correct first). Both files have exactly
the experiment's questions per paragraph, so every question is paired.

**Every question is paired, however much it changed.** `match_score` (1 = identical after case
and punctuation) and `change` (`CHANGE_LEVELS`) say how far apart the two versions are, and `correct_was_distractor` marks a
question whose correct answer in the experiment is one of the file's wrong answers: there the
file's guesses were scored against a different answer key. Neither leaves anything out; they are
there to be looked at.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.config import columns as Con
from src.config.datasets import EXPERIMENT_TEXT_COMPLETED_PATH
from src.explorations.guessability.data import SEP, load

SOURCES = {
    "guess": "Prolific guessing data (guess.tsv)",
    "pilot": "pilot batches, earlier version",
    "knowqa": "KnowQA, no-knowledge regime",
}
QUESTION_KEY = "question_key"
ANSWER_COLS = ["answer_A", "answer_B", "answer_C", "answer_D"]  # answer_A is the correct one

# A paragraph match must share at least this share of words with its experiment paragraph,
# and no other experiment paragraph may share more than this. The real matches sit at 0.80
# and above, the runners-up at 0.28 and below, so either bound failing means a wrong match.
PARAGRAPH_MIN_OVERLAP = 0.6
PARAGRAPH_MAX_RUNNER_UP = 0.4

# How much a question changed between two versions, from the words changed over the question
# and its four answers, after case and punctuation: none (`identical`); one or two (`1-2 words`);
# at least one of the five texts with more than half its words changed (`rewritten`); anything
# else (`reworded`). The cut-offs are a choice, to tell a small edit from a new text.
CHANGE_LEVELS = ["identical", "1-2 words", "reworded", "rewritten"]
ALMOST_THE_SAME = ("identical", "1-2 words")
SMALL_EDIT_WORDS = 2
REWRITTEN_SHARE = 0.5


def norm(s: str) -> str:
    """Lower case, curly quotes made straight, punctuation dropped, spaces collapsed."""
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _words(s: str) -> set:
    return set(norm(s).split())


def _overlap(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def _words_changed(a: str, b: str) -> tuple[int, int]:
    """Words changed from `a` to `b`, each edit counted on its longer side, and the longer
    text's length in words."""
    x, y = norm(a).split(), norm(b).split()
    ops = SequenceMatcher(None, x, y, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal"), max(len(x), len(y))


def change(texts_a: List[str], texts_b: List[str]) -> tuple[int, float, str]:
    """Words changed over all the texts, the largest share of one text's words changed, and the
    `CHANGE_LEVELS` label."""
    counts = [_words_changed(a, b) for a, b in zip(texts_a, texts_b)]
    total = sum(c for c, _ in counts)
    most = max(c / n for c, n in counts)
    if total == 0:
        level = "identical"
    elif total <= SMALL_EDIT_WORDS:
        level = "1-2 words"
    elif most > REWRITTEN_SHARE:
        level = "rewritten"
    else:
        level = "reworded"
    return total, most, level


def question_key(text_id_with_q: pd.Series) -> pd.Series:
    """`batch_article_level_paragraph_question` -> `batch_article_paragraph_question`."""
    parts = text_id_with_q.str.split("_", expand=True)
    assert parts.shape[1] == 5 and parts[2].isin(["Adv", "Ele"]).all(), "unexpected text_id_with_q"
    return parts[0] + "_" + parts[1] + "_" + parts[3] + "_" + parts[4]


def experiment_texts() -> pd.DataFrame:
    """The experiment's text table, one row per question and level, with `question_key`,
    the paragraph's key (`paragraph_key`) and the answers joined as `options`."""
    t = pd.read_csv(EXPERIMENT_TEXT_COMPLETED_PATH)
    t[QUESTION_KEY] = question_key(t[Con.TEXT_ID_WITH_Q_COLUMN])
    t["level"] = t[Con.TEXT_ID_WITH_Q_COLUMN].str.split("_").str[2]
    t["paragraph_key"] = t[QUESTION_KEY].str.rsplit("_", n=1).str[0]
    t["options"] = t[ANSWER_COLS].agg(SEP.join, axis=1)
    differs = t.groupby(QUESTION_KEY)[["question", "options"]].nunique().gt(1).any(axis=1)
    assert not differs.any(), f"{int(differs.sum())} questions differ between the two levels"
    return t


def _match_paragraphs(guesses: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """Each (level, paragraph text) the file shows -> the experiment's `paragraph_key`."""
    exp = texts.drop_duplicates(["paragraph_key", "level"]).set_index(["level", "paragraph_key"])["paragraph"]
    exp_words = exp.map(_words)
    rows = []
    for (level, text), _ in guesses.groupby(["difficulty", "paragraph"]):
        overlap = exp_words.xs(level).map(lambda w: _overlap(w, _words(text))).sort_values(ascending=False)
        rows.append({"difficulty": level, "paragraph": text, "paragraph_key": overlap.index[0],
                     "paragraph_overlap": overlap.iloc[0], "paragraph_runner_up": overlap.iloc[1]})
    out = pd.DataFrame(rows)
    weak = out[(out["paragraph_overlap"] < PARAGRAPH_MIN_OVERLAP)
               | (out["paragraph_runner_up"] > PARAGRAPH_MAX_RUNNER_UP)]
    assert weak.empty, f"{len(weak)} paragraph(s) have no clear match in the experiment:\n{weak}"
    return out


def match_questions(guesses: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """One row per question of a guessing file: its experiment `question_key` and how alike
    the two versions are. `guesses` is one dataset's guesses (`data.load`), OneStop only."""
    dataset = guesses["dataset"].iloc[0]
    paragraphs = _match_paragraphs(guesses, texts)
    g = guesses.merge(paragraphs[["difficulty", "paragraph", "paragraph_key"]],
                      on=["difficulty", "paragraph"], how="left", validate="many_to_one")
    asked = g.drop_duplicates(["paragraph_key", "question", "options"])[["paragraph_key", "question", "options"]]
    assert not asked.duplicated(["paragraph_key", "question"]).any(), (
        f"{dataset}: a question has two answer sets within one paragraph")
    exp = texts.drop_duplicates(QUESTION_KEY)

    rows: List[Dict] = []
    for par, mine in asked.groupby("paragraph_key"):
        theirs = exp[exp["paragraph_key"] == par].reset_index(drop=True)
        mine = mine.reset_index(drop=True)
        assert len(theirs) == len(mine), (
            f"{dataset}: paragraph {par} has {len(mine)} questions, the experiment {len(theirs)}")
        score = np.array([[np.mean([_similarity(a.question, b.question)]
                                   + [_similarity(x, y) for x, y in zip(a.options.split(SEP), b.options.split(SEP))])
                           for _, b in theirs.iterrows()] for _, a in mine.iterrows()])
        mi, ti = linear_sum_assignment(-score)
        for i, j in zip(mi, ti):
            a, b = mine.loc[i], theirs.loc[j]
            ao, bo = [norm(x) for x in a.options.split(SEP)], [norm(x) for x in b.options.split(SEP)]
            words, most, level = change([a.question, *a.options.split(SEP)], [b.question, *b.options.split(SEP)])
            rows.append({
                "dataset": dataset, "question": a.question, "options": a.options,
                QUESTION_KEY: b[QUESTION_KEY], "match_score": score[i, j],
                "identical": norm(a.question) == norm(b.question) and ao == bo,
                "correct_answer_same": ao[0] == bo[0],
                "correct_was_distractor": bo[0] in ao[1:],
                "words_changed": words, "most_changed_share": most, "change": level,
            })
    out = pd.DataFrame(rows)
    assert not out[QUESTION_KEY].duplicated().any(), f"{dataset}: two questions paired with one experiment question"
    return out


def matched_guesses(dataset: str, texts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One guessing file's OneStop guesses with their `question_key`, and its matching table."""
    g = load(dataset, verbose=False)
    g = g[g["source"] == "Onestop"].copy()
    pairs = match_questions(g, texts)
    g = g.merge(pairs[["question", "options", QUESTION_KEY]], on=["question", "options"],
                how="left", validate="many_to_one")
    assert g[QUESTION_KEY].notna().all(), f"{dataset}: a guess got no question_key"
    return g, pairs


def guess_rows(g: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """The columns every source shares: who guessed, which question, at which level, right or not.

    The same person guessing one question twice (at both levels) would count twice; it does
    not happen in any source, and stops the pooling if it does."""
    out = pd.DataFrame({"dataset": dataset, "participant": g["participant"].astype(str).to_numpy(),
                        QUESTION_KEY: g[QUESTION_KEY].to_numpy(), "difficulty": g["difficulty"].to_numpy(),
                        "is_correct": g["is_correct"].astype(int).to_numpy()})
    twice = out.duplicated(["participant", QUESTION_KEY])
    assert not twice.any(), f"{dataset}: {int(twice.sum())} guesses repeat a person and question"
    return out


def item_table(guesses: pd.DataFrame) -> pd.DataFrame:
    """One row per question: guesses and share right in each source, and pooled over all of
    them (`n_guesses`, `n_correct`, `share_correct`). Pooled means every guess counts once,
    so a source with more guesses per question weighs more."""
    grouped = guesses.groupby([QUESTION_KEY, "dataset"])["is_correct"]
    # A source that never showed the question has no guesses at it: a real zero count, and
    # a share that does not exist (NaN), not a share of zero.
    n = grouped.size().unstack("dataset").reindex(columns=list(SOURCES)).fillna(0).astype(int)
    k = grouped.sum().unstack("dataset").reindex(columns=list(SOURCES))
    out = pd.DataFrame(index=n.index)
    for s in SOURCES:
        out[f"n_{s}"] = n[s]
        out[f"share_{s}"] = k[s] / n[s].where(n[s] > 0)
    out["n_guesses"] = n.sum(axis=1)
    out["n_correct"] = k.sum(axis=1).astype(int)
    out["share_correct"] = out["n_correct"] / out["n_guesses"]
    assert out["n_guesses"].sum() == len(guesses)
    return out.reset_index()


def carry_over(items: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Does a question's guessability survive the pilot's edits?

    Over the questions both the pilot and `guess.tsv` guessed, by how much the pilot's version
    differs from the experiment's (`change`): how many, each source's mean share right, and the
    Spearman correlation of the two shares. `guess.tsv` asks the experiment's own version, so
    the `identical` row is the agreement to expect when nothing changed, noise included; a row
    well below it is a question that became a different item."""
    level = pairs.loc[pairs["dataset"] == "pilot"].set_index(QUESTION_KEY)["change"]
    both = items[(items["n_guess"] > 0) & (items["n_pilot"] > 0)]
    both = both.assign(change=both[QUESTION_KEY].map(level))
    assert both["change"].notna().all(), "a question guessed in the pilot has no pilot match"
    rows = []
    for lvl, sub in [*((lvl, both[both["change"] == lvl]) for lvl in CHANGE_LEVELS), ("all", both)]:
        rows.append({"pilot_change": lvl, "questions": len(sub),
                     "mean_share_pilot": sub["share_pilot"].mean(), "mean_share_guess": sub["share_guess"].mean(),
                     "spearman_r": sub["share_pilot"].corr(sub["share_guess"], method="spearman")})
    return pd.DataFrame(rows)
