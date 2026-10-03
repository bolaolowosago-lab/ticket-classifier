"""TF-IDF + logistic regression baseline."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from data import load_split
from evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to the ticket csv")
    p.add_argument("--text-col", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--no-cv", action="store_true", help="skip 5-fold cross-validation")
    args = p.parse_args()

    train, test = load_split(args.data, args.text_col, args.label_col)
    labels = sorted(train[args.label_col].unique())
    print(f"train={len(train)} test={len(test)} classes={len(labels)}")

    # baseline to beat: always predict the most common class
    majority = train[args.label_col].mode()[0]
    maj_acc = (test[args.label_col] == majority).mean()
    print(f"majority-class baseline accuracy: {maj_acc:.4f}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    if not args.no_cv:
        cv = cross_val_score(pipe, train[args.text_col], train[args.label_col],
                             cv=5, scoring="f1_macro", n_jobs=-1)
        print(f"5-fold CV macro F1 (train only): {cv.mean():.4f} +/- {cv.std():.4f}")

    pipe.fit(train[args.text_col], train[args.label_col])
    pred = pipe.predict(test[args.text_col])
    conf = pipe.predict_proba(test[args.text_col]).max(axis=1)

    evaluate(test[args.text_col], test[args.label_col], pred, conf, labels, "baseline")

    joblib.dump({"pipeline": pipe, "labels": labels}, ROOT / "models" / "baseline.joblib", compress=3)
    print("saved models/baseline.joblib")

    # which words push each class the most (nice for the README)
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    print("\ntop words per class:")
    for i, lab in enumerate(clf.classes_):
        top = names[np.argsort(clf.coef_[i])[-6:][::-1]]
        print(f"  {lab}: {', '.join(top)}")


if __name__ == "__main__":
    main()
