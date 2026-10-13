"""Dataset conversion and metrics; deliberately independent of MLX."""
import collections
import hashlib
import json
import random
import re
import string
from pathlib import Path

ABSTAIN = "INSUFFICIENT_EVIDENCE"
SYSTEM = ("Answer the question using only the supplied passage. Return a short answer "
          "copied from the passage, without explanation. If the passage does not support "
          "an answer, return exactly INSUFFICIENT_EVIDENCE.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def flatten_squad(payload):
    rows = []
    seen = set()
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if qa["id"] in seen:
                    raise ValueError("Duplicate question ID")
                seen.add(qa["id"])
                impossible = qa.get("is_impossible", False)
                answers = [answer["text"] for answer in qa["answers"]]
                if not impossible and not answers:
                    raise ValueError("Answerable example without reference answers")
                rows.append({"id": qa["id"], "title": article["title"],
                             "context": context, "question": qa["question"],
                             "answers": answers, "unanswerable": impossible})
    return rows


def messages(row, include_answer=False):
    result = [{"role": "system", "content": SYSTEM},
              {"role": "user", "content": f"Passage:\n{row['context']}\n\nQuestion: {row['question']}"}]
    if include_answer:
        answer = ABSTAIN if row["unanswerable"] else row["answers"][0]
        result.append({"role": "assistant", "content": answer})
    return result


def balanced_sample(rows, n, rng):
    pools = [[r for r in rows if r["unanswerable"] == flag] for flag in (False, True)]
    needs = [n - n // 2, n // 2]
    for pool, need in zip(pools, needs):
        if len(pool) < need:
            raise ValueError(f"Insufficient examples: need {need}, have {len(pool)}")
    result = sum((rng.sample(pool, need) for pool, need in zip(pools, needs)), [])
    rng.shuffle(result)
    return result


def prepare(train_payload, dev_payload, sizes, seed, max_chars=1600):
    """Partition articles first; also exclude identical passages across partitions."""
    if any(n <= 0 for n in sizes) or max_chars <= 0:
        raise ValueError("Split sizes and context budget must be positive")
    rng = random.Random(seed)
    train = flatten_squad(train_payload)
    dev = flatten_squad(dev_payload)
    titles = sorted({r["title"] for r in train})
    if len(titles) < 2:
        raise ValueError("Need at least two training articles")
    rng.shuffle(titles)
    cutoff = max(1, min(len(titles)-1, round(len(titles) * .8)))
    train_titles = set(titles[:cutoff])
    valid_titles = set(titles[cutoff:])
    test_pool = [r for r in dev if r["title"] not in train_titles | valid_titles]
    test_contexts = {r["context"] for r in test_pool}
    valid_pool = [r for r in train if r["title"] in valid_titles and r["context"] not in test_contexts]
    valid_contexts = {r["context"] for r in valid_pool}
    train_pool = [r for r in train if r["title"] in train_titles and
                  r["context"] not in test_contexts | valid_contexts]
    pools = [train_pool, valid_pool, test_pool]
    return {name: balanced_sample([r for r in pool if len(r["context"]) <= max_chars], n, rng)
            for name, pool, n in zip(("train", "valid", "test"), pools, sizes)}


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def abstains(text):
    return text.strip() == ABSTAIN


def answer_scores(prediction, answers):
    p = normalize(prediction)
    exact, f1 = 0., 0.
    for answer in answers:
        a = normalize(answer)
        exact = max(exact, float(p == a))
        pt, at = p.split(), a.split()
        common = sum((collections.Counter(pt) & collections.Counter(at)).values())
        score = (2 * common / (len(pt) + len(at))) if pt and at else float(pt == at)
        f1 = max(f1, score)
    return exact, f1


def score_predictions(rows, predictions):
    expected = {r["id"] for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate reference IDs")
    ids = [p["id"] for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Predictions must cover reference IDs exactly once")
    by_id = {p["id"]: p["prediction"] for p in predictions}
    tp = fp = fn = tn = 0
    ems, f1s, details = [], [], []
    total_em = total_f1 = 0.
    for row in rows:
        prediction = by_id[row["id"]]
        abstained = abstains(prediction)
        unanswerable = row["unanswerable"]
        tp += int(abstained and unanswerable)
        fp += int(abstained and not unanswerable)
        fn += int(not abstained and unanswerable)
        tn += int(not abstained and not unanswerable)
        if unanswerable:
            em = f1 = float(abstained)
        else:
            em, f1 = (0., 0.) if abstained else answer_scores(prediction, row["answers"])
            ems.append(em)
            f1s.append(f1)
        total_em += em
        total_f1 += f1
        details.append({**row, "prediction": prediction, "exact_match": em, "token_f1": f1})
    safe = lambda a,b: a / b if b else None
    metrics = {"n": len(rows), "answerable_n": len(ems),
               "answerable_exact_match": safe(sum(ems), len(ems)),
               "answerable_token_f1": safe(sum(f1s), len(f1s)),
               "overall_exact_match": safe(total_em, len(rows)),
               "overall_token_f1": safe(total_f1, len(rows)),
               "abstention_precision": safe(tp, tp+fp),
               "abstention_recall": safe(tp, tp+fn),
               "abstention_confusion": {"tp":tp, "fp":fp, "fn":fn, "tn":tn}}
    return metrics, details
