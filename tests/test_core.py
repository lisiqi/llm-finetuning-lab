import unittest
from finetuning_lab.core import answer_scores, score_predictions, prepare, messages

class MetricsTests(unittest.TestCase):
    def test_normalization_and_multiple_references(self):
        self.assertEqual(answer_scores("The Paris!", ["London", "Paris"]), (1., 1.))
        self.assertEqual(answer_scores("red", ["red blue"]), (0., 2/3))

    def test_abstention_confusion_and_no_answer_in_prompt(self):
        rows = [{"id":str(i),"title":"a","question":"q","context":"c",
                 "answers":["Paris"] if i < 2 else [], "unanswerable":i >= 2} for i in range(4)]
        preds = [{"id":str(i),"prediction":v} for i,v in enumerate(
            ["Paris", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE", "London"])]
        m,_ = score_predictions(rows,preds)
        self.assertEqual(m["abstention_confusion"], {"tp":1,"fp":1,"fn":1,"tn":1})
        self.assertEqual(m["abstention_precision"], .5)
        self.assertEqual(m["overall_exact_match"], .5)
        self.assertEqual(len(messages(rows[0])), 2)
        self.assertNotIn("Paris", str(messages(rows[0])))
        with self.assertRaises(ValueError):
            score_predictions(rows, preds + [preds[0]])

    def test_marker_is_exact_and_undefined_precision(self):
        row = {"id":"x", "answers":[], "unanswerable":True}
        metrics,_ = score_predictions([row], [{"id":"x", "prediction":"I cannot answer"}])
        self.assertIsNone(metrics["abstention_precision"])
        self.assertEqual(metrics["abstention_recall"], 0.)
        with self.assertRaises(ValueError):
            score_predictions([row], [])

    def test_article_split_is_reproducible_and_disjoint(self):
        def payload(prefix, count):
            return {"data":[{"title":f"{prefix}{i}", "paragraphs":[
                {"context":f"{prefix} passage {i}", "qas":[
                    {"id":f"{prefix}{i}-{j}", "question":"Q", "is_impossible":bool(j%2),
                     "answers":[] if j%2 else [{"text":"passage"}]} for j in range(10)]}
                ]} for i in range(count)]}
        a,b = payload("train",10),payload("dev",3)
        groups = prepare(a,b,(8,4,4),42)
        self.assertEqual(groups, prepare(a,b,(8,4,4),42))
        sets = [{row["title"] for row in rows} for rows in groups.values()]
        for i in range(3):
            for j in range(i): self.assertFalse(sets[i] & sets[j])
        for rows in groups.values():
            self.assertEqual(sum(r["unanswerable"] for r in rows), len(rows)//2)

if __name__ == "__main__": unittest.main()
