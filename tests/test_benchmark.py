import json
import tempfile
import unittest
from pathlib import Path
from finetuning_lab.benchmark import validate_models

class BenchmarkTests(unittest.TestCase):
    def test_sources_quantization_and_tokenizer_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/m for m in ('lora','qlora')]
            for i,p in enumerate(paths):
                p.mkdir()
                (p/'source.json').write_text(json.dumps({'repo':'q','revision':'r','quantized':bool(i)}))
                config={'model_type':'qwen2','hidden_size':32}
                if i: config['quantization']={'bits':4}
                (p/'config.json').write_text(json.dumps(config))
                for name in ('tokenizer.json','tokenizer_config.json'): (p/name).write_text('{}')
            validate_models(paths)
            (paths[1]/'tokenizer.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Tokenizer'):
                validate_models(paths)
            (paths[1]/'tokenizer.json').write_text('{}')
            (paths[1]/'source.json').write_text(json.dumps({'repo':'q','revision':'other','quantized':True}))
            with self.assertRaisesRegex(ValueError,'sources'):
                validate_models(paths)
