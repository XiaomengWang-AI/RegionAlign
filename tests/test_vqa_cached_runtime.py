"""The cached-feature VQA runtime against the complete multimodal model, on tiny random VLMs.

Each family builds a real Hugging Face VLM with random weights (Qwen2-VL always;
Qwen2.5-VL and Idefics3/SmolVLM when the installed transformers has them), wraps
its language model as the clean loader does, quantizes it to INT8 and compares
``CachedVQARegionRuntime`` with the complete-model methods it inherits.
"""
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
torch.set_num_threads(1)
import transformers

from regionalign.experiments import vqa as D
from regionalign.harness import core as P, region as R
from regionalign.runtimes import vqa as V

FULL = V.HFVQARegionRuntime


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(ch) % 50) for ch in text]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(map(str, tokens))


class StubProcessor:
    """Chat template that continues the generation prompt with ``lead`` + answer."""

    def __init__(self, lead=""):
        self.tokenizer, self.lead = StubTokenizer(), lead

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "<user>Q<end>\n<assistant>"
        if messages[-1]["role"] == "assistant":
            text += self.lead + messages[-1]["content"][0]["text"] + "<end>"
        return text


def randomise(model):
    for p in model.parameters():
        if p.dim() == 1:
            p.data.uniform_(0.5, 1.5)
    model.generation_config.eos_token_id = 1
    model.generation_config.pad_token_id = 0
    return model.eval().requires_grad_(False)


def qwen2_vl():
    torch.manual_seed(0)
    config = transformers.Qwen2VLConfig(
        vision_config=dict(depth=1, embed_dim=32, hidden_size=64, num_heads=2, mlp_ratio=2,
                           in_channels=3, patch_size=14, spatial_merge_size=2, temporal_patch_size=2),
        vocab_size=70, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        image_token_id=65, video_token_id=66, vision_start_token_id=67, vision_end_token_id=68,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}, tie_word_embeddings=False)
    return randomise(transformers.Qwen2VLForConditionalGeneration(config))


def qwen2_5_vl():
    torch.manual_seed(0)
    config = transformers.Qwen2_5_VLConfig(
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=56,
                           out_hidden_size=64, fullatt_block_indexes=[1], tokens_per_second=2),
        vocab_size=70, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        image_token_id=65, video_token_id=66, vision_start_token_id=67, vision_end_token_id=68,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}, tie_word_embeddings=True)
    return randomise(transformers.Qwen2_5_VLForConditionalGeneration(config))


def idefics3():
    torch.manual_seed(0)
    config = transformers.Idefics3Config(
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, image_size=28, patch_size=14),
        text_config=dict(model_type="llama", vocab_size=70, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=256, pad_token_id=0, tie_word_embeddings=False),
        image_token_id=65, scale_factor=2, pad_token_id=0)
    return randomise(transformers.Idefics3ForConditionalGeneration(config))


def language_of(model):
    """The clean loader's language wrapper (``model`` decoder + ``lm_head``)."""
    family = model.config.model_type
    language = torch.nn.Module()
    language.add_module("model", model.model.text_model if family == "idefics3" else model.model)
    language.add_module("lm_head", model.lm_head)
    language.config = model.config.text_config if family == "idefics3" else model.config
    return language


def prompt_inputs(family, index):
    g = torch.Generator().manual_seed(100 + index)
    text = torch.randint(3, 60, (4,), generator=g).tolist()
    if family == "idefics3":
        ids = [2, *text[:2], 65, 65, *text[2:]]              # two 28x28 tiles, one token each
        pixel = torch.randn(1, 2, 3, 28, 28, generator=g)
        ids = torch.tensor([ids])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": pixel,
                "pixel_attention_mask": torch.ones(1, 2, 28, 28, dtype=torch.bool)}
    ids = torch.tensor([[2, text[0], 67, 65, 65, 65, 65, 68, *text[1:]]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(16, 3 * 2 * 14 * 14, generator=g),
            "image_grid_thw": torch.tensor([[1, 4, 4]])}


def runtime_for(build, key, lead="", n=3, cls=V.CachedVQARegionRuntime):
    model = build()
    language = language_of(model)
    P.quantize(language, "int8")
    audit = {**P.quantization_audit(language), "visual_parameters_unchanged": True}
    rows = [{"sid": f"r{i}", "reference": "ab" if i else "cat", "answers": ["cat"] * 10,
             "question": "q?", "image": {"bytes": f"image {i}".encode(), "path": None}}
            for i in range(n)]
    rt = cls(model, StubProcessor(lead), language, {"search": rows}, key, "textvqa", "cpu",
             quantization=audit)
    rt.max_new_tokens = 3
    for index, row in enumerate(rt.rows["search"]):
        rt._prompt_cache[row["sid"]] = prompt_inputs(model.config.model_type, index)
    return rt


FAMILIES = [("qwen2_vl", qwen2_vl, "qwen2-vl-2b")]
if hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
    FAMILIES.append(("qwen2_5_vl", qwen2_5_vl, "qwen2.5-vl-3b"))
if hasattr(transformers, "Idefics3ForConditionalGeneration"):
    FAMILIES.append(("idefics3", idefics3, "smolvlm-instruct"))


class CachedRuntimeParityTests(unittest.TestCase):
    def test_families_under_test(self):
        self.assertIn("qwen2_vl", [family for family, _, _ in FAMILIES])

    def test_nll_logits_gradients_and_generation_match_the_complete_model(self):
        for family, build, key in FAMILIES:
            with self.subTest(family=family):
                rt = runtime_for(build, key)
                rows = rt.rows["search"]
                torch.testing.assert_close(torch.tensor(rt.nll(rows, bs=1)),
                                           torch.tensor(FULL.nll(rt, rows)), atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(rt.prefill_logits(rows), FULL.prefill_logits(rt, rows),
                                           atol=1e-5, rtol=1e-5)
                name = "model.layers.1.mlp.down_proj.weight"
                torch.testing.assert_close(rt.grads([name], rows, bs=1)[name],
                                           FULL.grads(rt, [name], rows)[name], atol=1e-6, rtol=1e-4)
                for row in rows:
                    fast, full = rt._generate(row), FULL._generate(rt, row)
                    self.assertEqual(fast[0], full[0])
                    self.assertTrue(torch.equal(fast[1].sequences, full[1].sequences))
                parity = rt.reference_parity(rows[0])
                self.assertTrue(parity["logits_allclose"] and parity["argmax_equal"])
                self.assertLess(parity["nll_abs_difference"], 1e-5)
                self.assertEqual(len(rt._features), len(rows))
                rt.close()

    def test_features_follow_every_legal_layout(self):
        for family, build, key in FAMILIES:
            with self.subTest(family=family):
                rt = runtime_for(build, key)
                rows, d = rt.rows["search"], P.dims(rt.lm)
                rng = random.Random(5)
                q = d.get("query_group_size", 1)
                hd = d["head_dim"]
                local = [rng.sample(range(hd), hd) for _ in range(d["heads"] // q)]
                vo = [h * hd + i for h in range(d["heads"]) for i in local[h // q]]
                spec = P.validate_spec({"residual": rng.sample(range(d["hidden"]), d["hidden"]),
                                        "mlp": {"0": rng.sample(range(d["mlp"]), d["mlp"])},
                                        "heads": {"1": list(range(d["heads"]))[::-1] if q == 1 else
                                                  [h for g in reversed(range(d["heads"] // q))
                                                   for h in range(g * q, (g + 1) * q)]},
                                        "vo": {"2": vo}}, d)
                before = rt.nll(rows, bs=1)
                before_logits = rt.prefill_logits(rows)
                fingerprint = R.model_checksum(rt.lm)
                P.apply_permutation(rt, spec)
                try:
                    after = rt.nll(rows, bs=1)
                    torch.testing.assert_close(torch.tensor(after), torch.tensor(before),
                                               atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(torch.tensor(after), torch.tensor(FULL.nll(rt, rows)),
                                               atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(rt.prefill_logits(rows), before_logits,
                                               atol=1e-4, rtol=1e-4)
                    for row in rows:
                        self.assertTrue(torch.equal(rt._generate(row)[1].sequences,
                                                    FULL._generate(rt, row)[1].sequences))
                finally:
                    P.apply_permutation(rt, spec, inverse=True)
                self.assertEqual(R.model_checksum(rt.lm), fingerprint)
                rt.close()

    def test_exact_reuses_apply_to_vqa_rows(self):
        for family, build, key in FAMILIES:
            with self.subTest(family=family):
                rt = runtime_for(build, key, n=3)
                rows = rt.rows["search"]
                rt.rows["search"] = rt._prepare_rows([rows[0], rows[1], rows[0], rows[2], rows[1]])
                self.assertEqual(P.distinct_rows(rt.rows["search"]), ([0, 1, 3], [0, 1, 0, 2, 1]))
                prefix, status = rt.prefix_states(rt.rows["search"], 1)
                self.assertEqual(status, "verified")
                region = R.make_region("down_proj", 1, 0, P.dims(rt.lm), bits=8, page_bytes=64)
                reused = R.RegionContext(rt, region, "noperm", 7, bs=1, grad_bs=1)
                plain = R.RegionContext(rt, region, "noperm", 7, bs=1, grad_bs=1, reuse=False)
                self.assertEqual(reused.nll(), plain.nll())
                self.assertEqual(reused.reuse["distinct_rows"], 3)
                self.assertEqual(reused.reuse["prefix"], "verified")
                self.assertTrue(torch.equal(reused.grad(), plain.grad()))
                rt.close()


class AnswerContinuationTests(unittest.TestCase):
    def test_answer_tokens_follow_the_chat_template_lead(self):
        rt = runtime_for(qwen2_vl, "qwen2-vl-2b", lead=" ")
        self.assertEqual(rt.answer_lead, " ")
        row = rt.rows["search"][0]
        self.assertEqual(rt._answer_ids(row).tolist(), StubTokenizer().encode(" " + row["reference"]))
        self.assertEqual(row["target_tokens"], len(" " + row["reference"]))
        rt.close()
        plain = runtime_for(qwen2_vl, "qwen2-vl-2b")
        self.assertEqual(plain.answer_lead, "")
        self.assertEqual(plain._answer_ids(plain.rows["search"][0]).tolist(),
                         StubTokenizer().encode(plain.rows["search"][0]["reference"]))
        plain.close()
        with self.assertRaisesRegex(ValueError, "before the answer"):
            runtime_for(qwen2_vl, "qwen2-vl-2b", lead=" Sure: ")


def pinned_transformers():
    for line in (Path(D.C.PACKAGE) / "requirements-tf451.txt").read_text().splitlines():
        if line.startswith("transformers=="):
            return line.split("==", 1)[1].strip()
    raise AssertionError("requirements-tf451.txt pins no transformers version")


class EnvironmentTests(unittest.TestCase):
    def test_the_declared_second_environment_runs_every_family(self):
        if transformers.__version__ != pinned_transformers():
            self.skipTest(f"transformers {transformers.__version__} is not the "
                          f"requirements-tf451.txt pin {pinned_transformers()}")
        for name in ("Gemma2ForCausalLM", "Gemma3ForCausalLM", "Idefics3ForConditionalGeneration",
                     "Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration",
                     "LlavaForConditionalGeneration", "StableLmForCausalLM"):
            self.assertTrue(hasattr(transformers, name), name)
        self.assertEqual([family for family, _, _ in FAMILIES], ["qwen2_vl", "qwen2_5_vl", "idefics3"])

    def test_plans_refuse_other_library_versions(self):
        live = D.M.library_versions()
        D.M.check_library_versions(None)            # plans from before versions were frozen
        D.M.check_library_versions(dict(live))
        with self.assertRaisesRegex(RuntimeError, "frozen with"):
            D.M.check_library_versions({**live, "transformers": "0.0.0"})


class PlanRuntimeTests(unittest.TestCase):
    def test_execution_block_accepts_only_a_known_runtime(self):
        expected = {"search_batch_size": 1, "real_gpu_preflight_required": True}
        self.assertTrue(D.execution_matches(dict(expected), expected))
        for runtime in V.RUNTIMES:
            self.assertTrue(D.execution_matches({**expected, "runtime": runtime}, expected))
        self.assertFalse(D.execution_matches({**expected, "runtime": "faster"}, expected))
        self.assertFalse(D.execution_matches({**expected, "search_batch_size": 2}, expected))
        self.assertFalse(D.execution_matches(None, expected))

    def test_plans_without_a_runtime_choice_run_the_complete_model(self):
        self.assertIs(D._runtime_class({"execution": {}}), V.HFVQARegionRuntime)
        self.assertIs(D._runtime_class({"execution": {"runtime": D.DEFAULT_RUNTIME}}),
                      V.CachedVQARegionRuntime)
        with self.assertRaisesRegex(ValueError, "unknown VQA runtime"):
            D._runtime_class({"execution": {"runtime": "faster"}})

    def test_official_okvqa_inputs_are_bound_and_copied_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run = Path(directory) / "inputs", Path(directory) / "run"
            panels = {}
            for split in ("search", "dev"):
                path = root / "panels" / "okvqa" / f"{split}.arrow"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{split} panel".encode())
                panels[split] = {"path": str(path.relative_to(root)), "sha256": D.M.sha256(path),
                                 "bytes": path.stat().st_size, "count": 1}
            (root / "audit.json").write_text("{}")
            audit = {"panels": panels, "selection": {"sha256": "s" * 64},
                     "official_sources": [{"archive": {"path": "a.zip", "sha256": "1", "bytes": 1},
                                           "json": {"path": "a.json", "sha256": "2", "bytes": 2}}],
                     "images": [{"image_id": "1"}], "search_bootstraps": [{"repeat": 0}]}
            run.mkdir()
            sources = []
            with mock.patch("regionalign.experiments.official_okvqa._verified_inputs",
                            return_value=(root, audit, {"answer_field": "answers"})):
                dataset = D._official_okvqa_dataset(run, root, sources)
            self.assertEqual(dataset["status"], "available")
            self.assertEqual(dataset["dataset_binding"]["provider"], "official_okvqa_and_coco")
            self.assertFalse(dataset["dataset_binding"]["hf_repository_revision_claimed"])
            self.assertEqual(dataset["panels"], panels)
            for split, record in panels.items():
                self.assertEqual((run / record["path"]).read_bytes(), f"{split} panel".encode())
            self.assertEqual([record["path"] for record in sources],
                             ["regionalign/experiments/official_okvqa.py"])
            self.assertTrue((run / sources[0]["snapshot"]).is_file())

    def test_plan_arguments_are_checked_before_any_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown VQA runtime"):
                D.make_plan(Path(directory) / "a", {"qwen2-vl-2b": Path(directory)}, {},
                            runtime="faster")
            with self.assertRaisesRegex(ValueError, "official OK-VQA inputs replace"):
                D.make_plan(Path(directory) / "b", {"qwen2-vl-2b": Path(directory)}, {},
                            missing_tasks=("okvqa",), okvqa_official=directory)


if __name__ == "__main__":
    unittest.main()
