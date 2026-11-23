"""Kernel rules for decoder families beyond the Llama core, on tiny random CPU models.

Gemma-2/3 declare ``head_dim`` (heads x head_dim differs from hidden_size), add
sandwich norms around the MLP that live on the residual stream, scale input
embeddings inside the model, and Gemma-2 soft-caps attention and output logits.
"""
import random
import unittest
from types import SimpleNamespace

import torch
torch.set_num_threads(1)
import transformers

from regionalign.experiments import clean_mmlu as M
from regionalign.harness import core as P, region as R
from regionalign.tests.helpers import TinyRuntime, tiny_lm

HAS_GEMMA3 = hasattr(transformers, "Gemma3ForCausalLM")


def tiny_gemma(kind="gemma2", kv_heads=2, softcap=None, seed=0):
    torch.manual_seed(seed)
    common = dict(vocab_size=64, hidden_size=72, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=16,
                  max_position_embeddings=128, query_pre_attn_scalar=16, pad_token_id=0,
                  bos_token_id=2, eos_token_id=1, final_logit_softcapping=softcap)
    if kind == "gemma2":
        config = transformers.Gemma2Config(attn_logit_softcapping=50.0, **common)
        config._attn_implementation = "eager"
        lm = transformers.Gemma2ForCausalLM(config)
    else:
        lm = transformers.Gemma3ForCausalLM(transformers.Gemma3TextConfig(
            sliding_window=4, sliding_window_pattern=2, **common))
    lm = lm.eval().requires_grad_(False)
    for p in lm.parameters():             # random gains so their permutation matters
        if p.dim() == 1:
            p.data.uniform_(0.5, 1.5)
    return lm


def legal_spec(d, rng, group=None):
    """Random legal layout touching every family; int4 orders keep groups of ``group``."""
    def order(size):
        if not group:
            return rng.sample(range(size), size)
        starts = list(range(0, size, group))
        full = [s for s in starts if s + group <= size]
        moved = rng.sample(full, len(full))
        out = [s + i for s in moved for i in rng.sample(range(group), group)]
        for s in starts[len(full):]:
            out += [s + i for i in rng.sample(range(size - s), size - s)]
        return out
    hd, q, n = d["head_dim"], d.get("query_group_size", 1), d["heads"]
    blocks = rng.sample(range(n // q), n // q)
    heads = [b * q + i for b in blocks for i in rng.sample(range(q), q)]
    local = [rng.sample(range(hd), hd) for _ in range(n // q)]
    vo = [h * hd + i for h in range(n) for i in local[h // q]]
    return P.validate_spec({"residual": order(d["hidden"]), "mlp": {"1": order(d["mlp"])},
                            "heads": {"0": heads}, "vo": {"0": vo, "2": vo}}, d)


class ExplicitHeadDimTests(unittest.TestCase):
    def test_dims_use_declared_head_dim_and_reject_bad_values(self):
        base = dict(num_hidden_layers=2, hidden_size=72, intermediate_size=128,
                    num_attention_heads=4, num_key_value_heads=2)
        self.assertEqual(P.dims_from_config(SimpleNamespace(**base, head_dim=16))["head_dim"], 16)
        self.assertEqual(P.dims_from_config(SimpleNamespace(**base))["head_dim"], 18)
        self.assertEqual(P.attention_width(P.dims_from_config(SimpleNamespace(**base, head_dim=16))), 64)
        for bad in (0, -1, True, 1.5):
            with self.subTest(head_dim=bad), self.assertRaisesRegex(ValueError, "head_dim"):
                P.dims_from_config(SimpleNamespace(**base, head_dim=bad))
        with self.assertRaisesRegex(ValueError, "implicit"):
            P.dims_from_config(SimpleNamespace(**{**base, "hidden_size": 70}))

    def check_family(self, kind, kv_heads):
        for fmt in ("int8", "int4"):
            with self.subTest(kind=kind, fmt=fmt):
                lm = tiny_gemma(kind, kv_heads)
                P.quantize(lm, fmt, group=16)
                rt, d = TinyRuntime(lm, []), P.validate_parameter_layout(lm)
                self.assertNotEqual(P.attention_width(d), d["hidden"])
                spec = legal_spec(d, random.Random(3), group=16 if fmt == "int4" else None)
                orders = P.parameter_axis_orders(lm, spec, d)
                for layer in range(d["layers"]):
                    for norm in P.EXTRA_RESIDUAL_NORMS:
                        self.assertIn((f"model.layers.{layer}.{norm}.weight", 0), orders)
                ids = torch.randint(3, 64, (2, 11))
                reference = lm(ids).logits
                weights = {n: p.data.clone() for n, p in lm.named_parameters()}
                scales = {n: s.clone() for n, s in lm.ar_quant["scales"].items()}
                P.apply_permutation(rt, spec)
                torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
                P.apply_permutation(rt, spec, inverse=True)
                for name, param in lm.named_parameters():
                    self.assertTrue(torch.equal(weights[name], param.data), name)
                for name, s in lm.ar_quant["scales"].items():
                    self.assertTrue(torch.equal(scales[name], s), name)

    def test_gemma2_layouts_keep_function_and_invert_exactly(self):
        self.check_family("gemma2", kv_heads=2)

    @unittest.skipUnless(HAS_GEMMA3, "transformers without Gemma-3")
    def test_gemma3_mqa_layouts_keep_function_and_invert_exactly(self):
        self.check_family("gemma3", kv_heads=1)

    def test_region_geometry_uses_the_attention_width(self):
        lm = tiny_gemma()
        d = P.dims(lm)
        for kind in R.KINDS:
            with self.subTest(kind=kind):
                region = R.make_region(kind, 1, 0, d, bits=8, page_bytes=64)
                self.assertEqual(R.tensor_shape(kind, d), tuple(lm.get_parameter(region["tensor"]).shape))
                rows = R.reachable_rows(region, d)
                self.assertTrue(rows)
                R.placement(region, d, rows[-1])


class ParameterAccountingTests(unittest.TestCase):
    def test_a_parameter_without_a_layout_rule_is_rejected(self):
        lm = tiny_lm()
        P.validate_parameter_layout(lm)
        lm.model.layers[0].self_attn.register_parameter(
            "rotary_gain", torch.nn.Parameter(torch.ones(4), requires_grad=False))
        with self.assertRaisesRegex(ValueError, "no legal layout rule"):
            P.validate_parameter_layout(lm)

    def test_optional_parameters_must_have_their_layout_shape(self):
        lm = tiny_gemma()
        norm = lm.model.layers[0].pre_feedforward_layernorm
        norm.weight = torch.nn.Parameter(torch.ones(8), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "pre_feedforward_layernorm"):
            P.validate_parameter_layout(lm)


class OutputHeadAndReuseTests(unittest.TestCase):
    def rows(self, n=4):
        g = torch.Generator().manual_seed(1)
        return [{"sid": f"s{i}", "prompt": torch.randint(3, 64, (6,), generator=g).tolist(),
                 "features": torch.randn(3, 72, generator=g), "answer": [5, 6], "answers": []}
                for i in range(n)]

    def test_runtime_head_applies_the_final_logit_soft_cap(self):
        lm = tiny_gemma(softcap=3.0)
        rt = TinyRuntime(lm, [])
        ids = torch.randint(3, 64, (2, 9))
        official = lm(ids).logits
        self.assertLessEqual(float(official.abs().max()), 3.0)
        torch.testing.assert_close(rt._logits(lm.model(input_ids=ids)[0]), official, atol=0, rtol=0)

    def test_resumed_passes_survive_embedding_scaling(self):
        # Gemma multiplies inputs_embeds by sqrt(hidden) inside the model; a resumed
        # pass must not apply that to the recorded residual state a second time.
        kinds = ["gemma2"] + (["gemma3"] if HAS_GEMMA3 else [])
        for kind in kinds:
            lm = tiny_gemma(kind, kv_heads=1 if kind == "gemma3" else 2)
            P.quantize(lm, "int8")
            rt = TinyRuntime(lm, self.rows())
            for layer in (1, 2):
                with self.subTest(kind=kind, layer=layer):
                    prefix, status = rt.prefix_states(rt.rows["search"], layer)
                    self.assertEqual(status, "verified")

    def test_layout_kernel_choice_follows_attention_soft_capping(self):
        self.assertEqual(M.attention_implementation({"attn_logit_softcapping": 50.0}), "eager")
        self.assertIsNone(M.attention_implementation({"attn_logit_softcapping": None}))
        self.assertIsNone(M.attention_implementation({}))
        self.assertEqual(M.attention_implementation(SimpleNamespace(attn_logit_softcapping=30.0)), "eager")
        self.assertIsNone(M.attention_implementation(SimpleNamespace()))


class DistinctRowTests(unittest.TestCase):
    def test_rows_with_raw_bytes_collapse_by_content(self):
        image = {"bytes": b"\x00\x01jpeg", "path": "a.jpg"}
        rows = [{"sid": "a", "image": image}, {"sid": "b", "image": {"bytes": b"other", "path": None}},
                {"sid": "a", "image": dict(image)}]
        self.assertEqual(P.distinct_rows(rows), ([0, 1], [0, 1, 0]))
        changed = [rows[0], {"sid": "a", "image": {"bytes": b"\x00\x01jpeg!", "path": "a.jpg"}}]
        self.assertEqual(P.distinct_rows(changed), ([0, 1], [0, 1]))
        self.assertIsNone(P.distinct_rows([{"sid": "a", "features": torch.ones(2)}]))


if __name__ == "__main__":
    unittest.main()
