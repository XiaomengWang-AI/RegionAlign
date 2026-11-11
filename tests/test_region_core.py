"""CPU checks of RegionAlign geometry, function preservation and evaluator rules.

Use tiny random models and synthetic text; no checkpoint or frozen panel opens.
"""
import json
import random
import subprocess
import sys
import unittest

import torch

torch.set_num_threads(1)

from pathlib import Path

from regionalign.algorithm import search as S
from regionalign.harness import core as P
from regionalign.harness import region as R
from regionalign.harness import evidence as E
from regionalign.tests.helpers import (TINY_GROUP, TinyRuntime, tiny_gqa_lm, tiny_lm,
                                       tiny_text_runtime)

PAGE = 64


def region(rt, kind, layer=1, page=None):
    d = P.dims(rt.lm)
    if page is None:
        page = R.page_count(kind, d, page_bytes=PAGE) // 2 + 1
    return R.make_region(kind, layer, page, d, page_bytes=PAGE)


def codes(rt, name):
    weight = rt.param(name).data
    return P.stored_codes(rt.lm, name, torch.arange(weight.numel())).view(weight.shape)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.d = P.dims(tiny_lm())

    def test_pages_respect_code_width_and_tensor_boundaries(self):
        self.assertEqual(R.elements_per_page(8), 4096)
        self.assertEqual(R.elements_per_page(4), 8192)
        self.assertEqual(R.page_count("gate_proj", self.d, bits=8), 2)
        self.assertEqual(R.page_count("gate_proj", self.d, bits=4), 1)
        page = R.make_region("down_proj", 0, 2, self.d, page_bytes=96)
        self.assertEqual(R.page_segments(page, self.d), [(1, 64, 128), (2, 0, 32)])
        self.assertEqual(R.primary_segment(page, self.d), (1, 64, 128))

    def test_gain_embedding_and_head_are_not_region_tensors(self):
        for kind in ("input_layernorm", "embed_tokens", "lm_head"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                R.make_region(kind, 0, 0, self.d)

    def test_q_and_k_row_reachability_keeps_rope_coordinates(self):
        q = R.make_region("q_proj", 0, 21, self.d, page_bytes=PAGE)
        self.assertEqual(R.reachable_rows(q, self.d), [5, 21, 37, 53])
        for kv_heads, expected_k, expected_v in ((2, [5, 21], 32), (1, [5], 16)):
            d = P.dims(tiny_gqa_lm(torch.float32, kv_heads=kv_heads))
            k = R.make_region("k_proj", 0, 5, d, page_bytes=PAGE)
            v = R.make_region("v_proj", 0, 5, d, page_bytes=PAGE)
            self.assertEqual(R.reachable_rows(k, d), expected_k)
            self.assertEqual(len(R.reachable_rows(v, d)), expected_v)


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.rt = tiny_text_runtime(fmt="int8", n=2)
        self.d = P.dims(self.rt.lm)
        self.rows = self.rt.rows["dev"]

    def test_every_kind_moves_its_source_row_into_the_page_and_keeps_the_function(self):
        reference = self.rt.prefill_logits(self.rows)
        before = R.model_checksum(self.rt.lm)
        rng = random.Random(0)
        for kind in R.KINDS:
            reg = region(self.rt, kind)
            row, first, stop = R.primary_segment(reg, self.d)
            source = rng.choice([r for r in R.reachable_rows(reg, self.d) if r != row])
            original = codes(self.rt, reg["tensor"])
            columns = [0, 3, 127] if kind == "down_proj" else []
            spec = R.placement(reg, self.d, source, columns)
            P.apply_permutation(self.rt, spec)
            try:
                deployed = codes(self.rt, reg["tensor"])
                cols = original.shape[1]
                for c in columns or range(first, stop):
                    [[_, e, _]] = R.deploy(reg, self.d, spec, [[reg["tensor"], source * cols + c, 0]])
                    self.assertEqual(e // cols, row, kind)
                    self.assertTrue(first <= e % cols < stop, kind)
                    self.assertEqual(int(deployed.view(-1)[e]), int(original[source, c]), kind)
                logits = self.rt.prefill_logits(self.rows)
                self.assertTrue(torch.allclose(logits, reference, atol=0.05, rtol=0.02), kind)
            finally:
                P.apply_permutation(self.rt, spec, inverse=True)
        self.assertEqual(R.model_checksum(self.rt.lm), before)

    def test_a_flip_has_the_same_effect_before_and_after_placement(self):
        reg = region(self.rt, "gate_proj")
        spec = R.placement(reg, self.d, 7)
        flips = [[reg["tensor"], 7 * 64 + 5, 7], [reg["tensor"], 7 * 64 + 9, 6]]
        P.apply_flips(self.rt.lm, flips)
        plain = P.mean_nll(self.rt, self.rows)
        P.apply_flips(self.rt.lm, flips)
        P.apply_permutation(self.rt, spec)
        deployed = R.deploy(reg, self.d, spec, flips)
        P.apply_flips(self.rt.lm, deployed)
        moved = P.mean_nll(self.rt, self.rows)
        P.apply_flips(self.rt.lm, deployed)
        P.apply_permutation(self.rt, spec, inverse=True)
        self.assertAlmostEqual(plain, moved, delta=0.02)

    def test_gqa_and_mqa_attention_rows_are_placed_with_legal_shared_kv_orders(self):
        for kv_heads, cases in (
                (2, (("q_proj", 21, 53), ("k_proj", 5, 21), ("v_proj", 5, 27))),
                (1, (("q_proj", 21, 53), ("v_proj", 5, 11)))):
            lm = tiny_gqa_lm(torch.float32, kv_heads=kv_heads)
            P.quantize(lm, "int8")
            rt, d = TinyRuntime(lm, []), P.dims(lm)
            ids = torch.randint(0, 64, (2, 11))
            reference = lm(ids).logits
            for kind, page, source in cases:
                reg = R.make_region(kind, 0, page, d, page_bytes=PAGE)
                target, first, stop = R.primary_segment(reg, d)
                original = codes(rt, reg["tensor"])
                spec = R.placement(reg, d, source)
                P.apply_permutation(rt, spec)
                try:
                    [[_, element, _]] = R.deploy(
                        reg, d, spec, [[reg["tensor"], source * original.shape[1] + 3, 0]])
                    self.assertEqual(element // original.shape[1], target, kind)
                    self.assertTrue(first <= element % original.shape[1] < stop, kind)
                    self.assertEqual(int(codes(rt, reg["tensor"]).view(-1)[element]),
                                     int(original[source, 3]), kind)
                    torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
                finally:
                    P.apply_permutation(rt, spec, inverse=True)
            torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)

    def test_int4_row_and_columns_move_only_as_complete_scale_groups(self):
        lm = tiny_lm(torch.float32)
        P.quantize(lm, "int4", group=TINY_GROUP)
        rt, d = TinyRuntime(lm, []), P.dims(lm)
        # 16 bytes hold 32 INT4 codes, exactly two test scale groups.
        reg = R.make_region("down_proj", 0, 1, d, bits=4, page_bytes=16)
        target, first, stop = R.primary_segment(reg, d)
        source, selected = 33, [3, 70]
        original = codes(rt, reg["tensor"])
        scales = {name: value.clone() for name, value in lm.ar_quant["scales"].items()}
        ids = torch.randint(0, 64, (2, 11))
        reference = lm(ids).logits
        spec = R.placement(reg, d, source, selected)
        self.assertTrue(P.group_preserving(spec["residual"], TINY_GROUP))
        self.assertTrue(P.group_preserving(spec["mlp"][0], TINY_GROUP))
        P.apply_permutation(rt, spec)
        try:
            for column in selected:
                [[_, element, _]] = R.deploy(
                    reg, d, spec, [[reg["tensor"], source * original.shape[1] + column, 0]])
                self.assertEqual(element // original.shape[1], target)
                self.assertTrue(first <= element % original.shape[1] < stop)
                self.assertEqual(int(codes(rt, reg["tensor"]).view(-1)[element]),
                                 int(original[source, column]))
            torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
        finally:
            P.apply_permutation(rt, spec, inverse=True)
        for name, value in lm.ar_quant["scales"].items():
            self.assertTrue(torch.equal(scales[name], value), name)



class RuleTests(unittest.TestCase):
    def setUp(self):
        self.rt = tiny_text_runtime(fmt="int8", n=2)
        self.d = P.dims(self.rt.lm)
        self.reg = region(self.rt, "up_proj")
        self.row = R.page_segments(self.reg, self.d)[0][0]

    def flip(self, row, col, bit):
        return [self.reg["tensor"], row * 64 + col, bit]

    def test_flips_must_stay_on_the_page_of_the_region_tensor(self):
        good = [self.flip(self.row, 1, 0), self.flip(self.row, 2, 1)]
        self.assertTrue(R.check_region_attack(self.rt, self.reg, good, 2)["valid"])
        off_page = [self.flip(self.row + 1, 1, 0)]
        self.assertFalse(R.check_region_attack(self.rt, self.reg, off_page, 1)["valid"])
        other = [["model.layers.0.mlp.up_proj.weight", self.row * 64 + 1, 0]]
        self.assertFalse(R.check_region_attack(self.rt, self.reg, other, 1)["valid"])
        self.assertFalse(R.check_region_attack(self.rt, self.reg, good, 1)["valid"])

    def test_region_geometry_must_match_the_model_code_width(self):
        lm = tiny_lm(torch.float32)
        P.quantize(lm, "int4", group=TINY_GROUP)
        rt, d = TinyRuntime(lm, []), P.dims(lm)
        wrong = R.make_region("up_proj", 0, 0, d, bits=8, page_bytes=PAGE)
        with self.assertRaisesRegex(ValueError, "declares 8-bit codes"):
            R.RegionContext(rt, wrong, "local", 1)
        verdict = R.check_region_attack(rt, wrong, [[wrong["tensor"], 0, 0]], 1)
        self.assertFalse(verdict["valid"])
        self.assertTrue(any("declares 8-bit codes" in reason for reason in verdict["reasons"]))

    def test_the_code_minus_128_is_out_of_range(self):
        unsigned = codes(self.rt, self.reg["tensor"])[self.row]
        moves = [(col, bit) for col in range(64) for bit in range(8)
                 if int(unsigned[col]) ^ (1 << bit) == 0x80]
        self.assertTrue(moves, "every INT8 row has a code one flip away from -128")
        col, bit = moves[0]
        verdict = R.check_region_attack(self.rt, self.reg, [self.flip(self.row, col, bit)], 1)
        self.assertFalse(verdict["valid"])
        self.assertTrue(any("violates inrange" in r for r in verdict["reasons"]))

    def test_each_arm_may_deploy_only_its_own_layout(self):
        moved = R.placement(self.reg, self.d, (self.row + 3) % 128)
        self.assertTrue(R.arm_reasons(self.reg, "noperm", moved, self.d, 1))
        self.assertFalse(R.arm_reasons(self.reg, "local", moved, self.d, 1))
        seeded = R.placement(self.reg, self.d, R.random_row(self.reg, self.d, 1))
        self.assertFalse(R.arm_reasons(self.reg, "random", seeded, self.d, 1))
        if P.spec_to_json(seeded) != P.spec_to_json(moved):
            self.assertTrue(R.arm_reasons(self.reg, "random", moved, self.d, 1))

    def test_the_budget_is_enforced_by_the_context(self):
        ctx = R.RegionContext(self.rt, self.reg, "local", 1, budget=4, log=lambda *a: None)
        ctx.nll()
        ctx.grad()
        with self.assertRaises(R.BudgetExceeded):
            ctx.nll()
        with self.assertRaises(ValueError):
            ctx.nll([["model.layers.0.mlp.up_proj.weight", 0, 0]])



class RandomBaselineCandidateTests(unittest.TestCase):
    @staticmethod
    def expected_preimage(ctx, spec):
        rows, cols = ctx.shape
        flips = [[ctx.name, element, 0] for element in range(rows * cols)]
        deployed = ctx.deploy(spec, flips)
        page = {row * cols + col for row, first, stop in ctx.segments
                for col in range(first, stop)}
        return {original[1] for original, image in zip(flips, deployed) if image[1] in page}

    @staticmethod
    def table(ctx, spec):
        W = ctx.weight()
        gradient = torch.zeros_like(W, dtype=torch.float32)
        activation = torch.ones(W.shape[1], dtype=torch.float32)
        return S.layout_page_table(ctx, W, gradient, activation, spec)

    def test_random_search_uses_the_complete_fixed_page_preimage_under_gqa_int4(self):
        lm = tiny_gqa_lm(torch.float32, kv_heads=2)
        P.quantize(lm, "int4", group=TINY_GROUP)
        rt, d = TinyRuntime(lm, []), P.dims(lm)
        reg = R.make_region("q_proj", 0, 3, d, bits=4, page_bytes=PAGE)
        ctx = R.RegionContext(rt, reg, "random", 17)
        spec = ctx.placement(ctx.forced_row)
        table = self.table(ctx, spec)
        expected = self.expected_preimage(ctx, spec)

        self.assertEqual(set(table[0].tolist()), expected)
        self.assertEqual(len(expected), R.elements_per_page(4, PAGE))
        self.assertGreater(len({element // ctx.shape[1] for element in expected}), 1)

        group = S.groups_of(ctx, ctx.weight(),
                            {"g": torch.zeros_like(ctx.weight(), dtype=torch.float32),
                             "act": torch.ones(ctx.shape[1])}, 0.0)[0]
        self.assertEqual(set(element for element, _ in group.queue), expected)
        self.assertEqual(P.spec_to_json(group.fixed_spec), P.spec_to_json(spec))

        # A physical 4 KiB INT4 page has 8192 codes.  The random arm must not
        # inherit the historical 4096-entry ranking cap and silently discard
        # half of this fixed page before exact screening begins.
        full = R.make_region("gate_proj", 0, 0, d, bits=4, page_bytes=4096)
        full_ctx = R.RegionContext(rt, full, "random", 17)
        full_group = S.groups_of(
            full_ctx, full_ctx.weight(),
            {"g": torch.zeros_like(full_ctx.weight(), dtype=torch.float32),
             "act": torch.ones(full_ctx.shape[1])}, 0.0)[0]
        full_expected = self.expected_preimage(full_ctx, full_group.fixed_spec)
        self.assertEqual(len(full_expected), 8192)
        self.assertEqual(set(element for element, _ in full_group.queue), full_expected)

    def test_fixed_page_preimage_honors_both_row_and_column_orders(self):
        lm = tiny_lm(torch.float32)
        P.quantize(lm, "int4", group=TINY_GROUP)
        rt, d = TinyRuntime(lm, []), P.dims(lm)
        reg = R.make_region("down_proj", 0, 1, d, bits=4, page_bytes=PAGE)
        ctx = R.RegionContext(rt, reg, "local", 3)

        residual = list(range(d["hidden"]))
        residual[:TINY_GROUP], residual[TINY_GROUP:2 * TINY_GROUP] = (
            residual[TINY_GROUP:2 * TINY_GROUP], residual[:TINY_GROUP])
        mlp = list(range(d["mlp"]))
        mlp[:TINY_GROUP], mlp[-TINY_GROUP:] = mlp[-TINY_GROUP:], mlp[:TINY_GROUP]
        spec = P.validate_spec({"residual": residual, "mlp": {"0": mlp}}, d)

        table = self.table(ctx, spec)
        expected = self.expected_preimage(ctx, spec)
        self.assertEqual(set(table[0].tolist()), expected)
        self.assertEqual(len(expected), R.elements_per_page(4, PAGE))
        orders = P.axis_orders(spec, d)
        self.assertIn((ctx.name, 0), orders)
        self.assertIn((ctx.name, 1), orders)


class EvidenceAndIsolationTests(unittest.TestCase):
    def test_package_is_self_contained_and_has_no_cell_or_track_rules(self):
        probe = (
            "import json,sys; import regionalign.experiments.llm; "
            "import regionalign.experiments.vqa; "
            "import regionalign.algorithm.search; "
            "print(json.dumps(sorted(name for name in sys.modules if "
            "name.split('.')[0] in ('alignment_attack', 'vlm_downstream', 'tests'))))"
        )
        result = subprocess.run([sys.executable, "-B", "-c", probe], capture_output=True, text=True,
                                check=True, cwd=Path(__file__).resolve().parents[2])
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), [])
        for name in ("TASK", "FORMAT", "TRACK", "MODE", "region_set", "TUNING_SEED", "evaluate_artifact"):
            self.assertFalse(hasattr(R, name), name)
        for name in ("cell_map", "TRACKS", "MODES", "PROFILED", "GUARDED", "HARDENED", "protected",
                     "row_bound", "Runtime", "SearchContext", "TASK", "MODEL_DIR"):
            self.assertFalse(hasattr(P, name), name)

    def test_call_accounting_rejects_reset_or_excess_spend(self):
        self.assertEqual(R.budget_reasons({"spent": 300, "calls": {"eval": 294, "grad": 2}}), [])
        for artifact in ({"spent": 0, "calls": {"eval": 294, "grad": 2}},
                         {"spent": 301, "calls": {"eval": 295, "grad": 2}},
                         {"spent": 0, "calls": {"eval": True, "grad": 0}}):
            self.assertTrue(R.budget_reasons(artifact))

    def test_json_sanitize_keeps_verdicts_and_removes_nonfinite_numbers(self):
        original = {"valid": False, "nll": float("inf"), "values": (1.5, float("nan"))}
        self.assertEqual(E.sanitize(original), {"valid": False, "nll": None, "values": [1.5, None]})
        self.assertTrue(torch.isinf(torch.tensor(original["nll"])))

if __name__ == "__main__":
    unittest.main()
