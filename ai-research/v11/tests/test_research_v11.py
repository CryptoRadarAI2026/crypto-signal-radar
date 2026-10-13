import importlib.util
import pathlib
import unittest
import numpy as np
import sys

path = pathlib.Path(__file__).resolve().parents[1] / "research_v11.py"

spec = importlib.util.spec_from_file_location("research_v11", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def sample(pair, entry, exit_, ret=0.01):
    return module.Sample(
        entry - 60,
        pair,
        (0.0,) * 16,
        1,
        ret,
        "TP",
        100.0,
        101.0,
        entry,
        exit_,
    )


class V11Tests(unittest.TestCase):

    def test_capacity_released_on_actual_exit(self):
        base = 1_800_000_000

        a = sample("BTC", base, base + 180)
        b = sample("ETH", base + 60, base + 900)
        blocked = sample("SOL", base + 120, base + 900)
        accepted = sample("SOL", base + 240, base + 900)

        selected = module.select_trades(
            [a, b, blocked, accepted],
            np.ones(4),
            0.5,
        )

        self.assertEqual(
            [x.pair for x, _ in selected],
            ["BTC", "ETH", "SOL"],
        )

        self.assertEqual(
            selected[-1][0].entry_ts,
            base + 240,
        )

    def test_cooldown_after_exit(self):
        base = 1_800_000_000

        a = sample("BTC", base, base + 180)
        early = sample(
            "BTC",
            base + 180 + 1799,
            base + 3000,
        )
        boundary = sample(
            "BTC",
            base + 180 + 1800,
            base + 3000,
        )

        selected = module.select_trades(
            [a, early, boundary],
            np.ones(3),
            0.5,
        )

        self.assertEqual(
            [x for x, _ in selected],
            [a, boundary],
        )

    def test_no_same_pair_overlap(self):
        base = 1_800_000_000

        a = sample("BTC", base, base + 900)
        b = sample("BTC", base + 60, base + 900)

        selected = module.select_trades(
            [a, b],
            np.ones(2),
            0.5,
        )

        self.assertEqual(len(selected), 1)

    def test_invalid_timestamp(self):
        base = 1_800_000_000

        with self.assertRaises(ValueError):
            module.select_trades(
                [sample("BTC", base, base - 1)],
                np.ones(1),
                0.5,
            )

    def test_simulated_exit_index(self):
        n = module.HORIZON + 1

        o = np.full(n, 100.0)
        h = np.full(n, 100.1)
        l = np.full(n, 99.9)
        c = np.full(n, 100.0)

        h[3] = 102.0

        net, reason, entry, exit_fill, exit_idx = (
            module.simulate_long(
                o, h, l, c, 0, 0.2
            )
        )

        self.assertEqual(
            (reason, exit_idx),
            ("TP", 3),
        )

        self.assertGreater(net, 0)

    def test_holdout_sealed_in_source(self):
        import inspect

        source = inspect.getsource(module.research)

        self.assertNotIn(
            "predict_proba(xte)",
            source,
        )

        self.assertIn(
            "holdout_evaluated=False",
            source,
        )


if __name__ == "__main__":
    unittest.main()
