"""CPU-only synthetic timing-table fixtures; no row is runtime evidence."""

import unittest

from graphbudget.plan import fingerprint
from graphbudget.policy import (
    ForcedEagerError,
    PATH_EAGER,
    PATH_NATIVE,
    compile_policy,
    decide_policy,
    dump_policy,
    load_policy,
)
from test_plan import _identity


def _policy(identity, **overrides):
    value = {
        "schema_version": 1,
        "fingerprint": fingerprint(identity),
        "capture_sizes": [2, 4, 8],
        "entries": [
            {
                "batch_size": 2,
                "padded_size": 2,
                "context_length": 1024,
                "graph_ns": 120.0,
                "eager_ns": 100.0,
                "noise_ns": 3.0,
                "samples": 10,
                "provenance": "measured",
            }
        ],
        "enabled": True,
    }
    value.update(overrides)
    return value


def _context(**overrides):
    value = {"batch_size": 2, "padded_size": 2, "context_length": 1024}
    value.update(overrides)
    return value


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.identity = _identity()
        self.digest = fingerprint(self.identity)
        self.policy = _policy(self.identity)

    def test_disabled_policy_is_fully_native(self):
        decision = decide_policy(
            _policy(self.identity, enabled=False),
            None,
            True,
            None,
        )
        self.assertEqual(decision, {
            "eligible": True,
            "selected_path": PATH_NATIVE,
            "reason": "policy_disabled",
            "forced_test": False,
        })
        forced_disabled = decide_policy(
            _policy(self.identity, enabled=False),
            _context(),
            True,
            self.digest,
            force_eager=True,
            test_only_force_eager=True,
        )
        self.assertEqual(forced_disabled["selected_path"], PATH_NATIVE)

    def test_upstream_eligibility_remains_authoritative(self):
        decision = decide_policy(self.policy, _context(), False, self.digest)
        self.assertFalse(decision["eligible"])
        self.assertEqual(decision["selected_path"], PATH_NATIVE)
        self.assertEqual(decision["reason"], "upstream_ineligible")

    def test_eager_only_when_beyond_noise(self):
        decision = decide_policy(self.policy, _context(), True, self.digest)
        self.assertEqual(decision["selected_path"], PATH_EAGER)
        within_noise = _policy(
            self.identity,
            entries=[dict(self.policy["entries"][0], graph_ns=102.0)],
        )
        native = decide_policy(within_noise, _context(), True, self.digest)
        self.assertEqual(native["selected_path"], PATH_NATIVE)
        self.assertEqual(native["reason"], "graph_not_slower_beyond_noise")

    def test_compile_policy_is_immutable_and_tuple_keyed(self):
        compiled = compile_policy(self.policy, self.digest)
        self.assertEqual(compiled["status"], "loaded")
        self.assertEqual(tuple(compiled["entries"]), ((2, 2, 1024),))
        decision = decide_policy(compiled, _context(), True, self.digest)
        self.assertEqual(decision["selected_path"], PATH_EAGER)
        with self.assertRaises(TypeError):
            compiled["status"] = "changed"
        with self.assertRaises(TypeError):
            compiled["entries"][(2, 2, 1024)]["graph_ns"] = 1

    def test_invalid_sample_types_disable_the_row_without_crashing(self):
        for value in ("invalid", [], {}, True, -1, 1.5):
            policy = _policy(self.identity,
                entries=[dict(self.policy["entries"][0], samples=value)])
            compiled = compile_policy(policy, self.digest)
            self.assertEqual(decide_policy(compiled, _context(), True, self.digest)["selected_path"],
                             PATH_NATIVE)
    def test_compile_rejects_stale_or_non_sha_fingerprint_to_native(self):
        stale = compile_policy(self.policy, "0" * 64)
        self.assertEqual(stale["status"], "native_fallback")
        self.assertEqual(
            decide_policy(stale, _context(), True, self.digest)["reason"],
            "profile_fingerprint_mismatch",
        )
        missing = compile_policy(self.policy, "not-a-digest")
        self.assertEqual(
            decide_policy(missing, _context(), True, self.digest)["reason"],
            "profile_fingerprint_missing",
        )
        invalid_policy = _policy(self.identity, fingerprint="not-a-digest")
        self.assertEqual(
            decide_policy(invalid_policy, _context(), True, self.digest)["reason"],
            "invalid_policy",
        )

    def test_compile_marks_insufficient_samples_native(self):
        low_samples = _policy(
            self.identity,
            entries=[dict(self.policy["entries"][0], samples=2)],
        )
        compiled = compile_policy(low_samples, self.digest)
        decision = decide_policy(compiled, _context(), True, self.digest)
        self.assertEqual(decision["selected_path"], PATH_NATIVE)
        self.assertEqual(decision["reason"], "profile_samples_insufficient")

    def test_padded_size_must_be_capture_ceiling(self):
        bad = _policy(
            self.identity,
            entries=[dict(self.policy["entries"][0], padded_size=4)],
        )
        compiled = compile_policy(bad, self.digest)
        decision = decide_policy(compiled, _context(), True, self.digest)
        self.assertEqual(decision["selected_path"], PATH_NATIVE)
        self.assertEqual(decision["reason"], "invalid_policy")

    def test_missing_noise_profile_mismatch_and_context_are_native(self):
        missing_noise_entry = dict(self.policy["entries"][0])
        del missing_noise_entry["noise_ns"]
        missing_noise = decide_policy(
            _policy(self.identity, entries=[missing_noise_entry]),
            _context(),
            True,
            self.digest,
        )
        self.assertEqual(missing_noise["selected_path"], PATH_NATIVE)

        stale = decide_policy(self.policy, _context(), True, "0" * 64)
        self.assertEqual(stale["reason"], "profile_fingerprint_mismatch")
        missing = decide_policy(self.policy, None, True, self.digest)
        self.assertEqual(missing["reason"], "missing_or_heterogeneous_context")
        other = decide_policy(self.policy, _context(context_length=2048), True, self.digest)
        self.assertEqual(other["reason"], "profile_entry_missing")

    def test_synthetic_profile_is_native(self):
        synthetic = _policy(
            self.identity,
            entries=[dict(self.policy["entries"][0], provenance="synthetic_cpu_fixture")],
        )
        decision = decide_policy(synthetic, _context(), True, self.digest)
        self.assertEqual(decision["selected_path"], PATH_NATIVE)
        self.assertEqual(decision["reason"], "profile_not_measured")

    def test_force_eager_requires_test_marker(self):
        with self.assertRaises(ForcedEagerError):
            decide_policy(self.policy, _context(), True, self.digest, force_eager=True)
        decision = decide_policy(
            self.policy,
            _context(),
            True,
            self.digest,
            force_eager=True,
            test_only_force_eager=True,
        )
        self.assertEqual(decision["selected_path"], PATH_EAGER)
        self.assertTrue(decision["forced_test"])
        no_profile = decide_policy(
            None,
            _context(),
            True,
            None,
            force_eager=True,
            test_only_force_eager=True,
        )
        self.assertEqual(no_profile["selected_path"], PATH_EAGER)

    def test_policy_round_trip(self):
        loaded = load_policy(dump_policy(self.policy))
        self.assertEqual(loaded, self.policy)


if __name__ == "__main__":
    unittest.main()
