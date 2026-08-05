# Owner(s): ["module: dynamo"]
"""Tests for ByteArrayVariable: bytearray support in Dynamo.

Ported from CPython's Lib/test/test_bytes.py (BaseBytesTest class).
Only tests covering operations supported in PR 1 (read-only ops + wiring)
are included; mutation, methods, and conversions will follow in later PRs.
"""

import sys
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test, run_tests


class BaseBytesTest:
    """Ported from CPython BaseBytesTest -- read-only subset.

    Subclass sets type2test = bytearray (or bytes for cross-validation).
    Ref: https://github.com/python/cpython/blob/v3.13.0/Lib/test/test_bytes.py#L44
    """

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    # CPython: BaseBytesTest.test_basics
    @make_dynamo_test
    def test_basics(self):
        b = self.type2test()
        self.assertEqual(type(b), self.type2test)

    # CPython: BaseBytesTest.test_empty_sequence
    @make_dynamo_test
    def test_empty_sequence(self):
        b = self.type2test()
        self.assertEqual(len(b), 0)
        self.assertRaises(IndexError, lambda: b[0])
        self.assertRaises(IndexError, lambda: b[1])
        self.assertRaises(IndexError, lambda: b[-1])

    # CPython: BaseBytesTest.test_from_iterable
    @make_dynamo_test
    def test_from_iterable(self):
        b = self.type2test(range(256))
        self.assertEqual(len(b), 256)
        self.assertEqual(list(b), list(range(256)))

    # CPython: BaseBytesTest.test_from_tuple
    @make_dynamo_test
    def test_from_tuple(self):
        b = self.type2test(tuple(range(256)))
        self.assertEqual(len(b), 256)
        self.assertEqual(list(b), list(range(256)))

    # CPython: BaseBytesTest.test_from_list
    @make_dynamo_test
    def test_from_list(self):
        b = self.type2test(list(range(256)))
        self.assertEqual(len(b), 256)
        self.assertEqual(list(b), list(range(256)))

    # CPython: BaseBytesTest.test_from_index
    @make_dynamo_test
    def test_from_index(self):
        b = self.type2test([0, 1, 2, 255])
        self.assertEqual(list(b), [0, 1, 2, 255])

    # CPython: BaseBytesTest.test_from_int
    @make_dynamo_test
    def test_from_int(self):
        b = self.type2test(0)
        self.assertEqual(b, self.type2test())
        b = self.type2test(10)
        self.assertEqual(b, self.type2test([0] * 10))

    # CPython: BaseBytesTest.test_compare
    @make_dynamo_test
    def test_compare(self):
        b1 = self.type2test([1, 2, 3])
        b2 = self.type2test([1, 2, 3])
        b3 = self.type2test([1, 3])

        self.assertEqual(b1, b2)
        self.assertTrue(b2 != b3)
        self.assertTrue(b1 <= b2)
        self.assertTrue(b1 <= b3)
        self.assertTrue(b1 < b3)
        self.assertTrue(b1 >= b2)
        self.assertTrue(b3 >= b2)
        self.assertTrue(b3 > b2)

        self.assertFalse(b1 != b2)
        self.assertFalse(b2 == b3)
        self.assertFalse(b1 > b2)
        self.assertFalse(b1 > b3)
        self.assertFalse(b1 >= b3)
        self.assertFalse(b1 < b2)
        self.assertFalse(b3 < b2)
        self.assertFalse(b3 <= b2)

    # CPython: BaseBytesTest.test_reversed
    @make_dynamo_test
    def test_reversed(self):
        input_data = list(map(ord, "Hello"))
        b = self.type2test(input_data)
        output = list(reversed(b))
        input_data.reverse()
        self.assertEqual(output, input_data)

    # CPython: BaseBytesTest.test_getslice
    @make_dynamo_test
    def test_getslice(self):
        b = self.type2test(b"Hello, world")
        self.assertEqual(b[:5], self.type2test(b"Hello"))
        self.assertEqual(b[1:5], self.type2test(b"ello"))
        self.assertEqual(b[5:7], self.type2test(b", "))
        self.assertEqual(b[7:], self.type2test(b"world"))
        self.assertEqual(b[7:12], self.type2test(b"world"))
        self.assertEqual(b[7:100], self.type2test(b"world"))
        self.assertEqual(b[:-7], self.type2test(b"Hello"))
        self.assertEqual(b[-11:-7], self.type2test(b"ello"))
        self.assertEqual(b[-7:-5], self.type2test(b", "))
        self.assertEqual(b[-5:], self.type2test(b"world"))
        self.assertEqual(b[-5:12], self.type2test(b"world"))
        self.assertEqual(b[-5:100], self.type2test(b"world"))
        self.assertEqual(b[-100:5], self.type2test(b"Hello"))

    # CPython: BaseBytesTest.test_extended_getslice -- full combinatorial
    # is too slow under torch.compile tracing; spot-check a few cases.
    @make_dynamo_test
    def test_extended_getslice(self):
        L = list(range(20))
        b = self.type2test(L)
        self.assertEqual(b[::2], self.type2test(L[::2]))
        self.assertEqual(b[1::2], self.type2test(L[1::2]))
        self.assertEqual(b[::-1], self.type2test(L[::-1]))
        self.assertEqual(b[3:10:3], self.type2test(L[3:10:3]))
        self.assertEqual(b[-1:-10:-1], self.type2test(L[-1:-10:-1]))
        self.assertEqual(b[0:0:1], self.type2test(L[0:0:1]))

    # CPython: BaseBytesTest.test_repeat
    @make_dynamo_test
    def test_repeat(self):
        b = self.type2test(b"abc")
        self.assertEqual(b * 3, b"abcabcabc")
        self.assertEqual(b * 0, b"")
        self.assertEqual(b * -1, b"")
        self.assertRaises(TypeError, lambda: b * 3.14)
        self.assertRaises(TypeError, lambda: 3.14 * b)

    # CPython: BaseBytesTest.test_repeat_1char
    @make_dynamo_test
    def test_repeat_1char(self):
        self.assertEqual(self.type2test(b"x") * 100, self.type2test([ord("x")] * 100))

    # CPython: BaseBytesTest.test_contains
    @make_dynamo_test
    def test_contains(self):
        b = self.type2test(b"abc")
        self.assertIn(ord("a"), b)
        self.assertIn(int(ord("a")), b)
        self.assertNotIn(200, b)

    # CPython: BaseBytesTest.test_concat -- needs sq_concat_impl (PR 2)
    @unittest.expectedFailure
    @make_dynamo_test
    def test_concat(self):
        b1 = self.type2test(b"abc")
        b2 = self.type2test(b"def")
        self.assertEqual(b1 + b2, b"abcdef")
        self.assertRaises(TypeError, lambda: b1 + "def")

    # Iteration -- from BaseBytesTest.test_from_iterable pattern
    @make_dynamo_test
    def test_iter(self):
        b = self.type2test(b"abc")
        result = list(b)
        self.assertEqual(result, [97, 98, 99])

    @make_dynamo_test
    def test_iter_sum(self):
        b = self.type2test(b"abc")
        total = 0
        for byte_val in b:
            total += byte_val
        self.assertEqual(total, 97 + 98 + 99)

    # Bool / truth value
    @make_dynamo_test
    def test_truth(self):
        self.assertFalse(bool(self.type2test()))
        self.assertTrue(bool(self.type2test(b"a")))

    # isinstance
    @make_dynamo_test
    def test_isinstance(self):
        b = self.type2test(b"abc")
        self.assertIsInstance(b, self.type2test)

    # len
    @make_dynamo_test
    def test_len(self):
        b = self.type2test(b"hello")
        self.assertEqual(len(b), 5)
        self.assertEqual(len(self.type2test()), 0)

    # repr -- bytearray-specific
    @make_dynamo_test
    def test_repr(self):
        b = self.type2test(b"abc")
        self.assertEqual(repr(b), repr(self.type2test(b"abc")))

    @make_dynamo_test
    def test_repr_empty(self):
        b = self.type2test()
        self.assertEqual(repr(b), repr(self.type2test()))


class ByteArrayTest(BaseBytesTest, torch._dynamo.test_case.TestCase):
    """bytearray-specific tests, ported from CPython ByteArrayTest."""

    type2test = bytearray

    # CPython: ByteArrayTest -- bytearray is unhashable
    @make_dynamo_test
    def test_unhashable(self):
        self.assertRaises(TypeError, hash, self.type2test(b"abc"))

    # CPython: BaseBytesTest.test_compare_to_str (bytearray != str)
    @make_dynamo_test
    def test_compare_to_str(self):
        self.assertEqual(self.type2test(b"\0a\0b\0c") == "abc", False)
        self.assertEqual(self.type2test() == str(), False)
        self.assertEqual(self.type2test() != str(), True)


if __name__ == "__main__":
    run_tests()
