from __future__ import annotations

import unittest

from dlef.registry import Registry, RegistryError


class _Thing:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class _OtherThing:
    pass


class RegistryTests(unittest.TestCase):
    def test_register_create_resolve(self) -> None:
        registry = Registry()
        registry.register("thing", "thing", aliases=("t",))(_Thing)
        instance = registry.create("thing", {"name": "thing", "value": 7}, expected_type=_Thing)
        self.assertEqual(instance.value, 7)
        self.assertIs(registry.resolve("thing", "t"), _Thing)
        self.assertIn("thing", registry.available("thing"))

    def test_duplicate(self) -> None:
        registry = Registry()
        registry.register("thing", "thing")(_Thing)
        with self.assertRaises(RegistryError):
            registry.register("thing", "thing")(_OtherThing)

    def test_unknown(self) -> None:
        registry = Registry()
        with self.assertRaises(RegistryError):
            registry.create("thing", {"name": "missing"})

    def test_class_path(self) -> None:
        registry = Registry()
        instance = registry.create("loss", {"class_path": "examples.toy.components:MSELoss"})
        self.assertTrue(callable(instance))


if __name__ == "__main__":
    unittest.main()
