#!/usr/bin/env python3
"""
Tests for mlpstorage_py.submission_checker.rule_registry module.

Covers the @rule decorator and discover_rules introspection helper.

Run with:
    pytest mlpstorage_py/tests/test_rule_registry.py -v
"""

import inspect
import pytest

from mlpstorage_py.submission_checker.rule_registry import rule, discover_rules


class TestRuleDecorator:
    """Tests for the @rule decorator factory."""

    def test_decorator_sets_rule_id(self):
        """Decorated function has __rule_id__ equal to the supplied rule_id."""
        @rule("2.1.2", "topLevelSubdirectories")
        def some_check(self):
            return True

        assert some_check.__rule_id__ == "2.1.2"

    def test_decorator_sets_rule_name(self):
        """Decorated function has __rule_name__ equal to the supplied rule_name."""
        @rule("2.1.2", "topLevelSubdirectories")
        def some_check(self):
            return True

        assert some_check.__rule_name__ == "topLevelSubdirectories"

    def test_decorator_returns_function_unchanged_signature(self):
        """Decorated function has the same inspect.signature as the original."""
        def original(self, a, b, c=3):
            return a + b + c

        original_sig = inspect.signature(original)
        decorated = rule("X", "Y")(original)
        assert inspect.signature(decorated) == original_sig

    def test_decorator_returns_function_unchanged_return_value(self):
        """Calling the decorated function yields the same return value as the original."""
        @rule("2.1.3", "openMatchesClosed")
        def compute(x, y):
            return x * y

        assert compute(3, 7) == 21

    def test_decorator_preserves_self_binding_as_method(self):
        """A @rule-decorated method on a class can be called via an instance (self binds correctly)."""
        class SomeCheck:
            def __init__(self):
                self.value = 42

            @rule("2.1.5", "requiredSubdirectories")
            def my_check(self):
                return self.value

        obj = SomeCheck()
        assert obj.my_check() == 42


class TestDiscoverRules:
    """Tests for the discover_rules introspection helper."""

    def test_discover_rules_returns_correct_count(self):
        """discover_rules returns exactly the number of @rule-decorated methods."""
        class ThreeRuleClass:
            @rule("2.1.1", "submitterRootDirectory")
            def check_a(self):
                pass

            @rule("2.1.2", "topLevelSubdirectories")
            def check_b(self):
                pass

            @rule("2.1.3", "openMatchesClosed")
            def check_c(self):
                pass

            def plain_method(self):
                pass

            def another_plain(self):
                pass

        result = discover_rules(ThreeRuleClass)
        assert len(result) == 3

    def test_discover_rules_keyed_by_rule_id(self):
        """discover_rules result maps rule_id -> (rule_name, method_name)."""
        class SomeClass:
            @rule("2.1.2", "topLevelSubdirectories")
            def top_level_subdirectories_check(self):
                pass

        result = discover_rules(SomeClass)
        assert "2.1.2" in result
        assert result["2.1.2"] == ("topLevelSubdirectories", "top_level_subdirectories_check")

    def test_discover_rules_empty_on_no_decorated_methods(self):
        """discover_rules returns an empty dict when no methods are @rule-decorated."""
        class NoRulesClass:
            def plain_one(self):
                pass

            def plain_two(self):
                pass

        result = discover_rules(NoRulesClass)
        assert result == {}

    def test_discover_rules_does_not_raise_with_non_callable_attributes(self):
        """discover_rules does not raise when the class has non-callable attributes (class vars, properties)."""
        class MixedClass:
            class_var = "hello"
            number_var = 42
            list_var = [1, 2, 3]

            @property
            def my_prop(self):
                return self.class_var

            @rule("3.1.1", "someRule")
            def rule_method(self):
                pass

        result = discover_rules(MixedClass)
        assert len(result) == 1
        assert "3.1.1" in result
