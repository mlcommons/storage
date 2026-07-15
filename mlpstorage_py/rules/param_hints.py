"""Known-typo hints for DLIO ``--params`` keys.

When a user types a param path that's syntactically similar to a real DLIO
config key but is missing an intermediate node (a common mistake for the
nested ``storage.storage_options.*`` family), fail fast at CLI parse time
with the canonical spelling. Also used as a fallback hint in the runtime
and submission-checker disallowed-override messages so users who hit those
paths (e.g. re-validating an older result) still see the suggestion.

See storage#795 for the reporter case: a user passed
``--params storage.storage_library=s3dlio``, expecting it to configure
DLIO's storage backend. It did not — DLIO's config uses the nested
``storage.storage_options.storage_library`` key, and the flat form was
a stray parameter that a) had zero effect on the run (the tool auto-
injects the nested key from ``STORAGE_LIBRARY`` env in ``--object`` mode)
and b) tripped the CLOSED-mode disallowed-override check.

The map is intentionally small: only paths that have been seen typed
incorrectly by real users. Do not add every conceivable typo — the
value of this hint depends on it staying targeted, not exhaustive.
"""

# Map of user-typed dotted key -> canonical DLIO dotted key.
KNOWN_PARAM_TYPOS = {
    # storage#795: reporter dropped the ``storage_options.`` middle segment
    # for all three ``storage.storage_options.*`` keys. All three are auto-
    # injected by ``_apply_object_storage_params`` in ``--object`` mode, so
    # the user typically does not need to pass them at all.
    'storage.storage_library': 'storage.storage_options.storage_library',
    'storage.uri_scheme': 'storage.storage_options.uri_scheme',
    'storage.prefetch_window': 'storage.storage_options.prefetch_window',
}


def format_typo_hint(user_key: str) -> str:
    """Return a ``' Did you mean X?'`` suffix for known-typo keys, else ''.

    Callers append this to their existing error message so the hint reads as
    a natural continuation. Returns the empty string for unknown keys so
    the caller can concatenate unconditionally.
    """
    canonical = KNOWN_PARAM_TYPOS.get(user_key)
    if not canonical:
        return ''
    return f" Did you mean '{canonical}'?"
