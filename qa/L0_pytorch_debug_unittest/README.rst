L0 PyTorch Debug Unittest
=========================

This directory contains the L0 PyTorch debug unittest runner.

Platform ignore rules
---------------------

``test.sh`` runs the same steps on every platform and does not filter tests.
A test that cannot pass on one platform is ignored in that platform's config,
``.github/configs/<platform>.yml``, under ``ignored_tests``, with the reason,
owner and restore condition in a comment above the entry. The CI runner passes
those entries to every pytest call in ``test.sh``, and the plan lists them as
coverage gaps.

See "Ignoring a test on one platform" in ``.github/scripts/ci/README.md`` for
the entry forms and for scoping an entry to the ``pytorch_debug`` job only.
