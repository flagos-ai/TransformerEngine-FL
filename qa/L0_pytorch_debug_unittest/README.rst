L0 PyTorch Debug Unittest
=========================

This directory contains the L0 PyTorch debug unittest runner.

MetaX ignore rules
------------------

MetaX-specific ignored tests are maintained in one place in ``test.sh`` through
the ``METAX_IGNORED_TESTS`` list.

The main execution flow only calls a helper to decide whether a test should be
skipped, instead of embedding platform-specific matching rules directly in the
main logic.

This keeps the script easier to maintain and makes it simpler to add new
ignored cases later if needed.

How to extend
-------------

If a new test needs to be skipped on MetaX:

1. Add the full test path to ``METAX_IGNORED_TESTS`` in ``test.sh``.
2. Avoid adding new platform-specific matching logic directly into the main
   execution flow.