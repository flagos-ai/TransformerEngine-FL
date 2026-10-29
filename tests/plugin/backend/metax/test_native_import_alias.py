# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Exercise the actual native importer without importing TE, Torch, or a GPU.

The extension's PyInit name and the TE-FL frontend both use
``transformer_engine_torch``.  Native builds differ in whether exec_module
registers that name, so restoring the frontend must not depend on that behavior.
"""

import ast
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import tempfile
import unittest


_FRONTEND_NAME = "transformer_engine_torch"
_NATIVE_NAME = "transformer_engine_torch_metax"


def _load_importer(namespace):
    """Compile only the real importer, retaining its original implementation."""
    repo_root = Path(__file__).resolve().parents[4]
    default_source = repo_root / "transformer_engine/plugin/core/backends/vendor/metax/metax.py"
    source = Path(os.environ.get("METAX_SOURCE", str(default_source)))
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_import_tex_module"
    ]
    assert len(functions) == 1, f"Expected one _import_tex_module in {source}"
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_import_tex_module"]


class _NativeLoadError(RuntimeError):
    pass


def _make_environment(tmp_path, owner_present, self_register, fail=False):
    home = tmp_path / "vendor"
    package = home / "transformer_engine"
    package.mkdir(parents=True)
    extension_path = package / "transformer_engine_torch.test.so"
    extension_path.touch()

    owner = ModuleType("tefl_frontend")
    native = ModuleType("native_extension")
    modules = {_FRONTEND_NAME: owner} if owner_present else {}
    old_flags = 17
    active_flags = old_flags
    flag_changes = []
    calls = {"spec": 0, "module": 0, "exec": 0}
    fake_os = SimpleNamespace(environ={"TE_METAX_TE_HOME": str(home)}, RTLD_LAZY=1, RTLD_GLOBAL=256)
    native_flags = fake_os.RTLD_LAZY | fake_os.RTLD_GLOBAL

    def set_flags(flags):
        nonlocal active_flags
        active_flags = flags
        flag_changes.append(flags)

    def exec_module(module):
        calls["exec"] += 1
        assert module is native
        assert active_flags == native_flags
        # The frontend must be out of the way while the native loader runs.
        assert _FRONTEND_NAME not in modules
        if self_register:
            modules[_FRONTEND_NAME] = native
        if fail:
            raise _NativeLoadError("native initialization failed")

    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=exec_module))

    def spec_from_file_location(name, path):
        calls["spec"] += 1
        assert name == _FRONTEND_NAME
        assert Path(path) == extension_path
        assert active_flags == native_flags
        return spec

    def module_from_spec(received_spec):
        calls["module"] += 1
        assert received_spec is spec
        return native

    fake_sys = SimpleNamespace(
        modules=modules, getdlopenflags=lambda: active_flags, setdlopenflags=set_flags
    )
    fake_util = SimpleNamespace(
        find_spec=lambda name: None,
        spec_from_file_location=spec_from_file_location,
        module_from_spec=module_from_spec,
    )
    namespace = {
        "Path": Path,
        "os": fake_os,
        "sys": fake_sys,
        "importlib": SimpleNamespace(util=fake_util),
    }
    return SimpleNamespace(
        importer=_load_importer(namespace),
        modules=modules,
        owner=owner,
        native=native,
        old_flags=old_flags,
        native_flags=native_flags,
        flags=lambda: active_flags,
        flag_changes=flag_changes,
        calls=calls,
    )


def _assert_frontend_restored(environment, owner_present):
    if owner_present:
        assert environment.modules[_FRONTEND_NAME] is environment.owner
    else:
        assert _FRONTEND_NAME not in environment.modules
    assert environment.flags() == environment.old_flags
    assert environment.flag_changes == [
        environment.native_flags,
        environment.old_flags,
    ]


class TestNativeImportAlias(unittest.TestCase):
    def _check_success(self, owner_present, self_register):
        with tempfile.TemporaryDirectory() as directory:
            environment = _make_environment(Path(directory), owner_present, self_register)

            self.assertIs(environment.importer(), environment.native)
            _assert_frontend_restored(environment, owner_present)
            self.assertIs(environment.modules[_NATIVE_NAME], environment.native)
            self.assertEqual(environment.calls, {"spec": 1, "module": 1, "exec": 1})

            # A second import must not touch the native loader or dlopen flags.
            self.assertIs(environment.importer(), environment.native)
            _assert_frontend_restored(environment, owner_present)
            self.assertEqual(environment.calls, {"spec": 1, "module": 1, "exec": 1})

    def _check_failure(self, owner_present, self_register):
        with tempfile.TemporaryDirectory() as directory:
            environment = _make_environment(
                Path(directory), owner_present, self_register, fail=True
            )

            with self.assertRaisesRegex(_NativeLoadError, "native initialization failed"):
                environment.importer()

            _assert_frontend_restored(environment, owner_present)
            self.assertNotIn(_NATIVE_NAME, environment.modules)
            self.assertEqual(environment.calls, {"spec": 1, "module": 1, "exec": 1})

    def test_success_without_owner_or_native_registration(self):
        self._check_success(owner_present=False, self_register=False)

    def test_success_without_owner_with_native_registration(self):
        self._check_success(owner_present=False, self_register=True)

    def test_success_with_owner_without_native_registration(self):
        self._check_success(owner_present=True, self_register=False)

    def test_success_with_owner_and_native_registration(self):
        self._check_success(owner_present=True, self_register=True)

    def test_failure_without_owner_or_native_registration(self):
        self._check_failure(owner_present=False, self_register=False)

    def test_failure_without_owner_with_native_registration(self):
        self._check_failure(owner_present=False, self_register=True)

    def test_failure_with_owner_without_native_registration(self):
        self._check_failure(owner_present=True, self_register=False)

    def test_failure_with_owner_and_native_registration(self):
        self._check_failure(owner_present=True, self_register=True)


if __name__ == "__main__":
    unittest.main()
