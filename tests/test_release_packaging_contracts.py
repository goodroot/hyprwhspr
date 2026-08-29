import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_payload_validator():
    path = ROOT / "scripts" / "validate-package-payload.py"
    spec = importlib.util.spec_from_file_location("validate_package_payload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleasePackagingContractTests(unittest.TestCase):
    def test_payload_validator_reads_plans_built_from_shared_constants(self):
        # PLAN_SPECS entries are assembled from CORE_IMPORTS, so the validator
        # must not depend on the whole entry being a literal
        validator = _load_payload_validator()
        manifests = validator._plan_manifests(ROOT)
        self.assertTrue(manifests)
        for manifest in manifests:
            self.assertTrue(manifest.is_file(), manifest)

    def test_bump_script_validates_payload_and_preserves_recipe_body(self):
        script = (ROOT / "bump-version.sh").read_text(encoding="utf-8")
        self.assertIn('scripts/validate-package-payload.py', script)
        self.assertIn('bash -n scripts/install.sh scripts/install-deps.sh', script)
        self.assertIn('git status --porcelain', script)
        self.assertNotIn('cp ', script, "version bump must not replace the AUR recipe")
        self.assertIn('s/^pkgver=.*/pkgver=$NEW_VERSION/', script)
        self.assertIn('s/^pkgrel=.*/pkgrel=1/', script)
        self.assertIn("s/sha256sums=('.*')/sha256sums=('$NEW_SHA256')/", script)

    def test_gtk4_layer_shell_workflow_uses_installer_contract(self):
        workflow = (ROOT / ".github" / "workflows" / "build-gtk4-layer-shell.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("import visualizer_runtime as v", workflow)
        self.assertIn("GTK4_LAYER_SHELL_COMMIT", workflow)
        self.assertIn("ubuntu-24.04", workflow)
        self.assertIn("Check immutable release", workflow)
        self.assertIn("sha256sum", workflow)
        build, publish = workflow.split("  publish:", 1)
        self.assertNotIn("sha256sum --check", build)
        self.assertIn("artifact_run_id", publish)
        self.assertIn("sha256sum --check", publish)

    def test_uninstall_includes_optional_gui_runtime(self):
        source = (ROOT / "lib" / "src" / "cli" / "uninstall.py").read_text(encoding="utf-8")
        self.assertIn("runtime_dir = USER_BASE / 'runtime'", source)
        self.assertIn("Optional GUI runtimes", source)


if __name__ == "__main__":
    unittest.main()
