import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleasePackagingContractTests(unittest.TestCase):
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
