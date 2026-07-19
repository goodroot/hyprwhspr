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


if __name__ == "__main__":
    unittest.main()
