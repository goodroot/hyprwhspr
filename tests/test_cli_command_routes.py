import argparse
import ast
import importlib.util
import re
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "lib" / "cli.py"
CLI_SPEC = importlib.util.spec_from_file_location("hyprwhspr_cli_entrypoint", CLI_PATH)
cli = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(cli)


class _ParserCaptured(Exception):
    pass


class CliCommandRouteTests(unittest.TestCase):
    def test_top_level_commands_match_all_routing_surfaces(self):
        captured = {}

        def capture(parser):
            captured["parser"] = parser
            raise _ParserCaptured

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with self.assertRaises(_ParserCaptured):
                cli.main()
        parser = captured["parser"]
        subparsers = next(
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        argparse_commands = set(subparsers.choices)

        launcher = (ROOT / "bin" / "hyprwhspr").read_text(encoding="utf-8")
        match = re.search(r'\^\(([^)]+)\)\$.*then', launcher)
        self.assertIsNotNone(match)
        launcher_commands = set(match.group(1).split("|"))

        main_tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))
        fallback_commands = None
        for node in ast.walk(main_tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "CLI_SUBCOMMANDS"
                    for target in node.targets):
                fallback_commands = set(ast.literal_eval(node.value))
        self.assertIsNotNone(fallback_commands)
        self.assertEqual(argparse_commands, launcher_commands)
        self.assertEqual(argparse_commands, fallback_commands)


if __name__ == "__main__":
    unittest.main()
