"""
Perform validation of a patch, on a given task instance.
"""

import ast
import itertools
import os
import shlex
import shutil
import subprocess
import tempfile
from os import PathLike
from os.path import join as pjoin
from pathlib import Path
from subprocess import PIPE, STDOUT
from typing import Tuple

from unidiff import PatchSet

from app import utils as app_utils
from app.analysis.sbfl import MethodId, method_ranges_in_file
from app.api import execution
from app.log import log_and_print


def validate(
    patch_file_path,
    repo_name,
    output_dir,  # the result dir for this task
    project_path,  # the actual code for the task project
    test_cmd,
    env_name,
    testcases_passing,
    testcases_failing,
    run_test_suite_log_file,
    logger,
) -> Tuple[bool, str]:
    """
    Returns:
        - Whether this patch has made the test suite pass.
        - Error message when running the test suite.
    """
    # (1) apply the patch to source code
    with app_utils.cd(project_path):
        apply_cmd = ["git", "apply", patch_file_path]
        cp = app_utils.run_command(logger, apply_cmd, capture_output=False, text=True)
        if cp.returncode != 0:
            # patch application failed
            raise RuntimeError(f"Error applying patch: {cp.stderr}")

    # (2) run the modified program against the test suite
    log_and_print(logger, "[Validation] Applied patch. Going to run test suite.")
    tests_passed, msg = execution.run_test_suite_for_correctness(
        repo_name,
        output_dir,
        project_path,
        test_cmd,
        env_name,
        testcases_passing,
        testcases_failing,
        run_test_suite_log_file,
        logger,
    )

    # (3) revert the patch to source code
    with app_utils.cd(project_path):
        app_utils.repo_clean_changes()

    log_and_print(logger, f"[Validation] Finishing. Result is {tests_passed}. Message: {msg}.")
    return tests_passed, msg


def perfect_angelic_debug(
    task_id: str,
    diff_file: str,
    project_path: str
) -> tuple[set, set, set]:
    """Do perfect angelic debugging and return a list of incorrect fix locations.

    Args:
        task_id: the task id, used to find developer patch
        diff_file: path of diff file

    Returns:
        A list of (filename, MethodId) that should not have been changed by diff_file
    """
    return compare_fix_locations(diff_file, get_developer_patch_file(task_id), project_path)


def compare_fix_locations(
    diff_file: str,
    dev_diff_file: str,
    project_path: str
) -> tuple[set, set, set]:
    """Compare the changed methods in two diff files

    Args:
        diff_file: path to diff file
        dev_diff_file: path to a "correct" diff file

    Returns:
        list of (filename, MethodId) that are changed in diff_file but not in dev_diff_file
    """
    methods_map = get_changed_methods(diff_file, project_path)
    dev_methods_map = get_changed_methods(dev_diff_file, project_path)

    methods_set = set(
        itertools.chain.from_iterable(
            [(k, method_id) for method_id in v] for k, v in methods_map.items()
        )
    )
    dev_methods_set = set(
        itertools.chain.from_iterable(
            [(k, method_id) for method_id in v] for k, v in dev_methods_map.items()
        )
    )

    return (
        methods_set - dev_methods_set,
        methods_set & dev_methods_set,
        dev_methods_set - methods_set,
    )


def get_developer_patch_file(task_id: str) -> str:
    processed_data_lite = Path(__file__).parent.parent.with_name("processed_data_lite")
    dev_patch_file = Path(processed_data_lite, "test", task_id, "developer_patch.diff").resolve()
    if not dev_patch_file.is_file():
        raise RuntimeError(f"Failed to find developer patch at {dev_patch_file!s}")
    return str(dev_patch_file)


def get_method_id(file: str, line: int) -> MethodId | None:
    ranges = method_ranges_in_file(file)
    for method_id, (lower, upper) in ranges.items():
        if lower <= line <= upper:
            return method_id
    return None


def get_changed_methods(diff_file: str, project_path: str = "") -> dict[str, set[MethodId]]:
    with open(diff_file, "r") as f:
        patch_content = f.read()

    changed_files = []

    patch = PatchSet(patch_content)
    for file in patch:
        file_name = file.source_file.removeprefix("a/").removeprefix("b/")
        changed_files.append(file_name)

    orig_definitions: dict[tuple[str, MethodId], str] = {}
    for file in changed_files:
        def_map = collect_method_definitions(Path(project_path, file))

        for method_id, definition in def_map.items():
            orig_definitions[(file, method_id)] = definition

    temp_dir = tempfile.mkdtemp(dir="/tmp", prefix="apply_patch_")
    for file in changed_files:
        copy_path = Path(temp_dir, file)
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(project_path, file), copy_path)
    patch_cmd = f"patch -p1 -f -i {diff_file}"
    cp = subprocess.run(shlex.split(patch_cmd), cwd=temp_dir, stdout=PIPE, stderr=STDOUT, text=True)
    if cp.returncode != 0:
        raise RuntimeError(
            f"Patch command in directory {temp_dir} exit with {cp.returncode}: {patch_cmd}"
        )

    new_definitions: dict[tuple[str, MethodId], str] = {}
    for file in changed_files:
        def_map = collect_method_definitions(Path(temp_dir, file))

        for method_id, definition in def_map.items():
            new_definitions[(file, method_id)] = definition

    shutil.rmtree(temp_dir)

    result = {}
    for key, definition in orig_definitions.items():
        if new_definitions.get(key, "") != definition:
            file, method_id = key
            result[file] = result.get(file, set()) | {method_id}

    return result


def collect_method_definitions(file: str | PathLike) -> dict[MethodId, str]:
    if not str(file).endswith(".py"):
        return {}

    collector = MethodDefCollector()

    source = Path(file).read_text()
    tree = ast.parse(source, file)

    collector.visit(tree)
    return collector.def_map


class MethodDefCollector(ast.NodeVisitor):
    def __init__(self):
        self.def_map: dict[MethodId, str] = {}
        self.class_name = ""

    def calc_method_id(self, method_name: str) -> MethodId:
        return MethodId(self.class_name, method_name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_name = node.name
        super().generic_visit(node)
        self.class_name = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        method_id = self.calc_method_id(node.name)
        self.def_map[method_id] = ast.unparse(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        method_id = self.calc_method_id(node.name)
        self.def_map[method_id] = ast.unparse(node)
