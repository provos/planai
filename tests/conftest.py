# Copyright 2024 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# tests/conftest.py
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

LOGGING = True


def pytest_configure(config):
    config.addinivalue_line("markers", "regression: marks tests as regression tests")
    config.addinivalue_line(
        "markers",
        "timeout(seconds): mark test to timeout after given number of seconds",
    )

    if LOGGING:
        logging.basicConfig(
            format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )


def pytest_addoption(parser):
    parser.addoption(
        "--run-regression",
        action="store_true",
        default=False,
        help="run regression tests",
    )

    parser.addoption(
        "--provider", action="store", default="ollama", help="LLM provider to use"
    )
    parser.addoption(
        "--model", action="store", default="llama3.2", help="Model name to use"
    )
    parser.addoption("--host", action="store", default=None, help="Host for Ollama")
    parser.addoption(
        "--hostname",
        action="store",
        default=None,
        help="SSH hostname (for remote Ollama)",
    )
    parser.addoption(
        "--username",
        action="store",
        default=None,
        help="SSH username (for remote Ollama)",
    )
    parser.addoption(
        "--test-timeout",
        action="store",
        type=int,
        default=30,  # 30 seconds default timeout
        help="Default test timeout in seconds",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-regression"):
        skip_regression = pytest.mark.skip(reason="need --run-regression option to run")
        for item in items:
            if "regression" in item.keywords:
                item.add_marker(skip_regression)

    timeout = config.getoption("--test-timeout")

    for item in items:
        # Skip if test already has a timeout marker
        if not any(marker.name == "timeout" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.timeout(timeout))
