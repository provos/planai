"""
Common workflow patterns for PlanAI.

This module provides reusable workflow patterns that can be used to build
complex task processing pipelines.
"""

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
# src/planai/__init__.py

from .planner import FinalPlan, PlanRequest, create_planning_worker
from .search_fetch import (
    ConsolidatedPages,
    SearchQuery,
    SearchResult,
    create_search_fetch_worker,
)

__all__ = [
    "create_search_fetch_worker",
    "SearchQuery",
    "SearchResult",
    "ConsolidatedPages",
    "create_planning_worker",
    "PlanRequest",
    "FinalPlan",
]
