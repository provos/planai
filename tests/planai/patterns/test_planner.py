import unittest
from typing import Optional

from planai import Graph, TaskWorker
from planai.patterns.planner import (
    FinalPlan,
    PlanCritique,
    PlanDraft,
    PlanRequest,
    RefinementRequest,
    create_planning_graph,
    create_planning_worker,
)
from planai.testing import (
    MockCache,
    MockLLM,
    MockLLMResponse,
    inject_mock_cache,
    unregister_output_type,
)


class TestPlanner(unittest.TestCase):
    def setUp(self):
        # Set up mock cache
        self.mock_cache = MockCache(dont_store=True)

        # Set up mock LLM with different responses for each worker type
        self.mock_llm = MockLLM(
            responses=[
                # PlanCreator responses
                MockLLMResponse(
                    pattern="Create a detailed plan.*",
                    response=PlanDraft(
                        plan="# Test Plan\n## Steps\n1. First step\n2. Second step"
                    ),
                ),
                # PlanCritiqueWorker responses
                MockLLMResponse(
                    pattern=".*Score each criterion from 0 .worst. to 1 .best.*",
                    response=PlanCritique(
                        comprehensiveness=0.8,
                        detail_orientation=0.7,
                        goal_achievement=0.9,
                        overall_score=0.8,
                        improvement_suggestions="Add more detail to step two",
                    ),
                ),
                # PlanRefinementWorker responses
                MockLLMResponse(
                    pattern="Create a refined, optimized plan.*",
                    response=FinalPlan(
                        plan="# Refined Plan\n## Steps\n1. Detailed first step\n2. Enhanced second step",
                        rationale="Combined best elements and added detail",
                    ),
                ),
            ]
        )

    def test_planning_workflow(self):
        # Create main graph and inject mock cache
        graph = Graph(name="TestGraph")
        planning = create_planning_worker(llm=self.mock_llm, name="TestPlanning")
        graph.add_workers(planning)
        graph.set_sink(planning, FinalPlan)
        inject_mock_cache(graph, self.mock_cache)

        # Create initial request
        request = PlanRequest(request="Create a plan for testing")
        initial_work = [(planning, request)]

        # Run the graph
        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Get output tasks
        output_tasks = graph.get_output_tasks()

        # Should have one final plan
        self.assertEqual(len(output_tasks), 1)
        final_plan = output_tasks[0]
        self.assertIsInstance(final_plan, FinalPlan)
        self.assertTrue("Refined Plan" in final_plan.plan)
        self.assertTrue(final_plan.rationale)

    def test_planning_graph_workflow(self):
        # Create graph using the plain graph version
        graph, entry_worker, exit_worker = create_planning_graph(
            llm=self.mock_llm, name="TestPlanning", num_variations=2
        )
        graph.set_sink(exit_worker, FinalPlan)

        # Inject mock cache into graph
        inject_mock_cache(graph, self.mock_cache)

        # Create initial request
        request = PlanRequest(request="Create a plan for testing")
        initial_work = [(entry_worker, request)]

        # Run the graph
        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Get output tasks
        output_tasks = graph.get_output_tasks()

        # Should have one final plan
        self.assertEqual(len(output_tasks), 1)
        final_plan = output_tasks[0]
        self.assertIsInstance(final_plan, FinalPlan)

    def test_plan_variations(self):
        # Test that correct number of variations are generated
        graph, entry_worker, exit_worker = create_planning_graph(
            llm=self.mock_llm, name="TestPlanning", num_variations=3
        )

        # Add a sink to capture PlanDraft tasks
        planner: Optional[TaskWorker] = graph.get_worker_by_output_type(PlanDraft)
        assert planner is not None
        unregister_output_type(planner, PlanDraft)
        graph.set_sink(planner, PlanDraft)
        inject_mock_cache(graph, self.mock_cache)

        # Create and run request
        request = PlanRequest(request="Create a plan for testing")
        initial_work = [(entry_worker, request)]
        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Get PlanDraft tasks
        draft_tasks = [t for t in graph.get_output_tasks() if isinstance(t, PlanDraft)]
        self.assertEqual(len(draft_tasks), 3)

    def test_critique_joining(self):
        graph, entry_worker, exit_worker = create_planning_graph(
            llm=self.mock_llm, name="TestPlanning", num_variations=2
        )

        # Add a sink to capture RefinementRequest tasks
        refiner = graph.get_worker_by_output_type(RefinementRequest)
        assert refiner is not None
        unregister_output_type(refiner, RefinementRequest)
        graph.set_sink(refiner, RefinementRequest)
        inject_mock_cache(graph, self.mock_cache)

        # Create and run request
        request = PlanRequest(request="Create a plan for testing")
        initial_work = [(entry_worker, request)]
        graph.run(
            initial_tasks=initial_work, run_dashboard=False, display_terminal=False
        )

        # Get RefinementRequest tasks
        refinement_tasks = [
            t for t in graph.get_output_tasks() if isinstance(t, RefinementRequest)
        ]
        self.assertEqual(len(refinement_tasks), 1)

        # Verify the refinement request contains all plans and critiques
        refinement = refinement_tasks[0]
        self.assertEqual(len(refinement.plans), 2)
        self.assertEqual(len(refinement.critiques), 2)
        self.assertTrue(all(isinstance(c, PlanCritique) for c in refinement.critiques))


if __name__ == "__main__":
    unittest.main()
