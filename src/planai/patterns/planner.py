from textwrap import dedent
from typing import List, Optional, Tuple, Type

from llm_interface import LLMInterface
from pydantic import Field

from ..cached_task import CachedTaskWorker
from ..graph import Graph
from ..graph_task import SubGraphWorker
from ..joined_task import InitialTaskWorker, JoinedTaskWorker
from ..llm_task import CachedLLMTaskWorker
from ..task import Task, TaskWorker


class PlanRequest(Task):
    request: str = Field(..., description="The original request to create a plan for")
    request_context: Optional[str] = Field(
        None, description="Additional context for understanding the purpose of the plan"
    )


class PlanDraft(Task):
    plan: str = Field(..., description="The draft plan in markdown format")


class PlanCritique(Task):
    comprehensiveness: float = Field(
        ..., description="Score for how comprehensive the plan is (0-1)"
    )
    detail_orientation: float = Field(
        ..., description="Score for how detailed the plan is (0-1)"
    )
    goal_achievement: float = Field(
        ...,
        description="Score for how well the plan achieves the original request goals (0-1)",
    )
    overall_score: float = Field(..., description="Combined score (0-1)")
    improvement_suggestions: str = Field(
        ..., description="Suggestions for improving the plan"
    )


class RefinementRequest(Task):
    original_request: str = Field(..., description="The original request")
    plans: List[str] = Field(..., description="The plans to be refined")
    critiques: List[PlanCritique] = Field(
        ..., description="The critiques for each plan"
    )


class FinalPlan(Task):
    plan: str = Field(..., description="The final refined plan")
    rationale: str = Field(..., description="Explanation of how the plan was refined")


class PlanEntryWorker(CachedTaskWorker):
    output_types: List[Type[Task]] = [PlanRequest]
    num_variations: int = Field(3, description="Number of plan variations to generate")

    def pre_consume_work(self, task):
        self.notify_status(task, "Creating a comprehensive plan")
        self.print(f"Creating plan for: {task.request}")

    def consume_work(self, task: PlanRequest):
        for _ in range(self.num_variations):
            self.publish_work(task=task.copy_public(), input_task=task)


class PlanCreator(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [PlanDraft]
    llm_input_type: Type[Task] = PlanRequest
    use_xml: bool = True
    prompt: str = dedent(
        """
        Create a detailed plan in markdown format based on the provided request and context.

        The plan should be:
        - Comprehensive and well-structured
        - Detailed and actionable
        - Achieving the goals of the original request

        Provide the plan in markdown format using appropriate headers, lists, and sections.

        Important: Prioritize the instructions provided in the context.
    """
    ).strip()


class PlanCritiqueWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [PlanCritique]
    llm_input_type: Type[Task] = PlanDraft
    use_xml: bool = True
    prompt: str = dedent(
        """
        Evaluate the above plan based on these criteria and the following original request.

        Request: {request}
        Context: {context}

        Score each criterion from 0 (worst) to 1 (best):
        1. Comprehensiveness: How complete and thorough is the plan?
        2. Detail Orientation: How specific and actionable are the steps?
        3. Goal Achievement: How well does the plan fulfill the goals of the original request especially the context?

        Provide improvement suggestions focused on the weakest aspects.

        Output should be JSON with: comprehensiveness, detail_orientation, goal_achievement, overall_score, improvement_suggestions
    """
    ).strip()

    def format_prompt(self, task):
        request: PlanRequest = task.find_input_task(PlanRequest)
        if request is None:
            raise ValueError("PlanRequest not found in critique input tasks")
        return self.prompt.format(
            request=request.request, context=request.request_context
        )

    def post_process(self, response: PlanCritique, input_task: PlanDraft):

        comp = min(1, max(0, response.comprehensiveness))
        detail = min(1, max(0, response.detail_orientation))
        goal = min(1, max(0, response.goal_achievement))

        # weight goal achievement more heavily
        response.overall_score = (0.4 * comp + 0.4 * detail + 0.6 * goal) / 3.0

        self.print(
            f"Plan Critique scored: {response.overall_score:.2f} - {response.improvement_suggestions}"
        )

        return super().post_process(response, input_task)


class PlanCritiqueJoiner(JoinedTaskWorker):
    output_types: List[Type[Task]] = [RefinementRequest]
    join_type: Type[TaskWorker] = InitialTaskWorker

    def consume_work_joined(self, tasks: List[PlanCritique]):
        if not tasks:
            raise ValueError("No critiques to join")

        plans = []
        critiques = []
        original_request = ""

        for critique in tasks:
            plan_draft: PlanDraft = critique.find_input_task(PlanDraft)
            if plan_draft is None:
                raise ValueError("PlanDraft not found in critique input tasks")
            if not original_request:
                plan_request: PlanRequest = critique.find_input_task(PlanRequest)
                if plan_request is None:
                    raise ValueError("PlanRequest not found in critique input tasks")
                original_request = plan_request.request
            plans.append(plan_draft.plan)
            critiques.append(critique)

        self.publish_work(
            RefinementRequest(
                original_request=original_request, plans=plans, critiques=critiques
            ),
            input_task=tasks[0],
        )


class PlanRefinementWorker(CachedLLMTaskWorker):
    output_types: List[Type[Task]] = [FinalPlan]
    llm_input_type: Type[Task] = RefinementRequest
    use_xml: bool = True
    prompt: str = dedent(
        """
        Create a refined, optimized plan by combining the best elements of multiple plans provided above.

        Request: {request}
        Context: {context}

        Create a final plan that:
        1. Incorporates the strongest elements from each plan
        2. Addresses the improvement suggestions from the critiques
        3. Forms a cohesive and comprehensive solution

        Provide your response as JSON with:
        - plan: The final refined plan in markdown format
        - rationale: Brief explanation of how you combined and improved the plans
    """
    ).strip()

    def format_prompt(self, task):
        request: PlanRequest = task.find_input_task(PlanRequest)
        if request is None:
            raise ValueError("PlanRequest not found in critique input tasks")
        return self.prompt.format(
            request=request.request, context=request.request_context
        )


class SimplePlanAdaptor(TaskWorker):
    input_types: List[Type[Task]] = [PlanDraft]
    output_types: List[Type[Task]] = [FinalPlan]

    def consume_work(self, task: PlanDraft):
        self.publish_work(
            FinalPlan(
                plan=task.plan, rationale="This is the draft plan without refinement"
            ),
            input_task=task,
        )


def create_planning_graph(
    llm: LLMInterface, name: str = "PlanningWorker", num_variations: int = 3
) -> Tuple[Graph, TaskWorker, TaskWorker]:
    """Creates a planning graph with multiple workers for plan generation and refinement.

    This function sets up a directed graph of workers that collaborate to create and refine plans.
    The graph includes workers for plan entry, creation, critique, joining critiques, and refinement.

    Args:
        llm (LLMInterface): Language model interface used by the workers
        name (str, optional): Base name for the graph. Defaults to "PlanningWorker"
        num_variations (int, optional): Number of plan variations to generate. Defaults to 3
            When set to 0, it will produce a simple plan without going through refinement

    Returns:
        Tuple[Graph, TaskWorker, TaskWorker]: A tuple containing:
            - The constructed planning graph
            - The entry worker node
            - The refinement worker node
    """
    graph = Graph(name=f"{name}Graph", strict=True)

    entry = PlanEntryWorker(num_variations=num_variations)
    creator = PlanCreator(llm=llm)
    critique = PlanCritiqueWorker(llm=llm)
    joiner = PlanCritiqueJoiner()
    refinement = PlanRefinementWorker(llm=llm)

    graph.add_workers(entry, creator, critique, joiner, refinement)

    graph.set_dependency(entry, creator).next(critique).next(joiner).next(refinement)

    return graph, entry, refinement


def create_simple_planning_graph(
    llm: LLMInterface, name: str = "SimplePlanningWorker"
) -> Tuple[Graph, TaskWorker, TaskWorker]:
    """Creates a simple planning graph with a single worker for plan generation.

    This function sets up a directed graph of workers that generates a plan based on a request.

    Args:
        llm (LLMInterface): Language model interface used by the workers
        name (str, optional): Base name for the graph. Defaults to "PlanningWorker"

    Returns:
        Tuple[Graph, TaskWorker, TaskWorker]: A tuple containing:
            - The constructed planning graph
            - The entry worker node
            - The refinement worker node
    """
    graph = Graph(name=f"{name}Graph", strict=True)

    entry = PlanEntryWorker(num_variations=1)
    creator = PlanCreator(llm=llm)
    adaptor = SimplePlanAdaptor()

    graph.add_workers(entry, creator, adaptor)

    graph.set_dependency(entry, creator).next(adaptor)

    return graph, entry, adaptor


def create_planning_worker(
    llm: LLMInterface, name: str = "PlanningWorker", num_variations: int = 2
) -> TaskWorker:
    """Creates a SubGraphWorker for plan generation and refinement.

    This worker creates a subgraph that:
    1. Generates multiple plan variations
    2. Critiques each plan
    3. Combines the best elements into a final plan

    Args:
        llm: LLM interface for plan generation and analysis
        name: Name for the worker
        num_variations: Number of plan variations to generate. If num_varations is 0, it will use a simple plan adaptor

    Input Task:
        PlanRequest: Task containing the original request to create a plan for

    Output Task:
        FinalPlan: The refined final plan with rationale
    """
    if num_variations == 0:
        graph, entry, refinement = create_simple_planning_graph(llm=llm, name=name)
    else:
        graph, entry, refinement = create_planning_graph(
            llm=llm, name=name, num_variations=num_variations
        )

    return SubGraphWorker(
        name=name, graph=graph, entry_worker=entry, exit_worker=refinement
    )
