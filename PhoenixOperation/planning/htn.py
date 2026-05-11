from __future__ import annotations

from planning.pddl import Action, Problem, apply_action, is_applicable
from planning.utils import Queue


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    ### Your code here ###

    if not hlas:
        return []

    frontier = Queue()

    for hla in hlas:
        frontier.push([hla])

    visited = set()

    while not frontier.isEmpty():

        plan = frontier.pop()

        signature = tuple(step.name for step in plan)

        if signature in visited:
            continue

        visited.add(signature)

        if is_plan_primitive(plan):

            state = problem.initial_state
            valid = True

            for action in plan:

                if not is_applicable(state, action):
                    valid = False
                    break

                state = apply_action(state, action)

            if valid and problem.isGoalState(state):
                return plan

            continue

        first_hla_index = None

        for i, step in enumerate(plan):

            if not is_primitive(step):
                first_hla_index = i
                break

        if first_hla_index is None:
            continue

        hla = plan[first_hla_index]

        for refinement in hla.refinements:

            new_plan = (
                plan[:first_hla_index]
                + refinement
                + plan[first_hla_index + 1:]
            )

            frontier.push(new_plan)

    return []

    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    ### Your code here ###

    state = problem.initial_state

    if not hasattr(problem, "_all_groundings"):
        from planning.pddl import get_all_groundings
        problem._all_groundings = get_all_groundings(
            problem.domain,
            problem.objects
        )

    robot = problem.objects["robots"][0]

    patients = problem.objects["patients"]
    supplies = problem.objects["supplies"]

    medical_posts = []

    for fluent in state:
        if fluent[0] == "MedicalPost":
            medical_posts.append(fluent[1])

    if not medical_posts:
        return []

    medical_post = medical_posts[0]

    def get_location(obj_name):

        for fluent in state:

            if fluent[0] == "At" and fluent[1] == obj_name:
                return fluent[2]

        return None

    def get_robot_location():
        return get_location(robot)

    def find_move_action(from_cell, to_cell):

        for action in problem._all_groundings:

            if not action.name.startswith("Move"):
                continue

            needed = {
                ("At", robot, from_cell),
                ("Adjacent", from_cell, to_cell),
            }

            if needed.issubset(action.precond_pos):
                return action

        return None

    def bfs_path(start, goal):

        if start == goal:
            return [start]

        frontier = Queue()
        frontier.push((start, [start]))

        visited = {start}

        adjacency = {}

        for fluent in state:

            if fluent[0] == "Adjacent":

                a = fluent[1]
                b = fluent[2]

                adjacency.setdefault(a, []).append(b)

        while not frontier.isEmpty():

            current, path = frontier.pop()

            if current == goal:
                return path

            for neighbor in adjacency.get(current, []):

                if neighbor in visited:
                    continue

                visited.add(neighbor)
                frontier.push((neighbor, path + [neighbor]))

        return []

    def build_navigate_hla(start, goal):

        path = bfs_path(start, goal)

        if len(path) <= 1:
            return HLA(f"Navigate({start},{goal})", [[]])

        moves = []

        for i in range(len(path) - 1):

            from_cell = path[i]
            to_cell = path[i + 1]

            move_action = find_move_action(from_cell, to_cell)

            if move_action is not None:
                moves.append(move_action)

        return HLA(
            f"Navigate({start},{goal})",
            [moves],
        )

    mission_refinements = []

    for patient in patients:

        patient_loc = get_location(patient)

        if patient_loc is None:
            continue

        supply = supplies[0]
        supply_loc = get_location(supply)

        robot_loc = get_robot_location()

        if robot_loc is None or supply_loc is None:
            continue

        navigate_to_supply = build_navigate_hla(robot_loc, supply_loc)

        pickup_supply = None

        for action in problem._all_groundings:

            if not action.name.startswith("PickUp"):
                continue

            required = {
                ("At", robot, supply_loc),
                ("At", supply, supply_loc),
            }

            if required.issubset(action.precond_pos):
                pickup_supply = action
                break

        navigate_to_medical = build_navigate_hla(supply_loc, medical_post)

        setup_supplies = None

        for action in problem._all_groundings:

            if not action.name.startswith("SetupSupplies"):
                continue

            required = {
                ("At", robot, medical_post),
                ("Holding", robot, supply),
            }

            if required.issubset(action.precond_pos):
                setup_supplies = action
                break

        navigate_to_patient = build_navigate_hla(medical_post, patient_loc)

        pickup_patient = None

        for action in problem._all_groundings:

            if not action.name.startswith("PickUp"):
                continue

            required = {
                ("At", robot, patient_loc),
                ("At", patient, patient_loc),
            }

            if required.issubset(action.precond_pos):
                pickup_patient = action
                break

        navigate_back = build_navigate_hla(patient_loc, medical_post)

        putdown_patient = None

        for action in problem._all_groundings:

            if not action.name.startswith("PutDown"):
                continue

            required = {
                ("At", robot, medical_post),
                ("Holding", robot, patient),
            }

            if required.issubset(action.precond_pos):
                putdown_patient = action
                break

        rescue_patient = None

        for action in problem._all_groundings:

            if not action.name.startswith("Rescue"):
                continue

            required = {
                ("At", robot, medical_post),
                ("At", patient, medical_post),
            }

            if required.issubset(action.precond_pos):
                rescue_patient = action
                break

        prepare_supplies = HLA(
            f"PrepareSupplies({supply})",
            [[
                navigate_to_supply,
                pickup_supply,
                navigate_to_medical,
                setup_supplies,
            ]],
        )

        extract_patient = HLA(
            f"ExtractPatient({patient})",
            [[
                navigate_to_patient,
                pickup_patient,
                navigate_back,
                putdown_patient,
                rescue_patient,
            ]],
        )

        full_mission = HLA(
            f"FullRescueMission({patient})",
            [[
                prepare_supplies,
                extract_patient,
            ]],
        )

        mission_refinements.append(full_mission)

    if not mission_refinements:
        return []

    if len(mission_refinements) == 1:
        return mission_refinements

    combined_refinement = []

    for mission in mission_refinements:
        combined_refinement.append(mission)

    root = HLA(
        "MultiRescueMission",
        [combined_refinement],
    )

    return [root]

    ### End of your code ###