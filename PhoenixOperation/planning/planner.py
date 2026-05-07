from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontier: Queue = Queue()
    frontier.push((start, []))  # (state, plan)
    visited: set[State] = {start}

    while not frontier.isEmpty():
        state, plan = frontier.pop()
        if problem.isGoalState(state):
            return plan

        for next_state, action, _cost in problem.getSuccessors(state):
            if next_state in visited:
                continue
            visited.add(next_state)
            frontier.push((next_state, plan + [action]))

    return []

    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    # En la notación de la diapositiva:
    #   Pos(g) = goal_set
    #   Neg(g) = ∅  (este proyecto modela metas como fluentes positivos)
    #
    # Relevancia: ADD(a) debe unificarse con al menos un literal positivo del objetivo.
    if not (action.add_list & goal_set):
        return None

    # Consistencia: la acción no puede negar (borrar) un literal que debe quedar verdadero.
    if action.del_list & goal_set:
        return None

    # Además, si algo debe ser verdadero antes de la acción (porque NO lo logra ADD),
    # no puede estar en precondiciones negativas de la acción.
    needed_before = goal_set - action.add_list
    if action.precond_neg & needed_before:
        return None

    # REGRESS(Pos(g), a) = (Pos(g) - ADD(a)) ∪ Pos(Precond(a))
    return (goal_set - action.add_list) | action.precond_pos

    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    ### Your code here ###
    start_state = problem.initial_state
    goal = problem.goal

    if goal.issubset(start_state):
        return []

    all_actions = get_all_groundings(problem.domain, problem.objects)
    actions_by_add: dict[tuple, list[Action]] = {}
    for action in all_actions:
        for f in action.add_list:
            actions_by_add.setdefault(f, []).append(action)

    def is_inconsistent(subgoal: State) -> bool:
        """
        Poda mínima: descartar metas imposibles por inconsistencias lógicas
        (sin requerir heurísticas ni conocimiento avanzado del dominio).
        """
        # Un mismo ente no puede estar en dos celdas a la vez.
        at_locs: dict[object, set[object]] = {}
        for f in subgoal:
            if f and f[0] == "At":
                _pred, ent, loc = f
                at_locs.setdefault(ent, set()).add(loc)
        if any(len(locs) > 1 for locs in at_locs.values()):
            return True

        robot = "robot"

        # El robot no puede estar HandsFree y Holding algo simultáneamente.
        if ("HandsFree", robot) in subgoal:
            for f in subgoal:
                if f and f[0] == "Holding" and len(f) >= 3 and f[1] == robot:
                    return True

        # El robot no puede sostener dos objetos distintos a la vez.
        held = {f[2] for f in subgoal if f and f[0] == "Holding" and f[1] == robot}
        if len(held) > 1:
            return True

        # Un paciente no puede estar rescatado y a la vez "At" en alguna celda.
        rescued = {f[1] for f in subgoal if f and f[0] == "Rescued"}
        if rescued:
            for f in subgoal:
                if f and f[0] == "At" and f[1] in rescued:
                    return True

        return False

    def has_static_contradiction(subgoal: State) -> bool:
        # Estos predicados no cambian nunca; si son requeridos pero no están en el
        # estado inicial, ese subobjetivo es imposible.
        static_preds = {"MedicalPost", "Adjacent", "Pickable"}
        for f in subgoal:
            if f and f[0] in static_preds and f not in start_state:
                return True
        return False

    def canonicalize(subgoal: State) -> State:
        # No queremos que el "estado objetivo" cargue cosas estáticas ni Free(c),
        # porque eso infla el espacio de búsqueda sin aportar.
        drop_preds = {"MedicalPost", "Adjacent", "Pickable", "Free"}
        return frozenset(f for f in subgoal if f and f[0] not in drop_preds)

    goal = canonicalize(goal)

    frontier: Queue = Queue()
    frontier.push((goal, []))  # (subgoal, actions_in_reverse_order)
    visited: set[State] = {goal}

    while not frontier.isEmpty():
        subgoal, actions_rev = frontier.pop()

        if subgoal.issubset(start_state):
            return list(reversed(actions_rev))

        # Acciones relevantes: las que logran algo que AÚN no está satisfecho
        # por el estado inicial (evita branching inútil).
        unsatisfied = subgoal - start_state
        if not unsatisfied:
            return list(reversed(actions_rev))

        candidate_actions: set[Action] = set()
        for f in unsatisfied:
            candidate_actions.update(actions_by_add.get(f, []))

        for action in candidate_actions:
            new_subgoal = regress(subgoal, action)
            if new_subgoal is None:
                continue
            if has_static_contradiction(new_subgoal):
                continue
            new_subgoal = canonicalize(new_subgoal)
            if is_inconsistent(new_subgoal):
                continue
            if new_subgoal in visited:
                continue

            visited.add(new_subgoal)
            frontier.push((new_subgoal, actions_rev + [action]))

    return []

    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###

    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
