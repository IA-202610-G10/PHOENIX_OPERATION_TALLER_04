from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    from planning.pddl import get_all_groundings
    unsatisfied = goal - state
    if not unsatisfied:
        return 0.0

    all_actions = get_all_groundings(domain, objects)
    cost = 0.0

    while unsatisfied:
        best_action = None
        max_covered = 0
        best_covered_set = frozenset()

        for action in all_actions:
            covered = action.add_list & unsatisfied
            if len(covered) > max_covered:
                max_covered = len(covered)
                best_action = action
                best_covered_set = covered

        if max_covered == 0:
            return float('inf')

        unsatisfied = unsatisfied - best_covered_set
        cost += 1.0

    return cost


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    from planning.pddl import get_applicable_actions
    current_state = state
    unsatisfied = goal - current_state
    if not unsatisfied:
        return 0.0

    cost = 0.0
    while unsatisfied:
        applicable_actions = get_applicable_actions(current_state, domain, objects)
        best_action = None
        best_score = (-1, -1)

        for action in applicable_actions:
            covered_goal = len(action.add_list & unsatisfied)
            new_fluents = len(action.add_list - current_state)
            if new_fluents == 0:
                continue

            score = (covered_goal, new_fluents)
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            return float('inf')

        current_state = current_state | best_action.add_list
        unsatisfied = goal - current_state
        cost += 1.0

    return cost
