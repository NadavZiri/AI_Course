import pytest
from typing import List, Tuple
import ex1
import search as search

# =====================================================================
# OK PROBLEMS - every student returns the optimal plan
# =====================================================================

# --- TESTPROBS (OK group) ---

# test_p1 - 4 persons, 2 elevators, bypass available. Optimal: 21
init_state_test_p1 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (2, 4, 6, 8), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (2, 3, 6),
        13: (6, 3, 1),
    },
}

# test_p3 - 5 persons, 2 elevators, weight forces single occupancy. Optimal: 16
init_state_test_p3 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6), 8),
        1: (6, (0, 1, 2, 3, 4, 5, 6), 8),
    },
    "Persons": {
        10: (0, 5, 6),
        11: (0, 5, 4),
        12: (6, 5, 0),
        13: (6, 5, 2),
        14: (3, 5, 6),
    },
}

# test_p4 - 4 persons, 2 elevators with overlap, weight allows only 1 each. Optimal: 25
init_state_test_p4 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (4, 5, 6, 7, 8), 10),
    },
    "Persons": {
        10: (0, 6, 8),
        11: (0, 4, 5),
        12: (8, 6, 0),
        13: (8, 5, 3),
    },
}

# --- BENCH (OK group) ---

# competition_1 - 4 persons, 2 elevators, all need transfers. Optimal: 24
init_state_competition_1 = {
    "height": 6,
    "Elevators": {
        0: (0, (0, 1, 2, 3,), 12),
        1: (6, (3, 4, 5, 6,), 12),
    },
    "Persons": {
        10: (0, 4, 5),
        11: (5, 4, 0),
        12: (2, 4, 6),
        13: (5, 4, 1),
    },
}

# competition_2 - shuffle: 3 persons across 3 elevators with overlap. Optimal: 16
init_state_competition_2 = {
    "height": 10,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 300),
        1: (5, (2, 3, 4, 5, 6, 7, 8), 300),
        2: (10, (5, 6, 7, 8, 9, 10), 300),
    },
    "Persons": {
        40: (0, 70, 10),
        41: (2, 70, 8),
        42: (10, 70, 0),
    },
}

# =====================================================================
# PARTIAL PROBLEMS - some students fail (reason annotated per problem)
# =====================================================================

# --- TESTPROBS (PARTIAL group) ---

# test_p2 - 4 persons, 3 elevators, extra bypass. Optimal: 22
# Partial: [Timeouts]
init_state_test_p2 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (2, 4, 6, 8), 10),
        2: (4, (7, 2), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (2, 3, 7),
        13: (6, 3, 1),
    },
}

# test_p5 - 4 persons, 3-elevator chain. Persons 10/11 need TWO transfers each. Optimal: 31
# Partial: [Pruning, Timeouts]
init_state_test_p5 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (3, 4, 5), 10),
        2: (5, (5, 6, 7, 8), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (4, 3, 7),
        13: (2, 3, 6),
    },
}

# test_p6 - 5 persons, 3 elevators (M3 topology + extra person at transfer floor). Optimal: 24
# Partial: [Non-optimal, Timeouts]
init_state_test_p6 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 10),
        1: (4, (2, 4, 6, 8), 10),
        2: (4, (7, 2), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (2, 3, 7),
        13: (6, 3, 1),
        14: (4, 3, 2),
    },
}

# test_p7 - 7-elevator single-floor-reach chain, 3 persons with progressively shorter trips. Optimal: 37
# Partial: [Pruning]
init_state_test_p7 = {
    "height": 14,
    "Elevators": {
        0: (0, (0, 2), 100),
        1: (2, (4,), 100),
        2: (4, (6,), 100),
        3: (6, (8,), 100),
        4: (8, (10,), 100),
        5: (10, (12,), 100),
        6: (12, (14,), 100),
    },
    "Persons": {
        10: (0, 8, 14),
        11: (0, 5, 10),
        12: (0, 5, 6),
    },
}

# test_p8 - 4 persons, 3 elevators with a short transfer chain. Optimal: 29
# Partial: [Pruning]
init_state_test_p8 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3), 9),
        1: (4, (3, 4, 5, 6), 9),
        2: (8, (6, 7, 8), 9),
    },
    "Persons": {
        33: (0, 3, 8),
        34: (8, 3, 0),
        35: (2, 2, 6),
        36: (6, 2, 1),
    },
}

# test_p9 - Branching W-chain, single-floor-reach + branch choice at floor 9. Optimal: 24
# Partial: [Pruning]
init_state_test_p9 = {
    "height": 14,
    "Elevators": {
        0: (0,  (0, 3),   100),
        1: (3,  (6,),     100),
        2: (6,  (9,),     100),
        3: (9,  (12,),    100),
        4: (9,  (4,),     100),
        5: (4,  (1,),     100),
    },
    "Persons": {
        20: (0, 10, 12),
        21: (0, 10, 1),
    },
}

# test_p10 - Mixed W-chain with a normal multi-floor elevator embedded mid-chain. Optimal: 17
# Partial: [Non-optimal, Pruning]
init_state_test_p10 = {
    "height": 16,
    "Elevators": {
        0: (0,  (0, 4),       100),
        1: (4,  (8,),         100),
        2: (8,  (8, 11, 14),  100),
        3: (14, (16,),        100),
    },
    "Persons": {
        30: (0,  10, 16),
        31: (8,  10, 11),
        32: (8,  10, 14),
    },
}

# --- BENCH (PARTIAL group) ---

# competition_3 - apartment: 5 persons, express + local + mid elevator. Optimal: 17
# Partial: [Timeouts]
init_state_competition_3 = {
    "height": 8,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5, 6, 7, 8), 200),
        1: (8, (0, 2, 4, 6, 8), 300),
        2: (4, (0, 1, 2, 3, 4), 150),
    },
    "Persons": {
        1: (1, 70, 7),
        2: (7, 90, 0),
        3: (0, 110, 8),
        4: (4, 80, 2),
        5: (2, 65, 6),
    },
}

# competition_4 - M1-style: 4 persons, 2 elevators with overlap at floor 4. Optimal: 25
# Partial: [Timeouts]
init_state_competition_4 = {
    "height": 7,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 12),
        1: (7, (4, 5, 6, 7), 12),
    },
    "Persons": {
        10: (0, 4, 6),
        11: (6, 4, 1),
        12: (2, 4, 7),
        13: (5, 4, 0),
    },
}

# competition_5 - M1-style variant: 2 elevators with overlap at floors 3,4. Optimal: 24
# Partial: [Non-optimal]
init_state_competition_5 = {
    "height": 8,
    "Elevators": {
        0: (1, (0, 1, 2, 3, 4), 10),
        1: (5, (3, 4, 5, 6, 7, 8), 10),
    },
    "Persons": {
        10: (0, 3, 8),
        11: (8, 3, 0),
        12: (1, 3, 6),
        13: (6, 3, 1),
    },
}

# competition_6 - H_chain-style: 4 persons, 3-elevator chain, double-transfer for 40/41. Optimal: 31
# Partial: [Pruning, Timeouts]
init_state_competition_6 = {
    "height": 10,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 12),
        1: (4, (4, 5, 6, 7), 12),
        2: (10, (6, 7, 8, 9, 10), 12),
    },
    "Persons": {
        40: (0, 3, 10),
        41: (10, 3, 0),
        42: (2, 3, 8),
        43: (7, 3, 1),
    },
}

# competition_7 - Variant of competition_5 with different starting positions and overlap floors. Optimal: 24
# Partial: [Pruning, Non-optimal]
init_state_competition_7 = {
    "height": 9,
    "Elevators": {
        0: (2, (0, 1, 2, 3, 4, 5), 10),
        1: (6, (4, 5, 6, 7, 8, 9), 10),
    },
    "Persons": {
        10: (0, 3, 7),
        11: (7, 3, 0),
        12: (2, 3, 9),
        13: (9, 3, 2),
    },
}

# competition_8 - Variant of competition_6: 4 persons, 3-elevator chain, mix of full and partial trips. Optimal: 31
# Partial: [Pruning, Timeouts]
init_state_competition_8 = {
    "height": 11,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4, 5), 12),
        1: (5, (4, 5, 6, 7, 8), 12),
        2: (11, (7, 8, 9, 10, 11), 12),
    },
    "Persons": {
        40: (0, 3, 11),
        41: (11, 3, 0),
        42: (3, 3, 6),
        43: (8, 3, 1),
    },
}

# competition_9 - 6 persons, 2 elevators with single overlap floor, mixed weights. Optimal: 33
# Partial: [Timeouts]
init_state_competition_9 = {
    "height": 7,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 8),
        1: (4, (4, 5, 6, 7), 8),
    },
    "Persons": {
        10: (0, 4, 7),
        11: (7, 4, 0),
        12: (1, 3, 5),
        13: (5, 3, 1),
        14: (4, 3, 6),
        15: (6, 3, 3),
    },
}

# competition_10 - 4 persons, 3-elevator chain, double-transfer required for 40/41. Optimal: 28
# Partial: [Pruning, Timeouts]
init_state_competition_10 = {
    "height": 9,
    "Elevators": {
        0: (0, (0, 1, 2, 3, 4), 14),
        1: (4, (3, 4, 5, 6, 7), 14),
        2: (9, (7, 8, 9), 14),
    },
    "Persons": {
        40: (0, 4, 9),
        41: (9, 4, 0),
        42: (2, 2, 7),
        43: (7, 2, 1),
    },
}

# =====================================================================
# TESTPROBS list (test_p1 .. test_p10 in execution order)
# =====================================================================

TESTPROBS: List[Tuple[dict, int]] = [
    (init_state_test_p1,  21),
    (init_state_test_p2,  22),
    (init_state_test_p3,  16),
    (init_state_test_p4,  25),
    (init_state_test_p5,  31),
    (init_state_test_p6,  24),
    (init_state_test_p7,  37),
    (init_state_test_p8,  29),
    (init_state_test_p9,  24),
    (init_state_test_p10, 17),
]

# =====================================================================
# BENCH list (competition_1 .. competition_10 in execution order)
# =====================================================================

BENCH: List[Tuple[str, dict, int]] = [
    ("competition_1",  init_state_competition_1,  24),
    ("competition_2",  init_state_competition_2,  16),
    ("competition_3",  init_state_competition_3,  17),
    ("competition_4",  init_state_competition_4,  25),
    ("competition_5",  init_state_competition_5,  24),
    ("competition_6",  init_state_competition_6,  31),
    ("competition_7",  init_state_competition_7,  24),
    ("competition_8",  init_state_competition_8,  29),  # note: 31 in header, corrected to match list
    ("competition_9",  init_state_competition_9,  33),
    ("competition_10", init_state_competition_10, 28),
]


# =====================================================================
# Helper
# =====================================================================

def solve(init_state):
    p = ex1.create_elevators_problem(init_state)
    result = search.astar_search(p, p.h_astar)
    assert result is not None, "A* returned no solution"
    node, _ = result
    assert isinstance(node, search.Node), "A* did not return a Node"
    path = node.path()[::-1]
    return [step.action for step in path][1:]


# =====================================================================
# TESTPROBS tests
# =====================================================================

@pytest.mark.timeout(30)
@pytest.mark.parametrize("init_state,expected", TESTPROBS, ids=[f"test_p{i+1}" for i in range(len(TESTPROBS))])
def test_testprob(init_state, expected):
    solution = solve(init_state)
    assert len(solution) == expected, f"Expected {expected} steps, got {len(solution)}: {solution}"


# =====================================================================
# BENCH tests
# =====================================================================

@pytest.mark.timeout(30)
@pytest.mark.parametrize("name,init_state,expected", BENCH, ids=[name for name, _, _ in BENCH])
def test_bench(name, init_state, expected):
    solution = solve(init_state)
    assert len(solution) == expected, f"[{name}] Expected {expected} steps, got {len(solution)}: {solution}"
