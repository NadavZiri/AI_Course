import ex1_check
import search as search
import utils as utils

id = ["No numbers - I'm special!"]



class ElevatorsProblem(search.Problem):
    """This class implements an elevators problem"""

    def __init__(self, initial):
        """ Constructor only needs the initial state.
        Don't forget to set the goal or implement the goal test"""
        elev_state = tuple(tuple((eid, floor) for eid, (floor, allowed, max_w) in initial["Elevators"].items()))
        person_state = tuple(sorted((pid, start, -1) for pid, (start, weight, goal) in initial["Persons"].items()))     
        hashable = (elev_state, person_state)
        self.weights = {pid: weight for pid, (start, weight, goal) in initial["Persons"].items()}   
        self.goals = {pid: goal for pid, (start, weight, goal) in initial["Persons"].items()}
        self.max_w = {eid: max_w for eid, (floor, allowed, max_w) in initial["Elevators"].items()}
        # frozenset -> O(1) membership. `allowed` = floors an elevator may MOVE to.
        self.allowed = {eid: frozenset(allowed) for eid, (floor, allowed, weight_limit) in initial["Elevators"].items()}
        # An elevator can also be boarded at (and transferred from) its START floor, even when
        # that floor is not in `allowed` (it just can't move back there). `positions` is the set
        # of floors where a passenger can board/leave the elevator -> used for the reachability graph.
        self.positions = {eid: self.allowed[eid] | {floor}
                          for eid, (floor, allowed, weight_limit) in initial["Elevators"].items()}

        # static elevator-reachability graph: two elevators are adjacent if they share a
        # boardable position (a passenger can transfer between them there).
        self._adj = {e: set() for e in self.allowed}
        elevs = list(self.allowed)
        for i, a in enumerate(elevs):
            for b in elevs[i + 1:]:
                if self.positions[a] & self.positions[b]:
                    self._adj[a].add(b)
                    self._adj[b].add(a)

        # memoized exact distances (depend only on the static graph)
        self._dist_cache = {}       # (floor, goal)   -> min elevators to board to chain floor -> goal
        self._elev_dist_cache = {}  # (eid, goal)     -> additional elevators to board after `eid`

        search.Problem.__init__(self, hashable)

    def _elev_dist(self, eid, goal):
        """Minimum number of *additional* elevators that must be boarded, beyond `eid`,
        for a passenger currently inside `eid` to reach `goal`. 0 if `eid` already reaches it.
        Floor-independent (depends only on which elevator), which is what keeps the heuristic
        consistent across MOVE actions."""
        if goal in self.allowed[eid]:
            return 0
        key = (eid, goal)
        if key in self._elev_dist_cache:
            return self._elev_dist_cache[key]
        visited = {eid}
        frontier = [eid]
        depth = 0
        result = None
        while frontier:
            depth += 1
            nxt = []
            for e in frontier:
                for n in self._adj[e]:
                    if n in visited:
                        continue
                    if goal in self.allowed[n]:
                        result = depth
                        nxt = []
                        break
                    visited.add(n)
                    nxt.append(n)
                if result is not None:
                    break
            if result is not None:
                break
            frontier = nxt
        self._elev_dist_cache[key] = result
        return result

    def _dist(self, floor, goal):
        """Minimum number of elevators that must be boarded to chain `floor` -> `goal`
        (exact, via the reachability graph). 0 if already there, None if unreachable."""
        if floor == goal:
            return 0
        key = (floor, goal)
        if key in self._dist_cache:
            return self._dist_cache[key]
        best = None
        for e in self.allowed:
            if floor in self.positions[e]:           # can board e here (allowed floor or its start)
                ed = self._elev_dist(e, goal)
                if ed is not None and (best is None or ed + 1 < best):
                    best = ed + 1
        self._dist_cache[key] = best
        return best

    def successor(self, state):
        """ Generates the successor states returns [(action, achieved_states, ...)]"""
        elev_state, person_state = state
        current_weights = {eid: 0 for eid, floor in elev_state}
        pax_goals = {eid: [] for eid, floor in elev_state}   # eid -> goals of its passengers
        waiting = {}                                         # floor -> set of goals of waiting persons

        for pid, floor, eid in person_state:
            if eid != -1:
                current_weights[eid] += self.weights[pid]
                pax_goals[eid].append(self.goals[pid])
            elif floor != self.goals[pid]:
                waiting.setdefault(floor, set()).add(self.goals[pid])

        move = []
        enter = []
        exit = []

        for eid, floor in elev_state:
            allowed = self.allowed[eid]
            # Collect the floors this elevator has a reason to visit.
            useful = set()
            # (a)/(b) serve current passengers: their goal, or a best transfer floor.
            for g in pax_goals[eid]:
                if g in allowed:
                    useful.add(g)
                else:
                    ed = self._elev_dist(eid, g)
                    for t in allowed:
                        if self._dist(t, g) == ed:        # t is an optimal hand-off floor
                            useful.add(t)
            # (c) pick up a waiting person for whom this elevator is on a shortest chain.
            for t, goals in waiting.items():
                if t in allowed:
                    for gp in goals:
                        ed = self._elev_dist(eid, gp)
                        d = self._dist(t, gp)
                        if d is not None and ed is not None and d == 1 + ed:
                            useful.add(t)
                            break

            for allowed_floor in useful:
                if allowed_floor == floor:
                    continue
                new_elev_state = tuple((e, floor if e != eid else allowed_floor) for e, floor in elev_state)
                new_person_state = tuple((pid, allowed_floor if elevator_id == eid else p_floor, elevator_id) for pid, p_floor, elevator_id in person_state)
                move.append((f"MOVE{{{eid},{allowed_floor}}}", (new_elev_state, new_person_state)))

        for pid, floor, eid in person_state:
            if eid == -1:
                goal = self.goals[pid]
                if floor == goal:
                    continue
                for e_id, e_floor in elev_state:
                    if e_floor != floor or current_weights[e_id] + self.weights[pid] > self.max_w[e_id]:
                        continue
                    # only board an elevator that is on a shortest chain to the goal
                    ed = self._elev_dist(e_id, goal)
                    if ed is None or self._dist(floor, goal) != 1 + ed:
                        continue
                    new_person_state = tuple((p, f, e_id) if p == pid else (p, f, el) for p, f, el in person_state)
                    enter.append((f"ENTER{{{pid},{e_id}}}", (elev_state, new_person_state)))
            else:
                goal = self.goals[pid]
                if goal in self.allowed[eid]:
                    if floor != goal:
                        continue                      # goal reachable here: ride to it, don't get off early
                else:
                    # transfer passenger: only get off at a best transfer floor
                    if self._dist(floor, goal) != self._elev_dist(eid, goal):
                        continue
                new_person_state = tuple((p, f, -1) if p == pid else (p, f, el) for p, f, el in person_state)
                exit.append((f"EXIT{{{pid},{eid}}}", (elev_state, new_person_state)))

        return move + enter + exit

    def goal_test(self, state):
        """ given a state, checks if this is the goal state, compares to the created goal state returns True/False"""
        elev_state, person_state = state
        for pid, current_floor, eid in person_state:
            if eid != -1 or current_floor != self.goals[pid]:
                return False
        return True

    def h_astar(self, node):
        """ This is the heuristic. It gets a node (not a state)
        and returns a goal distance estimate.

        Per-person ENTER/EXIT lower bound: a person who must board k elevators incurs
        exactly k ENTERs + k EXITs, none of which are shared between people, so summing
        is admissible. The estimate is also *consistent*: it is floor-independent for a
        passenger inside an elevator (via _elev_dist), so MOVE actions never drop it, and
        each ENTER/EXIT lowers it by at most the action's unit cost. Shared MOVE cost is
        deliberately omitted -- counting it per person would break both admissibility and
        consistency, and the move-pruning in successor recovers the lost speed instead."""
        _, person_state = node.state
        total = 0
        for pid, floor, eid in person_state:
            goal = self.goals[pid]
            if eid == -1:
                if floor != goal:
                    total += 2 * self._dist(floor, goal)        # ENTER + EXIT per boarded elevator
            elif goal in self.allowed[eid]:
                total += 1                                       # ride to goal, then EXIT
            else:
                total += 1 + 2 * self._elev_dist(eid, goal)      # EXIT here + board the remaining chain
        return total



def create_elevators_problem(game):
    print("<<create_elevators_problem")
    """ Create an elevators problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return ElevatorsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
