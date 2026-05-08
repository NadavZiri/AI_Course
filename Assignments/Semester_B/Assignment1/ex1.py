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
        self.allowed = {eid: allowed for eid, (floor, allowed, weight_limit) in initial["Elevators"].items()}


        search.Problem.__init__(self, hashable)

    def successor(self, state):
        """ Generates the successor states returns [(action, achieved_states, ...)]"""
        elev_state, person_state = state
        current_weights = {eid: 0 for eid, floor in elev_state}
        passengers = {eid: False for eid, floor in elev_state}  # eid -> has passengers
        waiting_floors = set()  # floors where someone is waiting (not in any elevator)

        for pid, floor, eid in person_state:
            if eid != -1:
                current_weights[eid] += self.weights[pid]
                passengers[eid] = True
            elif floor != self.goals[pid]:
                waiting_floors.add(floor)

        move = []
        enter = []
        exit = []

        loaded_elev_goals = {}  # eid -> set of passenger goal floors (only when all goals are reachable)
        for eid, floor in elev_state:
            if passengers[eid]:
                pax_goals = {self.goals[pid] for pid, _, elevator_id in person_state if elevator_id == eid}
                if all(g in self.allowed[eid] for g in pax_goals):
                    loaded_elev_goals[eid] = pax_goals

        for eid, floor in elev_state:
            for allowed_floor in self.allowed[eid]:
                if allowed_floor == floor:
                    continue
                # prune: empty elevator moving to a floor with no one waiting is useless
                if not passengers[eid] and allowed_floor not in waiting_floors:
                    continue
                # prune: loaded elevator (all goals reachable) only goes to passenger goals or waiting floors
                if eid in loaded_elev_goals and allowed_floor not in loaded_elev_goals[eid] and allowed_floor not in waiting_floors:
                    continue
                new_elev_state = tuple((e, floor if e != eid else allowed_floor) for e, floor in elev_state)
                new_person_state = tuple((pid, allowed_floor if elevator_id == eid else p_floor, elevator_id) for pid, p_floor, elevator_id in person_state)
                move.append((f"MOVE{{{eid},{allowed_floor}}}", (new_elev_state, new_person_state)))

        for pid, floor, eid in person_state:
            if eid == -1:
                if floor == self.goals[pid]:
                    continue
                for e_id, e_floor in elev_state:
                    if current_weights[e_id] + self.weights[pid] <= self.max_w[e_id] and floor == e_floor:
                        new_person_state = tuple((p, f, e_id) if p == pid else (p, f, el) for p, f, el in person_state)
                        enter.append((f"ENTER{{{pid},{e_id}}}", (elev_state, new_person_state)))
            else:
                if self.goals[pid] in self.allowed[eid] and floor != self.goals[pid]:
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
        and returns a goal distance estimate"""
        elev_state, person_state = node.state
        elev_floors = {eid: floor for eid, floor in elev_state}
        total = 0

        in_elev = {}     # (eid, goal_floor) -> count of persons sharing this elevator+destination
        not_in_elev = {} # (current_floor, goal_floor) -> count of persons sharing this trip

        for pid, floor, eid in person_state:
            goal = self.goals[pid]
            if floor == goal and eid == -1:
                continue
            if eid != -1:
                key = (eid, goal)
                in_elev[key] = in_elev.get(key, 0) + 1
            else:
                key = (floor, goal)
                not_in_elev[key] = not_in_elev.get(key, 0) + 1

        for (eid, goal), count in in_elev.items():
            if elev_floors[eid] == goal:
                total += count           # EXIT each
            elif goal in self.allowed[eid]:
                total += 1 + count       # 1 shared MOVE + EXIT each
            else:
                total += count * 3 + 1  # EXIT each + ENTER each + 1 shared MOVE + EXIT each

        for (floor, goal), count in not_in_elev.items():
            if any(floor in self.allowed[e] and goal in self.allowed[e] for e in self.allowed):
                elev_at_pickup = any(
                    e_floor == floor
                    for e, e_floor in elev_state
                    if floor in self.allowed[e] and goal in self.allowed[e]
                )
                total += count * 2 + (1 if elev_at_pickup else 2)
            else:
                total += count * 4 + 2  # ENTER each + EXIT each + ENTER each + 2 shared MOVEs + EXIT each

        return total



def create_elevators_problem(game):
    print("<<create_elevators_problem")
    """ Create an elevators problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return ElevatorsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
