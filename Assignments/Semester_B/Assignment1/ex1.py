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
        successors = []
        elev_state, person_state = state
        current_weights = {eid: 0 for eid, floor in elev_state}                                                                          
        for person in person_state:
            pid, current_floor, elevator_id = person
            if elevator_id != -1:
                current_weights[elevator_id] += self.weights[pid]
        move = []
        enter = []
        exit = []
        for elevator in elev_state:
            eid, floor = elevator
            for allowed_floor in self.allowed[eid]:
                if allowed_floor == floor:
                    continue
                new_elev_state = tuple((e, floor if e != eid else allowed_floor) for e, floor in elev_state)
                new_person_state = tuple((pid, allowed_floor if elevator_id == eid else p_floor, elevator_id) for pid, p_floor, elevator_id in person_state)
                new_state = (new_elev_state, new_person_state)
                action_string = f"MOVE{{{eid},{allowed_floor}}}"
                move.append((action_string, new_state))
        for person in person_state:
            pid, floor, elevator = person
            if elevator == -1 and floor == self.goals[pid]:
                continue
            if elevator == -1:
                for e in elev_state:
                    eid, elev_floor = e
                    if current_weights[eid] + self.weights[pid] <= self.max_w[eid] and floor == elev_floor:
                        new_person_state = tuple((p, f, eid) if p == pid else (p, f, el) for p, f, el in person_state)                                                                                                                              
                        new_state = (elev_state, new_person_state)                                                                                     
                        enter.append((f"ENTER{{{pid},{eid}}}", new_state))
            else:
                for e in elev_state:
                    eid, elev_floor = e
                    if eid != elevator:
                        continue
                    if self.goals[pid] in self.allowed[eid] and floor != self.goals[pid]:
                        continue
                    new_person_state = tuple((p, f, -1) if p == pid else (p, f, el) for p, f, el in person_state)
                    new_state = (elev_state, new_person_state)
                    exit.append((f"EXIT{{{pid},{eid}}}", new_state))
        successors = move + enter + exit
        return successors

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
        total = 0
        for pid, floor, eid in person_state:
            goal = self.goals[pid]
            if floor == goal and eid == -1:
                continue
            if eid == -1:
                if any(floor in self.allowed[e] and goal in self.allowed[e] for e in self.allowed):
                    total += 3  # ENTER + MOVE + EXIT
                else:
                    total += 5  # ENTER + EXIT + ENTER + MOVE + EXIT (transfer)
            elif floor == goal:
                total += 1  # EXIT
            elif goal in self.allowed[eid]:
                total += 2  # MOVE + EXIT
            else:
                total += 4  # EXIT + ENTER + MOVE + EXIT (transfer)
        return total



def create_elevators_problem(game):
    print("<<create_elevators_problem")
    """ Create an elevators problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return ElevatorsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
