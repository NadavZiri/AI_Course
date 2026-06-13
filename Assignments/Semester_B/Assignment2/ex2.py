import ext_elev, time
from collections import deque

id = ["000000000"]


class Controller:
    """Stochastic multi-elevator controller.

    Implement choose_next_action(state) to return a single legal action
    string. See the assignment PDF (Section 5) for the full API contract
    and the engine-access policy.
    """

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        # Initialize precomputations, caches, etc. here.
        self.capacities = self.game.get_capacities()
        self.reachable = self.game.get_reachable()
        self.expected_rewards = {}
        self.elevator_action_prob = {}
        self.person_action_prob = {}

        initial_state = self.game.get_initial_state()
        initial_elevators, initial_persons, _ = initial_state
        for pid, _ in initial_persons:
            rewards = self.game.get_person_reward(pid)
            self.expected_rewards[pid] = sum(rewards) / len(rewards)
            self.person_action_prob[pid] = self.game.get_person_action_prob(pid)
        
        for eid, _, _ in initial_elevators:
            self.elevator_action_prob[eid] = self.game.get_elevator_action_prob(eid)
            
        self.all_floors = set()
        for reachable_set in self.reachable.values():
            self.all_floors.update(reachable_set)

        # Initialize expected time matrix
        self.min_time = {f1: {f2: float('inf') for f2 in self.all_floors} for f1 in self.all_floors}
        for f in self.all_floors:
            self.min_time[f][f] = 0.0

        # Base edges: direct elevator paths weighted by expected time to succeed
        for eid, reachable_set in self.reachable.items():
            p_succ = self.elevator_action_prob[eid]
            exp_moves = 1.0 / p_succ  # 95% = 1.05 moves. 25% = 4.0 moves.
            floors = list(reachable_set)
            
            for i in range(len(floors)):
                for j in range(len(floors)):
                    if i != j:
                        f1, f2 = floors[i], floors[j]
                        if exp_moves < self.min_time[f1][f2]:
                            self.min_time[f1][f2] = exp_moves

        # Floyd-Warshall algorithm inside __init__
        for k in self.all_floors:
            for i in self.all_floors:
                for j in self.all_floors:
                    if self.min_time[i][k] != float('inf') and self.min_time[k][j] != float('inf'):
                        cost = self.min_time[i][k] + self.min_time[k][j]
                        if k != i and k != j:
                            cost += 1.5  # Soften this from 4.0/6.0 to 1.5
                        if cost < self.min_time[i][j]:
                            self.min_time[i][j] = cost


    def get_legal_actions(self, state):
        elevators_t, persons_t, _ = state
        actions = []

        # 1. The Greedy Farmer Fix
        if self.expected_rewards:
            max_reward = max(self.expected_rewards.values())
            if max_reward > self.game.get_goal_reward():
                actions.append("RESET")
        else:
            actions.append("RESET")

        elev_info = {eid: {'floor': f, 'weight': w} for eid, f, w in elevators_t}

        # 2. Compute useful target floors per elevator (pruning MOVE actions)
        # Only move to floors where a person waits, needs delivery, or needs a transfer.
        useful_targets = {eid: set() for eid in elev_info}

        for pid, loc in persons_t:
            p_goal = self.game.get_person_goal(pid)
            person_weight = self.game.get_person_weight(pid)

            if loc[0] == 'floor':
                p_floor = loc[1]
                for eid, info in elev_info.items():
                    if (p_floor in self.reachable[eid] and
                            info['weight'] + person_weight <= self.capacities[eid]):
                        current_dist = self.min_time[p_floor].get(p_goal, float('inf'))
                        if any(self.min_time[f].get(p_goal, float('inf')) < current_dist
                               for f in self.reachable[eid]):
                            useful_targets[eid].add(p_floor)

            elif loc[0] == 'in':
                eid_in = loc[1]
                if p_goal in self.reachable[eid_in]:
                    useful_targets[eid_in].add(p_goal)
                    # Pre-position a MORE-reliable elevator at the current floor for a possible switch.
                    # Never send a broken elevator to shadow a reliable one — it wastes search budget.
                    e_floor_in = elev_info[eid_in]['floor']
                    for eid2, info2 in elev_info.items():
                        if (eid2 != eid_in
                                and e_floor_in in self.reachable[eid2]
                                and p_goal in self.reachable[eid2]
                                and info2['weight'] + person_weight <= self.capacities[eid2]
                                and self.elevator_action_prob[eid2] >= self.elevator_action_prob[eid_in]):
                            useful_targets[eid2].add(e_floor_in)
                else:
                    # Find the best transfer floor(s) reachable by this elevator
                    best_time = min(
                        (self.min_time[f].get(p_goal, float('inf')) for f in self.reachable[eid_in]),
                        default=float('inf')
                    )
                    if best_time < float('inf'):
                        transfer_floors = {
                            f for f in self.reachable[eid_in]
                            if self.min_time[f].get(p_goal, float('inf')) == best_time
                        }
                        useful_targets[eid_in].update(transfer_floors)
                        # Pre-position other elevators at the transfer floor
                        for t_floor in transfer_floors:
                            for eid2, info2 in elev_info.items():
                                if eid2 != eid_in and t_floor in self.reachable[eid2]:
                                    if info2['weight'] + person_weight <= self.capacities[eid2]:
                                        useful_targets[eid2].add(t_floor)

        # 3. Generate MOVEs to useful floors only
        for eid, info in elev_info.items():
            current_floor = info['floor']
            for target_floor in useful_targets[eid]:
                if target_floor != current_floor:
                    actions.append(f"MOVE{{{eid},{target_floor}}}")

        # 4. Generate ENTERs and EXITs
        for pid, loc in persons_t:
            person_weight = self.game.get_person_weight(pid)
            p_goal = self.game.get_person_goal(pid)

            if loc[0] == 'floor':
                for eid, info in elev_info.items():
                    if info['floor'] == loc[1] and info['weight'] + person_weight <= self.capacities[eid]:
                        current_dist = self.min_time[loc[1]].get(p_goal, float('inf'))
                        if any(self.min_time[f].get(p_goal, float('inf')) < current_dist
                               for f in self.reachable[eid]):
                            actions.append(f"ENTER{{{pid},{eid}}}")

            elif loc[0] == 'in':
                actions.append(f"EXIT{{{pid},{loc[1]}}}")

        # 5. Deadlock fallback
        if not actions:
            actions.append("RESET")
            
        return actions

    def get_transitions(self, state, action):
        elevators_t, persons_t, total_persons_remaining = state
        if action == "RESET":
            return [(1.0, self.game.get_initial_state(), 0.0)]
        action_parts = action.split("{")
        action_type = action_parts[0]
        
        args_str = action_parts[1].rstrip("}")
        arg1, arg2 = map(int, args_str.split(","))
        
        transitions = []
        if action_type == "ENTER":
            p, e = arg1, arg2
            prob_success = self.person_action_prob[p]
            
            if prob_success < 1.0:
                transitions.append((1.0 - prob_success, state, 0.0))
            
            new_elevs = list(elevators_t)
            new_persons = list(persons_t)
            
            for i, (pid, loc) in enumerate(new_persons):
                if pid == p and loc[0] == 'floor':
                    new_persons[i] = (pid, ('in', e))
                    break
            
            p_weight = self.game.get_person_weight(p)
            for i, (eid, cur_f, cur_w) in enumerate(new_elevs):
                if eid == e:
                    new_elevs[i] = (eid, cur_f, cur_w + p_weight)
                    break
            
            success_state = (tuple(new_elevs), tuple(new_persons), total_persons_remaining)
            transitions.append((prob_success, success_state, 0.0))
        
        elif action_type == "EXIT":
            p, e = arg1, arg2
            prob_success = self.person_action_prob[p]
            
            # FAILURE BRANCH
            if prob_success < 1.0:
                transitions.append((1.0 - prob_success, state, 0.0))
                
            # SUCCESS BRANCH
            new_elevs = list(elevators_t)
            new_persons = list(persons_t)
            
            # Find current floor of elevator e
            current_floor = next(f for eid, f, w in new_elevs if eid == e)
            p_weight = self.game.get_person_weight(p)
            p_goal = self.game.get_person_goal(p)
            
            # Update elevator weight
            for i, (eid, f, w) in enumerate(new_elevs):
                if eid == e:
                    new_elevs[i] = (eid, f, w - p_weight)
                    break
                    
            reward = 0.0
            new_total_remaining = total_persons_remaining
            
            if current_floor == p_goal:
                # Delivered! Remove person and add expected reward
                new_persons = [person for person in new_persons if person[0] != p]
                reward = self.expected_rewards[p]
                new_total_remaining -= 1
                
                # Check for global goal state
                if new_total_remaining == 0:
                    reward += self.game.get_goal_reward()
                    success_state = self.game.get_initial_state()
                    transitions.append((prob_success, success_state, reward))
                    return transitions # Short-circuit, episode restarts
            else:
                # Just stepped out onto the wrong floor
                for i, (pid, loc) in enumerate(new_persons):
                    if pid == p:
                        new_persons[i] = (pid, ('floor', current_floor))
                        break
                        
            success_state = (tuple(new_elevs), tuple(new_persons), new_total_remaining)
            transitions.append((prob_success, success_state, reward))

        # ---------------------------------------------------------
        # 3. MOVE ACTION: MOVE{e, target_f}
        # ---------------------------------------------------------
        elif action_type == "MOVE":
            e, target_f = arg1, arg2
            prob_success = self.elevator_action_prob[e]
            current_floor = next(f for eid, f, w in elevators_t if eid == e)
            
            # SUCCESS BRANCH
            new_elevs_succ = list(elevators_t)
            for i, (eid, f, w) in enumerate(new_elevs_succ):
                if eid == e:
                    new_elevs_succ[i] = (eid, target_f, w)
                    break
            success_state = (tuple(new_elevs_succ), persons_t, total_persons_remaining)
            transitions.append((prob_success, success_state, 0.0))
            
            # FAILURE BRANCHES (Uniform distribution over other floors)
            if prob_success < 1.0:
                reachable = set(self.reachable[e])
                # Failure options: current floor + all reachable EXCEPT target
                failure_floors = (reachable - {target_f}) | {current_floor}
                
                if len(failure_floors) > 0:
                    prob_per_failure = (1.0 - prob_success) / len(failure_floors)
                    
                    for fail_f in failure_floors:
                        new_elevs_fail = list(elevators_t)
                        for i, (eid, f, w) in enumerate(new_elevs_fail):
                            if eid == e:
                                new_elevs_fail[i] = (eid, fail_f, w)
                                break
                        fail_state = (tuple(new_elevs_fail), persons_t, total_persons_remaining)
                        transitions.append((prob_per_failure, fail_state, 0.0))

        return transitions
    
    
    def get_cost_on_floor(self, p_floor, p_goal, elev_floors, p_action_prob):
        best_cost = float('inf')

        for eid, e_floor in elev_floors.items():
            if p_floor in self.reachable[eid]:
                arrive = 0.0 if e_floor == p_floor else (1.0 / self.elevator_action_prob[eid])

                if p_goal in self.reachable[eid]:
                    travel = 0.0 if p_floor == p_goal else (1.0 / self.elevator_action_prob[eid])
                    cost = arrive + travel
                else:
                    best_drop = float('inf')
                    for f in self.reachable[eid]:
                        dist_f = 0.0 if f == p_floor else (1.0 / self.elevator_action_prob[eid])
                        future = self.min_time[f].get(p_goal, float('inf'))

                        if dist_f + future + 1.5 < best_drop:
                            best_drop = dist_f + future + 1.5

                    cost = arrive + best_drop

                if cost < best_cost:
                    best_cost = cost

        return best_cost + (2.0 / p_action_prob)

    def evaluate_heuristic(self, state):
        elevators_t, persons_t, total_remaining = state
        if total_remaining == 0:
            return 0.0

        total_value = 0.0
        elev_floors = {eid: f for eid, f, w in elevators_t}

        goal_slice = self.game.get_goal_reward() / total_remaining

        for pid, loc in persons_t:
            expected_reward = self.expected_rewards[pid] + goal_slice
            p_action_prob = self.person_action_prob[pid]
            p_goal = self.game.get_person_goal(pid)

            expected_time = float('inf')

            if loc[0] == 'floor':
                expected_time = self.get_cost_on_floor(
                    loc[1], p_goal, elev_floors, p_action_prob
                )

            elif loc[0] == 'in':
                eid = loc[1]
                e_floor = elev_floors[eid]

                if p_goal in self.reachable[eid]:
                    travel = 0.0 if e_floor == p_goal else (1.0 / self.elevator_action_prob[eid])
                    exit_cost = 1.0 / p_action_prob
                    expected_time = travel + exit_cost
                else:
                    best_transfer_time = float('inf')
                    for f in self.reachable[eid]:
                        drop_time = (0.0 if e_floor == f else (1.0 / self.elevator_action_prob[eid])) + (1.0 / p_action_prob)
                        future_time = self.get_cost_on_floor(
                            f, p_goal, elev_floors, p_action_prob
                        )
                        if drop_time + future_time < best_transfer_time:
                            best_transfer_time = drop_time + future_time
                    expected_time = best_transfer_time

            if expected_time != float('inf'):
                discounted_reward = expected_reward * (0.99 ** expected_time)
                total_value += discounted_reward

        return total_value
    
    
    def expectimax(self, state, depth):
        # 1. Check the memoization cache first to save massive compute time
        state_key = (state, depth)
        if state_key in self.memo:
            return self.memo[state_key]

        _, _, total_remaining = state
        
        if total_remaining == 0:
            return 0 
            
        if depth == 0:
            val = self.evaluate_heuristic(state)
            self.memo[state_key] = val
            return val
            
        legal_actions = self.get_legal_actions(state)
        max_value = float('-inf')
        
        for action in legal_actions:
            action_value = 0
            for prob, next_state, reward in self.get_transitions(state, action):
                # ADD 0.99 MULTIPLIER HERE:
                action_value += prob * (reward + 0.99 * self.expectimax(next_state, depth - 1))
            
            if action_value > max_value:
                max_value = action_value
                
        # Cache the result before returning
        self.memo[state_key] = max_value if max_value != float('-inf') else 0
        return self.memo[state_key]

    def choose_next_action(self, state):
        start_time = time.time()
        time_budget = 0.55  # Safe limit per step (engine allows ~0.5s)

        legal_actions = self.get_legal_actions(state)
        if not legal_actions:
            return "RESET"
        if len(legal_actions) == 1:
            return legal_actions[0] # Don't waste time thinking if forced

        best_action = "RESET"

        # Clear the cache for the new turn to avoid memory leaks
        self.memo = {}

        # Tiebreaker lookups: goal-floor EXIT and empty-reliable-elev ENTER
        elevators_t, _, _ = state
        elev_floors = {eid: f for eid, f, _ in elevators_t}
        elev_weights = {eid: w for eid, _, w in elevators_t}

        depth = 1
        while True:
            current_best_action = None
            current_best_value = float('-inf')

            for action in legal_actions:
                # Emergency Stop: If we exceed budget, return the best action from the LAST depth
                if time.time() - start_time > time_budget:
                    return best_action if best_action != "RESET" else legal_actions[0]

                expected_utility = 0
                for prob, next_state, reward in self.get_transitions(state, action):
                    # Added the 0.99 multiplier
                    expected_utility += prob * (reward + 0.99 * self.expectimax(next_state, depth - 1))

                # Tiebreaker: prefer goal-floor exits and first boarding into empty reliable elevators.
                # These bonuses are small enough to flip only near-ties, fixing routing order issues
                # without overriding clear search preferences.
                if action.startswith('EXIT{'):
                    pid, eid = map(int, action[5:-1].split(','))
                    if elev_floors.get(eid) == self.game.get_person_goal(pid):
                        expected_utility += 0.02  # delivering at goal – always prefer
                elif action.startswith('ENTER{'):
                    pid, eid = map(int, action[6:-1].split(','))
                    if (self.elevator_action_prob[eid] >= 0.7 and elev_weights.get(eid, 0) == 0):
                        expected_utility += 0.006  # board empty reliable elevator first

                if expected_utility > current_best_value:
                    current_best_value = expected_utility
                    current_best_action = action

            # Successfully completed this depth layer
            best_action = current_best_action
            depth += 1

            if depth > 4:
                break

        return best_action