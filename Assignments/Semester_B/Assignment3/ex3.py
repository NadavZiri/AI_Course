import ext_elev
import time
import math

id = ["211388921"]


class Controller:
    """UCB exploration-exploitation controller for the RL multi-elevator domain.

    The model is hidden: elevator/person success probabilities and per-person
    reward distributions are unknown. We learn them online by observing state
    transitions and rewards, then plan using our learned estimates with UCB
    bonuses to encourage exploration of under-sampled entities.

    AI assistance used: Claude Sonnet 4.6 was used to help design and implement
    the UCB learning mechanism and adapt the ex2.py planning code.
    """

    def __init__(self, game: ext_elev.GameAPI):
        self.game = game
        self.capacities = game.get_capacities()
        self.reachable = game.get_reachable()
        self.goal_reward = game.get_goal_reward()

        initial_state = game.get_initial_state()
        initial_elevators, initial_persons, _ = initial_state

        # Cache static person info
        self.person_weight = {}
        self.person_goal = {}
        for pid, _ in initial_persons:
            self.person_weight[pid] = game.get_person_weight(pid)
            self.person_goal[pid] = game.get_person_goal(pid)
        self.n_persons = len(initial_persons)

        # --- Learning: elevator MOVE success rates ---
        # Optimistic prior: 1 success / 1 attempt => mean = 1.0 initially.
        # Resets quickly on first observed failure.
        self.elev_successes = {eid: 1 for eid, _, _ in initial_elevators}
        self.elev_attempts  = {eid: 1 for eid, _, _ in initial_elevators}

        # --- Learning: person ENTER/EXIT success rates ---
        self.person_successes = {pid: 1 for pid, _ in initial_persons}
        self.person_attempts  = {pid: 1 for pid, _ in initial_persons}

        # --- Learning: per-person delivery rewards ---
        self.reward_sum = {pid: 0.0 for pid, _ in initial_persons}
        self.reward_n   = {pid: 0   for pid, _ in initial_persons}

        # UCB exploration constants
        self.C_elev   = 0.5
        self.C_person = 0.5
        self.C_reward = 2.0

        # Step counter (shared time for UCB log(t) term)
        self.t = 0

        # Previous step bookkeeping for learning updates
        self.prev_state  = None
        self.prev_action = None

        # All floors reachable by any elevator
        self.all_floors = set()
        for rs in self.reachable.values():
            self.all_floors.update(rs)

        # Floyd-Warshall min expected moves between floors
        self._rebuild_floyd_warshall()
        self.fw_last_rebuilt = 0

        # Whether reset-farming a single person beats completing full cycles.
        # Recomputed alongside Floyd-Warshall as reward estimates improve.
        self.allow_reset = self._reset_beneficial()

        # Memoization cache (cleared every step)
        self.memo = {}

    # =========================================================================
    # ESTIMATE ACCESSORS
    # =========================================================================

    def _get_elev_prob(self, eid):
        """ML estimate of elevator MOVE success probability."""
        return self.elev_successes[eid] / self.elev_attempts[eid]

    def _get_person_prob(self, pid):
        """ML estimate of person ENTER/EXIT success probability."""
        return self.person_successes[pid] / self.person_attempts[pid]

    def _get_person_reward_estimate(self, pid):
        """Mean observed delivery reward; falls back to goal_reward/n_persons."""
        if self.reward_n[pid] > 0:
            return self.reward_sum[pid] / self.reward_n[pid]
        return self.goal_reward / max(1, self.n_persons)

    def _get_person_reward_ucb(self, pid):
        """UCB-inflated reward estimate to encourage delivery exploration."""
        mean  = self._get_person_reward_estimate(pid)
        bonus = self.C_reward * math.sqrt(
            math.log(self.t + 2) / max(1, self.reward_n[pid])
        )
        return mean + bonus

    def _ucb_bonus_for_action(self, action):
        """Additive UCB bonus based on how little we have sampled each entity."""
        if action == "RESET":
            return 0.0
        log_t = math.log(self.t + 2)
        if action.startswith("MOVE{"):
            eid = int(action[5:action.index(',')])
            return self.C_elev * math.sqrt(log_t / self.elev_attempts[eid])
        elif action.startswith("ENTER{"):
            inner = action[6:-1]
            pid, eid = map(int, inner.split(','))
            return (self.C_person * math.sqrt(log_t / self.person_attempts[pid]) +
                    self.C_elev   * math.sqrt(log_t / self.elev_attempts[eid]))
        elif action.startswith("EXIT{"):
            pid = int(action[5:action.index(',')])
            return self.C_person * math.sqrt(log_t / self.person_attempts[pid])
        return 0.0

    # =========================================================================
    # FLOYD-WARSHALL (rebuilt periodically as estimates improve)
    # =========================================================================

    def _rebuild_floyd_warshall(self):
        """Recompute min expected moves between all floor pairs using ML estimates."""
        inf = float('inf')
        self.min_time = {f1: {f2: inf for f2 in self.all_floors}
                         for f1 in self.all_floors}
        for f in self.all_floors:
            self.min_time[f][f] = 0.0

        for eid, reachable_set in self.reachable.items():
            p_succ    = self._get_elev_prob(eid)
            exp_moves = 1.0 / p_succ
            floors    = list(reachable_set)
            for i in range(len(floors)):
                for j in range(len(floors)):
                    if i != j:
                        f1, f2 = floors[i], floors[j]
                        if exp_moves < self.min_time[f1][f2]:
                            self.min_time[f1][f2] = exp_moves

        for k in self.all_floors:
            for i in self.all_floors:
                for j in self.all_floors:
                    if self.min_time[i][k] != inf and self.min_time[k][j] != inf:
                        cost = self.min_time[i][k] + self.min_time[k][j]
                        if k != i and k != j:
                            cost += 1.5
                        if cost < self.min_time[i][j]:
                            self.min_time[i][j] = cost

    # =========================================================================
    # RESET-FARMING GATE
    # =========================================================================

    def _reset_beneficial(self):
        """Decide whether reset-farming a single person beats full delivery.

        Reset-farming pays off only on "reset-friendly" layouts where one cheap,
        high-reward person can be delivered repeatedly (deliver -> RESET -> repeat)
        for a higher per-step reward than completing the whole delivery cycle
        (which additionally earns goal_reward). We compare the best single-person
        farm rate against the full-cycle completion rate, both estimated from the
        initial state using current learned reward/probability estimates.

        When farming does NOT win we suppress RESET entirely (see
        get_legal_actions), which stops the planner from wastefully looping on a
        cheap person instead of finishing deliveries.
        """
        init_elevs, init_persons, _ = self.game.get_initial_state()
        elev_floors = {eid: f for eid, f, w in init_elevs}

        best_farm_rate = 0.0
        cycle_reward   = self.goal_reward
        cycle_time     = 0.0

        for pid, loc in init_persons:
            p_goal  = self.person_goal[pid]
            p_prob  = self._get_person_prob(pid)
            p_floor = loc[1] if loc[0] == 'floor' else elev_floors.get(loc[1])
            t = self.get_cost_on_floor(p_floor, p_goal, elev_floors, p_prob)
            if t == float('inf'):
                continue
            r = self._get_person_reward_estimate(pid)
            cycle_reward += r
            cycle_time   += t
            farm_rate = r / (t + 1.0)   # +1 step for the RESET action
            if farm_rate > best_farm_rate:
                best_farm_rate = farm_rate

        if cycle_time <= 0:
            return True
        return best_farm_rate > (cycle_reward / cycle_time)

    # =========================================================================
    # ONLINE LEARNING UPDATE
    # =========================================================================

    def _update_from_observation(self, current_state):
        """Update estimates from transition (prev_state, prev_action) → current_state.

        Called at the TOP of choose_next_action so `current_state` is the result
        of applying `self.prev_action` to `self.prev_state`.
        """
        if self.prev_state is None or self.prev_action is None:
            return

        action = self.prev_action
        if action == "RESET":
            return

        prev_elevs, prev_persons, prev_total = self.prev_state
        curr_elevs, curr_persons, _          = current_state

        action_type = action[:action.index('{')]
        args_str    = action[action.index('{') + 1:-1]
        arg1, arg2  = map(int, args_str.split(','))

        last_reward = self.game.get_last_gained_reward()

        if action_type == "MOVE":
            eid, target_f = arg1, arg2
            self.elev_attempts[eid] += 1
            curr_floor = next((f for e, f, w in curr_elevs if e == eid), None)
            if curr_floor == target_f:
                self.elev_successes[eid] += 1

        elif action_type == "ENTER":
            pid, eid = arg1, arg2
            self.person_attempts[pid] += 1
            curr_loc = next((loc for p, loc in curr_persons if p == pid), None)
            if curr_loc == ('in', eid):
                self.person_successes[pid] += 1

        elif action_type == "EXIT":
            pid, eid = arg1, arg2
            self.person_attempts[pid] += 1
            new_loc = next((loc for p, loc in curr_persons if p == pid), None)

            if new_loc is None:
                # Person delivered; episode did NOT reset (prev_total > 1).
                # last_reward is the pure delivery reward.
                self.person_successes[pid] += 1
                if last_reward > 0:
                    self.reward_sum[pid] += last_reward
                    self.reward_n[pid]   += 1

            elif new_loc[0] == 'floor':
                if last_reward > 0:
                    # Delivery of the last person triggered an episode reset.
                    # State snapped to initial so pid re-appears at start floor.
                    # last_reward = delivery_reward + goal_reward; subtract goal.
                    self.person_successes[pid] += 1
                    pure = last_reward - self.goal_reward
                    if pure > 0:
                        self.reward_sum[pid] += pure
                        self.reward_n[pid]   += 1
                else:
                    # Person stepped out at a non-goal floor (success, no reward).
                    self.person_successes[pid] += 1

            # else: person still ('in', eid) → EXIT failed; don't increment successes

    # =========================================================================
    # PLANNING (adapted from ex2.py; all API probability/reward calls replaced)
    # =========================================================================

    def get_legal_actions(self, state):
        elevators_t, persons_t, _ = state
        # RESET only offered when reset-farming actually beats full delivery;
        # otherwise it is suppressed so the planner commits to completing cycles.
        actions = ["RESET"] if self.allow_reset else []

        elev_info = {eid: {'floor': f, 'weight': w} for eid, f, w in elevators_t}
        useful_targets = {eid: set() for eid in elev_info}

        for pid, loc in persons_t:
            p_goal        = self.person_goal[pid]
            person_weight = self.person_weight[pid]

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
                    e_floor_in = elev_info[eid_in]['floor']
                    for eid2, info2 in elev_info.items():
                        if (eid2 != eid_in
                                and e_floor_in in self.reachable[eid2]
                                and p_goal in self.reachable[eid2]
                                and info2['weight'] + person_weight <= self.capacities[eid2]
                                and self._get_elev_prob(eid2) >= self._get_elev_prob(eid_in)):
                            useful_targets[eid2].add(e_floor_in)
                else:
                    best_time = min(
                        (self.min_time[f].get(p_goal, float('inf'))
                         for f in self.reachable[eid_in]),
                        default=float('inf')
                    )
                    if best_time < float('inf'):
                        transfer_floors = {
                            f for f in self.reachable[eid_in]
                            if self.min_time[f].get(p_goal, float('inf')) == best_time
                        }
                        useful_targets[eid_in].update(transfer_floors)
                        for t_floor in transfer_floors:
                            for eid2, info2 in elev_info.items():
                                if eid2 != eid_in and t_floor in self.reachable[eid2]:
                                    if info2['weight'] + person_weight <= self.capacities[eid2]:
                                        useful_targets[eid2].add(t_floor)

        for eid, info in elev_info.items():
            current_floor = info['floor']
            for target_floor in useful_targets[eid]:
                if target_floor != current_floor:
                    actions.append(f"MOVE{{{eid},{target_floor}}}")

        for pid, loc in persons_t:
            person_weight = self.person_weight[pid]
            p_goal        = self.person_goal[pid]

            if loc[0] == 'floor':
                for eid, info in elev_info.items():
                    if (info['floor'] == loc[1] and
                            info['weight'] + person_weight <= self.capacities[eid]):
                        current_dist = self.min_time[loc[1]].get(p_goal, float('inf'))
                        if any(self.min_time[f].get(p_goal, float('inf')) < current_dist
                               for f in self.reachable[eid]):
                            actions.append(f"ENTER{{{pid},{eid}}}")

            elif loc[0] == 'in':
                actions.append(f"EXIT{{{pid},{loc[1]}}}")

        return actions

    def get_transitions(self, state, action):
        elevators_t, persons_t, total_persons_remaining = state
        if action == "RESET":
            return [(1.0, self.game.get_initial_state(), 0.0)]

        action_parts = action.split("{")
        action_type  = action_parts[0]
        args_str     = action_parts[1].rstrip("}")
        arg1, arg2   = map(int, args_str.split(","))

        transitions = []

        if action_type == "ENTER":
            p, e = arg1, arg2
            prob_success = self._get_person_prob(p)

            if prob_success < 1.0:
                transitions.append((1.0 - prob_success, state, 0.0))

            new_elevs   = list(elevators_t)
            new_persons = list(persons_t)

            for i, (pid, loc) in enumerate(new_persons):
                if pid == p and loc[0] == 'floor':
                    new_persons[i] = (pid, ('in', e))
                    break

            p_weight = self.person_weight[p]
            for i, (eid, cur_f, cur_w) in enumerate(new_elevs):
                if eid == e:
                    new_elevs[i] = (eid, cur_f, cur_w + p_weight)
                    break

            success_state = (tuple(new_elevs), tuple(new_persons), total_persons_remaining)
            transitions.append((prob_success, success_state, 0.0))

        elif action_type == "EXIT":
            p, e = arg1, arg2
            prob_success = self._get_person_prob(p)

            if prob_success < 1.0:
                transitions.append((1.0 - prob_success, state, 0.0))

            new_elevs   = list(elevators_t)
            new_persons = list(persons_t)

            current_floor = next(f for eid, f, w in new_elevs if eid == e)
            p_weight      = self.person_weight[p]
            p_goal        = self.person_goal[p]

            for i, (eid, f, w) in enumerate(new_elevs):
                if eid == e:
                    new_elevs[i] = (eid, f, w - p_weight)
                    break

            reward             = 0.0
            new_total_remaining = total_persons_remaining

            if current_floor == p_goal:
                new_persons    = [person for person in new_persons if person[0] != p]
                reward         = self._get_person_reward_ucb(p)
                new_total_remaining -= 1

                if new_total_remaining == 0:
                    reward       += self.goal_reward
                    success_state = self.game.get_initial_state()
                    transitions.append((prob_success, success_state, reward))
                    return transitions
            else:
                for i, (pid, loc) in enumerate(new_persons):
                    if pid == p:
                        new_persons[i] = (pid, ('floor', current_floor))
                        break

            success_state = (tuple(new_elevs), tuple(new_persons), new_total_remaining)
            transitions.append((prob_success, success_state, reward))

        elif action_type == "MOVE":
            e, target_f   = arg1, arg2
            prob_success  = self._get_elev_prob(e)
            current_floor = next(f for eid, f, w in elevators_t if eid == e)

            new_elevs_succ = list(elevators_t)
            for i, (eid, f, w) in enumerate(new_elevs_succ):
                if eid == e:
                    new_elevs_succ[i] = (eid, target_f, w)
                    break
            success_state = (tuple(new_elevs_succ), persons_t, total_persons_remaining)
            transitions.append((prob_success, success_state, 0.0))

            if prob_success < 1.0:
                transitions.append((1.0 - prob_success, state, 0.0))

        return transitions

    def get_cost_on_floor(self, p_floor, p_goal, elev_floors, p_action_prob):
        best_cost = float('inf')

        for eid, e_floor in elev_floors.items():
            if p_floor in self.reachable[eid]:
                ep = self._get_elev_prob(eid)
                arrive = 0.0 if e_floor == p_floor else (1.0 / ep)

                if p_goal in self.reachable[eid]:
                    travel = 0.0 if p_floor == p_goal else (1.0 / ep)
                    cost   = arrive + travel
                else:
                    best_drop = float('inf')
                    for f in self.reachable[eid]:
                        dist_f = 0.0 if f == p_floor else (1.0 / ep)
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
        goal_slice  = self.goal_reward / total_remaining

        for pid, loc in persons_t:
            expected_reward = self._get_person_reward_ucb(pid) + goal_slice
            p_action_prob   = self._get_person_prob(pid)
            p_goal          = self.person_goal[pid]

            expected_time = float('inf')

            if loc[0] == 'floor':
                expected_time = self.get_cost_on_floor(
                    loc[1], p_goal, elev_floors, p_action_prob
                )

            elif loc[0] == 'in':
                eid     = loc[1]
                e_floor = elev_floors[eid]
                ep      = self._get_elev_prob(eid)

                if p_goal in self.reachable[eid]:
                    travel        = 0.0 if e_floor == p_goal else (1.0 / ep)
                    exit_cost     = 1.0 / p_action_prob
                    expected_time = travel + exit_cost
                else:
                    best_transfer = float('inf')
                    for f in self.reachable[eid]:
                        drop_time   = (0.0 if e_floor == f
                                       else (1.0 / ep)) + (1.0 / p_action_prob)
                        future_time = self.get_cost_on_floor(
                            f, p_goal, elev_floors, p_action_prob
                        )
                        if drop_time + future_time < best_transfer:
                            best_transfer = drop_time + future_time
                    expected_time = best_transfer

            if expected_time != float('inf'):
                total_value += expected_reward * (0.99 ** expected_time)

        return total_value

    def expectimax(self, state, depth):
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
        max_value     = float('-inf')

        for action in legal_actions:
            action_value = 0
            for prob, next_state, reward in self.get_transitions(state, action):
                action_value += prob * (reward + 0.99 * self.expectimax(next_state, depth - 1))
            if action_value > max_value:
                max_value = action_value

        self.memo[state_key] = max_value if max_value != float('-inf') else 0
        return self.memo[state_key]

    # =========================================================================
    # MAIN ACTION SELECTION
    # =========================================================================

    def choose_next_action(self, state):
        start_time  = time.time()
        time_budget = 0.45

        # 1. Increment step counter (used in UCB log(t) terms)
        self.t += 1

        # 2. Update estimates from last observed transition
        self._update_from_observation(state)

        # 3. Rebuild Floyd-Warshall if estimates have changed significantly
        if self.t == 1 or (self.t - self.fw_last_rebuilt) >= 15:
            self._rebuild_floyd_warshall()
            self.fw_last_rebuilt = self.t
            self.allow_reset = self._reset_beneficial()

        # 4. Clear memoization cache (estimates changed since last step)
        self.memo = {}

        legal_actions = self.get_legal_actions(state)
        if not legal_actions:
            self._save_prev(state, "RESET")
            return "RESET"
        if len(legal_actions) == 1:
            self._save_prev(state, legal_actions[0])
            return legal_actions[0]

        best_action = "RESET"

        elevators_t, _, _ = state
        elev_floors  = {eid: f for eid, f, _ in elevators_t}
        elev_weights = {eid: w for eid, _, w in elevators_t}

        depth = 1
        while True:
            current_best_action = None
            current_best_value  = float('-inf')

            for action in legal_actions:
                if time.time() - start_time > time_budget:
                    chosen = best_action if best_action != "RESET" else legal_actions[0]
                    self._save_prev(state, chosen)
                    return chosen

                expected_utility = 0.0
                for prob, next_state, reward in self.get_transitions(state, action):
                    expected_utility += prob * (reward + 0.99 * self.expectimax(next_state, depth - 1))

                # Tiebreakers: prefer goal-floor exits and first boarding on reliable elevators
                if action.startswith('EXIT{'):
                    pid, eid = map(int, action[5:-1].split(','))
                    if elev_floors.get(eid) == self.person_goal[pid]:
                        expected_utility += 0.02
                elif action.startswith('ENTER{'):
                    pid, eid = map(int, action[6:-1].split(','))
                    if (self._get_elev_prob(eid) >= 0.7 and elev_weights.get(eid, 0) == 0):
                        expected_utility += 0.006

                # UCB bonus: encourage sampling under-explored elevators/persons
                expected_utility += self._ucb_bonus_for_action(action)

                if expected_utility > current_best_value:
                    current_best_value  = expected_utility
                    current_best_action = action

            best_action = current_best_action
            depth += 1

            if depth > 8:
                break

        self._save_prev(state, best_action)
        return best_action

    def _save_prev(self, state, action):
        self.prev_state  = state
        self.prev_action = action
