import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 mdp_vi.py <reward>")
        return
    r = float(sys.argv[1])
    gamma = 0.9
    epsilon = 1e-6

    # Define grid layout
    blocked = (1, 4)
    terminals = {(4, 4): 1.0, (3, 2): -1.0}
    states = []
    for row in range(1, 5):
        for col in range(1, 5):
            if (row, col) == blocked or (row, col) in terminals:
                continue
            states.append((row, col))

    # Initialize utilities
    U = {s: 0.0 for s in states}
    U.update(terminals)

    action_map = {
        'Up': [('Up', 0.8), ('Left', 0.1), ('Right', 0.1)],
        'Left': [('Left', 0.8), ('Up', 0.1), ('Down', 0.1)],
        'Down': [('Down', 0.8), ('Left', 0.1), ('Right', 0.1)],
        'Right': [('Right', 0.8), ('Up', 0.1), ('Down', 0.1)]
    }

    while True:
        delta = 0
        U_new = U.copy()
        for s in states:
            max_expected = -float('inf')
            for action in action_map:
                total = 0.0
                for direction, prob in action_map[action]:
                    nc, nr = s[0], s[1]
                    if direction == 'Up': nr += 1
                    elif direction == 'Down': nr -= 1
                    elif direction == 'Left': nc -= 1
                    elif direction == 'Right': nc += 1

                    # Check boundaries and blocked
                    if not (1 <= nr <= 4 and 1 <= nc <= 4) or (nc, nr) == blocked:
                        nc, nr = s[0], s[1]
                    s_prime = (nc, nr)
                    util = terminals.get(s_prime, U[s_prime])
                    total += prob * util

                if total > max_expected:
                    max_expected = total

            new_u = r + gamma * max_expected
            delta = max(delta, abs(new_u - U[s]))
            U_new[s] = new_u

        U = U_new
        if delta < epsilon * (1 - gamma) / gamma:
            break

    # Determine optimal policy
    policy = {}
    for s in states:
        best_action = None
        best_value = -float('inf')
        for action in action_map:
            total = 0.0
            for direction, prob in action_map[action]:
                nc, nr = s[0], s[1]
                if direction == 'Up': nr += 1
                elif direction == 'Down': nr -= 1
                elif direction == 'Left': nc -= 1
                elif direction == 'Right': nc += 1

                if not (1 <= nr <= 4 and 1 <= nc <= 4) or (nc, nr) == blocked:
                    nc, nr = s[0], s[1]
                s_prime = (nc, nr)
                util = terminals.get(s_prime, U[s_prime])
                total += prob * util

            if total > best_value:
                best_value = total
                best_action = action
        policy[s] = best_action

    # Output results
    print("State pi* U")
    for s in sorted(states, key=lambda x: (x[0], x[1])):
        print(f"({s[0]}, {s[1]}) {policy[s][0]} {U[s]:.4f}")

if __name__ == "__main__":
    main()