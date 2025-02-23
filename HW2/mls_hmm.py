#!/usr/bin/env python3
import sys

def main():
    # Read evidence from command-line arguments.
    evidence = sys.argv[1:]
    if not evidence:
        print("Usage: python3 mls_hmm.py <evidence as T or F>")
        sys.exit(1)
    e = [x.upper() == 'T' for x in evidence]
    n = len(e)
    
    # HMM parameters:
    prior = {True: 0.5, False: 0.5}
    trans = {
        True: {True: 0.7, False: 0.3},
        False: {True: 0.4, False: 0.6}
    }
    sensor = {
        True: {True: 0.9, False: 0.1},
        False: {True: 0.3, False: 0.7}
    }
    
    # v[t] holds the maximum probability of any path ending in each state at time t.
    # backpointer[t] stores the previous state leading to the best path.
    v = []
    backpointer = []
    
    # Time t = 1: consider all possibilities from X0.
    v1 = {}
    bp1 = {}
    for x in [True, False]:
        best_prob = 0.0
        best_prev = None
        for prev in [True, False]:
            prob = prior[prev] * trans[prev][x] * sensor[x][e[0]]
            if prob > best_prob:
                best_prob = prob
                best_prev = prev
        v1[x] = best_prob
        bp1[x] = best_prev  # This backpointer points to the best X0 for state x at time 1.
    v.append(v1)
    backpointer.append(bp1)
    
    # For t = 2 to n.
    for t in range(1, n):
        vt = {}
        bpt = {}
        for x in [True, False]:
            best_prob = 0.0
            best_prev = None
            for prev in [True, False]:
                prob = v[t-1][prev] * trans[prev][x] * sensor[x][e[t]]
                if prob > best_prob:
                    best_prob = prob
                    best_prev = prev
            vt[x] = best_prob
            bpt[x] = best_prev
        v.append(vt)
        backpointer.append(bpt)
    
    # Backtrack: find the state at time n with the highest probability.
    last_state = max(v[n-1], key=v[n-1].get)
    path = [None] * n
    path[n-1] = last_state
    
    # Reconstruct the most likely path backwards.
    for t in range(n-1, 0, -1):
        path[t-1] = backpointer[t][path[t]]
    
    # Convert boolean states to 'T'/'F' for printing.
    result = ['T' if state else 'F' for state in path]
    print(result)

if __name__ == '__main__':
    main()
