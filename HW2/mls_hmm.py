import sys

def main():
    evidence = sys.argv[1:]
    if not evidence:
        print("No evidence")
        return
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
    
    v = []
    backpointer = []

    v0 = {}
    bp0 = {}
    for x in [True, False]:
        v0[x] = prior[x] * sensor[x][e[0]]
        bp0[x] = None
    v.append(v0)
    backpointer.append(bp0)

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
    
    last_state = max(v[n-1], key=v[n-1].get)
    path = [None] * n
    path[n-1] = last_state
    
    for t in range(n-1, 0, -1):
        path[t-1] = backpointer[t][path[t]]
    
    result = ['T' if state else 'F' for state in path]
    print(result)

if __name__ == '__main__':
    main()