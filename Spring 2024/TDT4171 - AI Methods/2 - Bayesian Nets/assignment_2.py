import numpy as np

def main():
    A = 1/3 * np.array(
        [
            [0, .5, .5], 
            [0, 0, 1], 
            [0, 1, 0],
            [0, 0, 1],
            [.5, 0, .5],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [.5, .5, 0]
        ])

    B = 1/2 * np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
    )

    def compute_proba(M, C, O):
        if M == O: return 0
        p = 1/3 * A[M * 3 + C, O] / B[M, O]
        return p

    dic = {
        0: 'A',
        1: 'B',
        2: 'C'
    }

    P = []
    for M in range(3):
        for O in range(3):
            P.append([])
            for C in range(3):
                P[M*3 + O].append(compute_proba(M, C, O))
                print(f"P(C={dic[C]} | O={dic[O]}, M={dic[M]}) = {compute_proba(M, C, O)}")

if __name__=="__main__":
    print("Solve monty hall problem")
    main()