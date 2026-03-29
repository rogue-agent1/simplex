#!/usr/bin/env python3
"""simplex - Simplex method for linear programming."""
import sys

def solve(c, A, b):
    """Maximize c^T x subject to Ax <= b, x >= 0. Returns (optimal_value, x)."""
    m, n = len(A), len(c)
    # Build tableau with slack variables
    tab = []
    for i in range(m):
        row = list(A[i]) + [0]*m + [b[i]]
        row[n + i] = 1
        tab.append(row)
    obj = [-ci for ci in c] + [0]*m + [0]
    tab.append(obj)
    basis = list(range(n, n + m))
    cols = n + m
    for _ in range(1000):
        # Find pivot column (most negative in obj row)
        pivot_col = min(range(cols), key=lambda j: tab[-1][j])
        if tab[-1][pivot_col] >= -1e-9: break
        # Find pivot row (minimum ratio test)
        pivot_row = -1
        min_ratio = float('inf')
        for i in range(m):
            if tab[i][pivot_col] > 1e-9:
                ratio = tab[i][-1] / tab[i][pivot_col]
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = i
        if pivot_row == -1: return None, None  # unbounded
        # Pivot
        piv = tab[pivot_row][pivot_col]
        tab[pivot_row] = [x / piv for x in tab[pivot_row]]
        for i in range(m + 1):
            if i != pivot_row:
                factor = tab[i][pivot_col]
                tab[i] = [tab[i][j] - factor * tab[pivot_row][j] for j in range(cols + 1)]
        basis[pivot_row] = pivot_col
    x = [0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tab[i][-1]
    return tab[-1][-1], x

def test():
    # Max 3x + 2y s.t. x + y <= 4, x + 3y <= 6, x,y >= 0
    val, x = solve([3, 2], [[1, 1], [1, 3]], [4, 6])
    assert abs(val - 12.0) < 1e-6  # x=4, y=0
    assert abs(x[0] - 4.0) < 1e-6
    assert abs(x[1] - 0.0) < 1e-6
    # Max x + y s.t. x <= 2, y <= 3
    val2, x2 = solve([1, 1], [[1, 0], [0, 1]], [2, 3])
    assert abs(val2 - 5.0) < 1e-6
    print("simplex: all tests passed")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("Usage: simplex.py --test")
