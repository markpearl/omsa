import numpy as np
import matplotlib.pyplot as plt

def splex(start_basis, A, b, c):
    idx = [int(val-1) for val in start_basis]
    count = 0
    optimal = False
    while not optimal:
        count += 1
        print('Starting Iteration {}'.format(count))
        print('----------------------------------------------------------------')
        print('Basis B = [A_{}, A_{}, A_{}]:\n{}'.format(idx[0]+1, idx[1]+1, idx[2]+1, A[:, idx]))
        B_inv = np.linalg.inv(A[:, idx])
        print('\nB^-1:\n{}\n'.format(np.round(B_inv, 2)))
        x = np.zeros(A.shape[1])
        x_B = (B_inv @ b).reshape((3,))
        check = [True if item >= 0 else False for item in x_B]
        if sum(check) == len(check):
            print('Solution from this Basis is Feasible.  Continuing...\n')
        else:
            print('Infeasible or Degenerate solution found...')
            break
        x_N_idx = np.setdiff1d(list(range(A.shape[1])), idx)
        x[idx] = x_B
        plt.plot(x[0], x[1], 'o', color='black')
        plt.annotate('({:.2f}, {:.2f})'.format(x[0], x[1]),
                     (x[0] - 0.45, x[1] + 0.25),
                     color='maroon',
                     weight='bold')
        plt.annotate('Obj: {:.2f}'.format(c[idx].T @ x[idx]),
                     (x[0] - 0.4, x[1] - 0.5),
                     color='purple', weight='bold')
        plt.pause(1.5)
        x_Btext = '['
        x_Ntext = '['
        for col in idx:
            if col != idx[-1]:
                x_Btext += 'x_{}, '.format(col+1)
            else:
                x_Btext += 'x_{}]'.format(col+1)
        for col in x_N_idx:
            if col != x_N_idx[-1]:
                x_Ntext += 'x_{}, '.format(col+1)
            else:
                x_Ntext += 'x_{}]'.format(col+1)
        print('x_B = {} = {}'.format(x_Btext, np.round(x_B.T, 2)))
        print('x_N = {} = [0. 0.]'.format(x_Ntext))

        # Find reduced cost for nonbasic variables
        cp = c[x_N_idx[0]] - c[idx] @ B_inv @ A[:, x_N_idx[0]]
        cj = c[x_N_idx[1]] - c[idx] @ B_inv @ A[:, x_N_idx[1]]
        print(u'c\u0305_{} = {:.2f} and c\u0305_{} = {:.2f}'.format(x_N_idx[0] + 1, cp, x_N_idx[1] + 1, cj))
        # Check for improving direction
        if cp < 0 or cj < 0:
            c_min_idx = np.argmin([cp, cj])
            enter_idx = x_N_idx[c_min_idx]
            print('\nImprovement is possible and x_{} enters the basis.\n'.format(enter_idx+1))
        else:
            print('\nNo improving direction.  Current solution is optimal.')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Optimal solution is x = {}'.format(np.round(x, 2)))
            print('The optimal cost is {:.2f}'.format(c.T@x))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            break
        # Calculate d_B
        d_B = -1*B_inv@A[:, enter_idx]
        idx_neg = np.flatnonzero(d_B < 0)
        print('d_B = {}'.format(d_B))
        if (d_B >= 0).all():
            print('Unbounded optimum.  Terminating...')
            break
        else:
            r = -1 * (x_B[idx_neg] / d_B[idx_neg])
            theta = np.min(r)
            exit_idx = idx[idx_neg[np.argmin(r)]]
            print(u'Ratio test for optimal \u03B8: min({}) = {:.2f}'.format(np.round(r, 2), theta))
            print('x_{} will exit the basis.\n'.format(exit_idx+1))
        x[idx] += theta*d_B
        # Guarantee no rounding/truncation issues for variables entering or exiting the basis
        x[exit_idx] = 0
        x[enter_idx] = 1
        # Update basis indices
        idx = np.where(np.array(x) > 0)[0]
    return NotImplemented