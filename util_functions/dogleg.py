import numpy as np


def dog_leg(grad, B, delta):
    """
    Calculate the dog-leg solution for the trust-region subproblem.
    """
    # Solving the linear system B * pB = -grad
    pB = -np.linalg.solve(B, grad)

    if np.linalg.norm(pB) <= delta:
        print("Newton step selected")
        return pB
    else:
        gBg = grad.T @ B @ grad
        pu = -grad.T @ grad / gBg * grad

        # Obtain step size
        c = np.linalg.norm(pu) ** 2 - delta**2
        a = np.linalg.norm(pB - pu) ** 2
        b = 2 * pu.T @ (pB - pu)

        if np.linalg.norm(pu) >= delta:
            print("Cauchy point selected")
            return -delta * grad / np.linalg.norm(grad)
        else:
            print("Dogleg step selected")
            t1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            t2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

            t = max(t1, t2)
            return pu + t * (pB - pu)
