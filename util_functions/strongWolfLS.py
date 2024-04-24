import numpy as np

def strongwolfe(f, d, x0, alpham):
    alpha0 = 0
    alphap = alpha0
    c1 = 1e-4
    c2 = 0.5
    alphax = alpham * np.random.rand(1)
    fx0, gx0 = f(x0)
    fxp = fx0
    gxp = gx0
    i = 1
    max_iters = 50

    while i < max_iters:
        xx = x0 + alphax * d
        fxx, gxx = f(xx)

        if (fxx > fx0 + c1 * alphax * np.dot(gx0.T, d)) or ((i > 1) and (fxx >= fxp)):
            alphas = zoom(f, x0, d, alphap, alphax)
            return alphas

        if abs(np.dot(gxx.T, d)) <= -c2 * np.dot(gx0.T, d):
            alphas = alphax
            return alphas

        if np.dot(gxx.T, d) >= 0:
            alphas = zoom(f, x0, d, alphax, alphap)
            return alphas

        alphap = alphax
        fxp = fxx
        alphax = alphax + (alpham - alphax) * np.random.rand(1)
        alphas = alphax
        i += 1

    return alphas


def zoom(f, x0, d, alphal, alphah):
    c1 = 1e-4
    c2 = 0.5
    fx0, gx0 = f(x0)
    i = 1
    max_iters = 10

    while i < max_iters:
        alphax = 0.5 * (alphal + alphah)
        xx = x0 + alphax * d
        fxx, gxx = f(xx)
        xl = x0 + alphal * d
        fxl, _ = f(xl)
        descend = fx0 + c1 * alphax * np.dot(gx0.T, d)

        if (fxx > descend) or (fxx >= fxl):
            alphah = alphax
        else:
            if abs(np.dot(gxx.T, d)) <= -c2 * np.dot(gx0.T, d):
                alphas = alphax
                return alphas

            if np.dot(gxx.T, d) * (alphah - alphal) >= 0:
                alphah = alphal

            alphal = alphax

        alphas = alphax
        i += 1

    return alphas