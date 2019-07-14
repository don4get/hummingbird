class AlphaFilter:
    """Discrete-time realization of first order low pass filter.

        # y[k] = alpha * y[k-1] + (1 - alpha) * u[k]

        .. seealso:: https://en.wikipedia.org/wiki/Low-pass_filter#Discrete-time_realization

        :param alpha: approximated time constant [s]
        :param y0: initial state
    """
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.y * self.alpha + u * (1.0 - self.alpha)
        return self.y
