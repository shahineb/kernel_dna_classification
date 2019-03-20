catalog = dict()


def penalty(marker):
    """Workaround wrapper to map each penalty to a string

    Args:
        marker (str)
    """
    global generator_catalog

    def wrapper(func):
        func._marker = marker
        catalog[marker] = func
        return func
    return wrapper


@penalty("linear")
def linear(d, e):
    def wrapper(n):
        return 0 if n == 0 else d + e * (n - 1)
    return wrapper
