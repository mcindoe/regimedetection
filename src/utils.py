def closest_even_integer(x):
    '''
    Computes closest even integer to an input float or integer.
    If x is an integer, the output is x+1 if x is odd else x
    '''
    if not isinstance(x, (int, float)):
        raise ValueError(f'Expected {x} to be an integer or a float')

    if isinstance(x, int):
        if x%2 == 0:
            return x
        return x+1

    lower = floor(x)
    upper = ceil(x)

    # If the closest integer to x is smaller than x
    if abs(lower - x) < abs(upper - x):
        return lower if lower%2 == 0 else upper
    return upper if upper%2 == 0 else lower


