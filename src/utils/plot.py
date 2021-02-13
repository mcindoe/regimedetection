PALETTE = [
    "#7F2CCB",
    "#98473E",
    "#B4B8C5",
    "#FFD400",
    "#7C8483",
    "#119DA4",
    "#F29559",
    "#84DD63",
    "#F7A1C4",
]


def make_colours(n_colours):
    """
    Generates a colour palette (list of hex codes) of a given
    length. Uses as many as possible from the predefined
    palette first, which have been chosen to be contrasting,
    and then fills the rest with random hex codes.
    """
    if n_colours < len(PALETTE):
        return PALETTE[:n_colours]

    palette = PALETTE.copy()
    remaining_colours = n_colours - len(palette)
    for _ in range(remaining_colours):
        colour = "#" + "".join(random.choices("0123456789ABCD", k=6))
        palette.append(colour)
    return palette
