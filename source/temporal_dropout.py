import rp


def temporal_dropout_boolean_list(length, proportion=0.5):
    """
    Creates a boolean list with temporal dropout patterns.

    Generates a list of length 'length' containing boolean values where
    some values are randomly dropped out (set to False) according to the
    specified proportion, tending torwards contiguous chunks.

    Args:
        length (int): The length of the output list.
        proportion (float, optional): Target proportion of True values.
            Defaults to 0.5.

    Returns:
        list: Boolean list. Guaranteed to contain at least one True value.
    """
    num_splits = rp.random_int(0, length - 1)
    indices = [0] + rp.random_batch(range(1, length), num_splits, retain_order=True)
    exponent = (
        1 / proportion - 1
    )  # Inverse of 1/(x+1), see https://chatgpt.com/share/681313ce-8eb8-8006-90e2-f4b3012a3cb4
    probability = rp.random_float() ** exponent
    keeps = [rp.random_chance(probability) for _ in indices]

    assert len(indices) == len(keeps)

    output = [None] * length
    for index in range(length):
        if indices and index == indices[0]:
            del indices[0]
            keep = keeps.pop(0)
        output[index] = keep

    assert None not in output

    if not any(output):
        # Don't allow all 0's
        output[rp.random_index(output)] = True

    return output


@rp.globalize_locals
def demo_temporal_dropout_boolean_list(proportion=0.25):
    height, width = 100, 49

    rows = []
    for _ in range(height):
        row = temporal_dropout_boolean_list(width, proportion)
        rows.append(row)

    rows = sorted(rows, key=sum, reverse=True)
    rows = rp.as_numpy_array(rows)

    print(rows.mean())
    scale = 5
    graph = rp.cv_line_graph(
        y_values=rp.np.arange(len(rows))[::-1] * scale,
        x_values=list(map(sum, rows)),
        height=height * scale,
        width=width,
        background_color="dark cyan",
        line_color="light green",
    )
    preview_image = rows
    preview_image = rp.vertically_concatenated_images(
        [
            rp.blend_images(
                "dark dark altbw random blue", "white white white randomgray", row[None]
            )
            for row in rows
        ]
    )
    preview_image = rp.cv_resize_image(preview_image, scale, interp="nearest")  #
    preview_image = rp.horizontally_concatenated_images(preview_image, graph)
    preview_image = rp.labeled_image(
        preview_image,
        f"mean={rows.mean():.3}",
        font="R:Futura",
        background_color="dark blue",
        size=0.05,
    )

    rp.save_image(preview_image, "demo_temporal_dropout_boolean_list.png")
    rp.display_image(preview_image)


if __name__ == "__main__":
    demo_temporal_dropout_boolean_list()
