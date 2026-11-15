
import cv2
import turtle
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import sys

# ============================================================
#              IMAGE TO TURTLE ART GENERATOR 5.0
#                 MANGA ILLUSTRATION ENGINE
# ============================================================

# ======================= CONFIGURATION =======================

WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 850

MAX_IMAGE_SIZE = 1100

# Image processing
INK_THRESHOLD_OFFSET = 0

CANNY_LOW = 35
CANNY_HIGH = 110

DETAIL_BLOCK_SIZE = 19
DETAIL_C = 8

MIN_FILL_AREA = 25
MIN_LINE_LENGTH = 10

FILL_EPSILON = 0.9
LINE_EPSILON = 0.7
DETAIL_EPSILON = 0.8

# Cross-hatching
ENABLE_HATCHING = True

HATCH_DARK_LIMIT = 65
HATCH_LIGHT_LIMIT = 190

HATCH_SPACING = 9
HATCH_MIN_LENGTH = 12

HATCH_ANGLES = [45, -45]

# Drawing appearance
PEN_COLOR = "black"
PEN_SIZE = 1

# Animation
POINTS_PER_FRAME = 5
FRAME_DELAY = 2

# Feature switches
ENABLE_BLACK_FILLS = True
ENABLE_MAIN_OUTLINES = True
ENABLE_FINE_DETAILS = True

# ============================================================
#                     IMAGE SELECTION
# ============================================================


def select_image():

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select an image for Turtle Art",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return path


# ============================================================
#                     IMAGE LOADING
# ============================================================


def load_image(path):

    raw = np.fromfile(path, dtype=np.uint8)

    img = cv2.imdecode(
        raw,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError("Unable to load the selected image.")

    return img


def resize_image(img):

    h, w = img.shape[:2]

    scale = min(
        MAX_IMAGE_SIZE / w,
        MAX_IMAGE_SIZE / h,
        1.0
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
#                  COMMON PATH BUILDER
# ============================================================


def make_path(
    points,
    color="black",
    fill=False,
    closed=False,
    layer=0,
    width=1
):

    return {
        "points": points,
        "color": color,
        "fill": fill,
        "closed": closed,
        "layer": layer,
        "width": width
    }


# ============================================================
#                    BLACK INK EXTRACTION
# ============================================================


def get_ink_paths(gray):

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    threshold, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    threshold = np.clip(
        threshold + INK_THRESHOLD_OFFSET,
        0,
        255
    )

    _, binary = cv2.threshold(
        blurred,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    paths = []

    def visit(index, depth):

        contour = contours[index]
        area = cv2.contourArea(contour)

        if area >= MIN_FILL_AREA:

            approx = cv2.approxPolyDP(
                contour,
                FILL_EPSILON,
                True
            )

            if len(approx) >= 3:

                color = (
                    "black"
                    if depth % 2 == 0
                    else "white"
                )

                paths.append(
                    make_path(
                        approx.reshape(-1, 2),
                        color=color,
                        fill=True,
                        closed=True,
                        layer=0
                    )
                )

        child = hierarchy[index][2]

        while child != -1:

            visit(child, depth + 1)
            child = hierarchy[child][0]

    for index, item in enumerate(hierarchy):

        if item[3] == -1:
            visit(index, 0)

    return paths


# ============================================================
#                MULTI-SCALE EDGE DETECTION
# ============================================================


def get_main_outline_paths(gray):

    height, width = gray.shape

    blur1 = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    edges1 = cv2.Canny(
        blur1,
        CANNY_LOW,
        CANNY_HIGH
    )

    small = cv2.resize(
        gray,
        None,
        fx=0.75,
        fy=0.75,
        interpolation=cv2.INTER_AREA
    )

    blur2 = cv2.GaussianBlur(
        small,
        (3, 3),
        0
    )

    edges2 = cv2.Canny(
        blur2,
        CANNY_LOW,
        CANNY_HIGH
    )

    edges2 = cv2.resize(
        edges2,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    combined = cv2.bitwise_or(
        edges1,
        edges2
    )

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    paths = []

    for contour in contours:

        length = cv2.arcLength(
            contour,
            False
        )

        if length < MIN_LINE_LENGTH:
            continue

        approx = cv2.approxPolyDP(
            contour,
            LINE_EPSILON,
            False
        )

        if len(approx) < 2:
            continue

        paths.append(
            make_path(
                approx.reshape(-1, 2),
                color="black",
                fill=False,
                closed=False,
                layer=1
            )
        )

    paths.sort(
        key=lambda p: len(p["points"]),
        reverse=True
    )

    return paths


# ============================================================
#                  FINE DETAIL EXTRACTION
# ============================================================


def get_detail_paths(gray):

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        DETAIL_BLOCK_SIZE,
        DETAIL_C
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    paths = []

    for contour in contours:

        length = cv2.arcLength(
            contour,
            False
        )

        if length < MIN_LINE_LENGTH:
            continue

        approx = cv2.approxPolyDP(
            contour,
            DETAIL_EPSILON,
            False
        )

        if len(approx) < 2:
            continue

        paths.append(
            make_path(
                approx.reshape(-1, 2),
                color="black",
                fill=False,
                closed=False,
                layer=2
            )
        )

    paths.sort(
        key=lambda p: len(p["points"]),
        reverse=True
    )

    return paths


# ============================================================
#                  CROSS-HATCHING ENGINE
# ============================================================


def get_hatch_paths(gray):

    height, width = gray.shape

    mask = cv2.inRange(
        gray,
        HATCH_DARK_LIMIT,
        HATCH_LIGHT_LIMIT
    )

    paths = []

    center_x = width / 2
    center_y = height / 2

    for angle in HATCH_ANGLES:

        matrix = cv2.getRotationMatrix2D(
            (center_x, center_y),
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Calculate inverse transformation once.
        inv = cv2.invertAffineTransform(matrix)

        for y in range(0, height, HATCH_SPACING):

            row = rotated[y, :]
            active = np.where(row > 0)[0]

            if len(active) == 0:
                continue

            splits = np.where(
                np.diff(active) > 1
            )[0] + 1

            runs = np.split(active, splits)

            for run in runs:

                if len(run) < HATCH_MIN_LENGTH:
                    continue

                x1 = int(run[0])
                x2 = int(run[-1])

                # Homogeneous coordinates [x, y, 1]
                p1 = inv @ np.array(
                    [x1, y, 1.0],
                    dtype=float
                )

                p2 = inv @ np.array(
                    [x2, y, 1.0],
                    dtype=float
                )

                points = np.array([
                    [
                        int(round(p1[0])),
                        int(round(p1[1]))
                    ],
                    [
                        int(round(p2[0])),
                        int(round(p2[1]))
                    ]
                ])

                paths.append(
                    make_path(
                        points,
                        color="black",
                        fill=False,
                        closed=False,
                        layer=3
                    )
                )

    return paths


# ============================================================
#                   COORDINATE MAPPING
# ============================================================


def map_point(px, py, width, height):

    available_width = WINDOW_WIDTH * 0.88
    available_height = WINDOW_HEIGHT * 0.82

    scale = min(
        available_width / width,
        available_height / height
    )

    x = (px - width / 2) * scale
    y = (height / 2 - py) * scale

    return x, y


# ============================================================
#                      TURTLE SETUP
# ============================================================


def create_turtle():

    screen = turtle.Screen()

    screen.setup(
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    screen.bgcolor("white")
    screen.title("Image to Turtle Art 5.0")

    screen.tracer(0, 0)

    pen = turtle.Turtle()

    pen.hideturtle()
    pen.speed(0)
    pen.penup()
    pen.pensize(PEN_SIZE)

    return screen, pen


# ============================================================
#                    LIVE DRAWING ENGINE
# ============================================================


def animate(screen, pen, paths, width, height):

    drawing_paths = []

    for path in paths:

        points = [
            map_point(px, py, width, height)
            for px, py in path["points"]
        ]

        if len(points) < 2:
            continue

        drawing_paths.append({
            "points": points,
            "color": path["color"],
            "fill": path["fill"],
            "closed": path["closed"],
            "width": path["width"]
        })

    total_points = sum(
        len(path["points"])
        for path in drawing_paths
    )

    state = {
        "path": 0,
        "point": 0,
        "drawn": 0,
        "filling": False
    }

    def draw_step():

        steps = 0

        while (
            state["path"] < len(drawing_paths)
            and steps < POINTS_PER_FRAME
        ):

            current = drawing_paths[state["path"]]
            points = current["points"]

            index = state["point"]

            # Start a new path
            if index == 0:

                pen.penup()
                pen.color(current["color"])
                pen.fillcolor(current["color"])
                pen.pensize(current["width"])

                pen.goto(points[0])
                pen.pendown()

                if current["fill"]:

                    pen.begin_fill()
                    state["filling"] = True

                state["point"] = 1
                continue

            # Finish current path
            if index >= len(points):

                if current["closed"]:
                    pen.goto(points[0])

                if state["filling"]:
                    pen.end_fill()
                    state["filling"] = False

                pen.penup()

                state["path"] += 1
                state["point"] = 0

                continue

            # Draw next point
            pen.goto(points[index])

            state["point"] += 1
            state["drawn"] += 1
            steps += 1

        screen.update()

        percentage = (
            state["drawn"] / total_points * 100
            if total_points else 100
        )

        screen.title(
            f"Image to Turtle Art 5.0 | "
            f"{percentage:.1f}% complete"
        )

        if state["path"] < len(drawing_paths):

            screen.ontimer(
                draw_step,
                FRAME_DELAY
            )

        else:

            pen.penup()
            screen.update()

            screen.title(
                "Image to Turtle Art 5.0 | Complete"
            )

            print("\nDrawing completed!")
            print(f"Points drawn: {state['drawn']}")

    draw_step()


# ============================================================
#                           MAIN
# ============================================================


def main():

    print("=" * 60)
    print("          IMAGE TO TURTLE ART GENERATOR 5.0")
    print("               MANGA ILLUSTRATION ENGINE")
    print("=" * 60)

    image_path = select_image()

    if not image_path:
        print("No image selected.")
        return

    print(
        f"\nSelected: {os.path.basename(image_path)}"
    )

    img = load_image(image_path)
    img = resize_image(img)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    all_paths = []

    # PASS 1: BLACK INK
    if ENABLE_BLACK_FILLS:

        print("\nExtracting black ink regions...")

        ink_paths = get_ink_paths(gray)
        all_paths.extend(ink_paths)

        print(f"Ink paths: {len(ink_paths)}")

    # PASS 2: MAIN OUTLINES
    if ENABLE_MAIN_OUTLINES:

        print("\nExtracting main outlines...")

        outline_paths = get_main_outline_paths(gray)
        all_paths.extend(outline_paths)

        print(f"Outline paths: {len(outline_paths)}")

    # PASS 3: FINE DETAILS
    if ENABLE_FINE_DETAILS:

        print("\nExtracting fine details...")

        detail_paths = get_detail_paths(gray)
        all_paths.extend(detail_paths)

        print(f"Detail paths: {len(detail_paths)}")

    # PASS 4: CROSS-HATCHING
    if ENABLE_HATCHING:

        print("\nGenerating cross-hatching...")

        hatch_paths = get_hatch_paths(gray)
        all_paths.extend(hatch_paths)

        print(f"Hatching paths: {len(hatch_paths)}")

    if not all_paths:
        raise ValueError("No drawable paths found.")

    # Draw layers in order
    all_paths.sort(
        key=lambda p: p["layer"]
    )

    print("\nPreparing Turtle animation...")

    screen, pen = create_turtle()

    print("Live drawing started!")

    animate(
        screen,
        pen,
        all_paths,
        width,
        height
    )

    turtle.done()


# ============================================================
#                         EXECUTION
# ============================================================


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(f"\nERROR: {error}")

        try:
            input("\nPress Enter to exit...")

        except EOFError:
            pass
