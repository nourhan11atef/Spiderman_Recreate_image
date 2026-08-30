# Turtle Image Sketcher

Turns any image into a hand-drawn-style animation using Python's `turtle` module. The script extracts the silhouette/outline of an image with OpenCV, then "draws" it live on screen stroke by stroke — filling in each shape with solid black as it goes.

![demo placeholder](demo.gif)
<!-- Replace with an actual screen recording or GIF of the drawing in action -->

## How it works

1. **Load & resize** — Loads the input image and resizes it to a fixed height (700px) while preserving aspect ratio.
2. **Threshold** — Converts to grayscale and applies binary thresholding to separate dark regions (the subject) from the light background.
3. **Find contours** — Uses OpenCV's `findContours` to trace the outline of every shape, filters out tiny noise specks, and sorts them largest-first so the main silhouette draws before fine details.
4. **Animate** — A `turtle` pen walks along each contour's points, periodically refreshing the screen (`screen.update()`) to create a smooth "hand-drawn" animation, filling each shape with black as it's traced.

## Requirements

```bash
pip install opencv-python
```

`turtle` ships with the Python standard library, so no extra install is needed for it.

## Usage

1. Place an image in the project folder.
2. Update the `IMAGE` variable at the top of the script with your filename:
   ```python
   IMAGE = "spiderman.png"
   ```
3. Run the script:
   ```bash
   python main.py
   ```

A turtle graphics window will open and animate the drawing. Close the window or press any key to exit once `turtle.done()` is reached (depending on your Python/turtle setup).

## Customization

| Variable | Effect |
|---|---|
| `height` | Target image height in pixels (width scales to match) |
| `180` (in `cv2.threshold`) | Threshold cutoff — lower catches more detail, higher keeps only the darkest areas |
| `cv2.contourArea(c) > 15` | Minimum contour size kept — raise this to ignore more small artifacts |
| `scale` | Scales the drawing size on the turtle canvas |
| `UPDATE_EVERY` | Points drawn between each screen refresh — lower = smoother/slower animation, higher = faster/jumpier |
| `screen.setup(width=..., height=...)` | Turtle window size |

## Notes

- Works best with high-contrast images (clear subject against a light background) since it relies on simple brightness thresholding rather than more advanced edge detection.
- Contours from OpenCV aren't always perfectly ordered closed loops, so fills on complex/self-intersecting shapes may occasionally look slightly imperfect — this is a limitation of pixel-contour tracing, not a bug.

## License

MIT (or your license of choice)
