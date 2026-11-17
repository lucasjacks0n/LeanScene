# Logo

| File | Use |
|---|---|
| `leanscene-wordmark.svg` | The wordmark with dark lettering, for light backgrounds. Transparent background |
| `leanscene-wordmark-dark.svg` | White lettering for dark backgrounds, with a thinner LEAN (UltraLight). Transparent background |
| `leanscene-icon.svg` | Square icon (favicon, avatar) |
| `leanscene-icon-32.png`, `-180.png`, `-512.png` | The icon as PNG, transparent corners |
| `build.py` | Regenerates the three SVG files |

The thin LEAN and heavy SCENE are separated by a cut marker: the same triangle and colour
(`#d9480f`) the demo's timeline uses for a detected cut.

The lettering is Helvetica Neue (Light, UltraLight on dark, and Bold) converted to outlines, so the files render the
same everywhere. `build.py` reads that font from macOS and needs `fonttools`:

```sh
python logo/build.py
```

The PNGs are the icon rendered by headless Chrome at each size with a transparent background;
`build.py` does not make them.
