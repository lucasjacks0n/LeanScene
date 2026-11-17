"""Build the LeanScene logo files in this folder.

    python logo/build.py        (needs fonttools; macOS, for the Helvetica Neue system font)

Writes leanscene-wordmark.svg (dark lettering, for light backgrounds), leanscene-wordmark-dark.svg
(white lettering, for dark backgrounds) and leanscene-icon.svg (square, for a favicon or avatar).
The lettering is Helvetica Neue (Light or UltraLight, and Bold) converted to outlines, so the files do not depend
on any font being installed. The backgrounds of the wordmarks are transparent.
"""
import os
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTCollection

HERE = os.path.dirname(os.path.abspath(__file__))
INK, WHITE, CUT = "#111111", "#ffffff", "#d9480f"  # CUT is the demo timeline's cut colour
FACES = {f["name"].getDebugName(4): f for f in TTCollection("/System/Library/Fonts/HelveticaNeue.ttc").fonts}
LIGHT, BOLD, ULTRALIGHT = FACES["Helvetica Neue Light"], FACES["Helvetica Neue Bold"], FACES["Helvetica Neue UltraLight"]


def outline(font, text, size, x, baseline, spacing=0.0, anchor="start"):
    """SVG path data for `text`, and its (x0, y0, x1, y1) bounds."""
    glyphs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    k = size / font["head"].unitsPerEm
    names = [cmap[ord(c)] for c in text]
    width = sum(hmtx[n][0] * k + spacing for n in names)
    pen_x = x - width if anchor == "end" else x
    path, box = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip(".")), BoundsPen(glyphs)
    for n in names:
        t = (k, 0, 0, -k, pen_x, baseline)  # font units, y up -> SVG px, y down
        glyphs[n].draw(TransformPen(path, t))
        glyphs[n].draw(TransformPen(box, t))
        pen_x += hmtx[n][0] * k + spacing
    return path.getCommands(), box.bounds


def marker(x, top, half, line_top, line_bottom, width):
    return (f'<path d="M{x-half} {top}h{2*half}l-{half} {half*1.6:.1f}z" fill="{CUT}"/>'
            f'<path d="M{x} {line_top}V{line_bottom}" stroke="{CUT}" stroke-width="{width}"/>')


def wordmark(colour, thin=LIGHT):
    join, gap, size, base, pad = 130, 6, 46, 77, 10
    lean, b1 = outline(thin, "LEAN", size, join - gap, base, 1, "end")
    scene, b2 = outline(BOLD, "SCENE", size, join + gap, base, -1)
    x0, x1, y0, y1 = b1[0] - pad, b2[2] + pad, 16 - pad, 88 + pad
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {y0} {x1-x0:.1f} {y1-y0}" width="{(x1-x0)*2:.0f}" height="{(y1-y0)*2}" role="img" aria-label="LeanScene">\n'
            f'<title>LeanScene</title>\n<path d="{lean}{scene}" fill="{colour}"/>\n{marker(join, 16, 8, 34, 88, 2)}\n</svg>\n')


def icon():
    l, _ = outline(LIGHT, "L", 76, 52, 92, anchor="end")
    s, _ = outline(BOLD, "S", 76, 66, 92)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120" role="img" aria-label="LeanScene">\n'
            f'<title>LeanScene</title>\n<rect width="120" height="120" rx="22" fill="{INK}"/>\n<path d="{l}{s}" fill="{WHITE}"/>\n{marker(59, 8, 9, 26, 104, 3)}\n</svg>\n')


# The dark wordmark has a thinner LEAN: light lettering on dark reads heavier than dark on light.
for name, text in (("leanscene-wordmark.svg", wordmark(INK)), ("leanscene-wordmark-dark.svg", wordmark(WHITE, ULTRALIGHT)), ("leanscene-icon.svg", icon())):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(text)
    print("wrote logo/" + name, f"({len(text) / 1000:.1f} KB)")
