from __future__ import print_function, division, absolute_import
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from reportlab.graphics.shapes import Path
from reportlab.lib import colors
from reportlab.graphics import renderPM
from reportlab.graphics.shapes import Group, Drawing, scale


class ReportLabPen(BasePen):
    """A pen for drawing onto a reportlab.graphics.shapes.Path object."""

    def __init__(self, glyphSet, path=None):
        BasePen.__init__(self, glyphSet)
        if path is None:
            path = Path()
        self.path = path

    def _moveTo(self, p):
        (x, y) = p
        self.path.moveTo(x, y)

    def _lineTo(self, p):
        (x, y) = p
        self.path.lineTo(x, y)

    def _curveToOne(self, p1, p2, p3):
        (x1, y1) = p1
        (x2, y2) = p2
        (x3, y3) = p3
        self.path.curveTo(x1, y1, x2, y2, x3, y3)

    def _closePath(self):
        self.path.closePath()


def is_Chinese(word):
    for ch in word:
        if '\u4e00' > ch or word > '\u9fff':
            return False
    return True

def ttfToImage(fontName, imagePath, fmt="png"):
    font = TTFont(fontName)
    gs = font.getGlyphSet()
    glyphNames = font.getGlyphNames()

    m_dict = font.getBestCmap()

    unicode_list = []
    gs_key=[]
    for key, value in m_dict.items():
        unicode_list.append(key)
        gs_key.append(value)
    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    #2500-29000 FZSONG_ZhongHuaSongPlane00_2021120120211201171438.ttf
    # 275 FZSONG_ZhongHuaSongPlane02_2021120120211201171459.ttf

    for i in range(len(char_list)):
        if fontName=='FZSONG_ZhongHuaSongPlane00_2021120120211201171438.ttf' and (i<2500 or i>29000):
            continue
        if fontName=='FZSONG_ZhongHuaSongPlane02_2021120120211201171459.ttf' and (i<275 or i>100000000):
            continue
        thechr=char_list[i]
        print(thechr)
        gs_ind=gs_key[i]
        if gs_ind[0] == '.':  # 跳过'.notdef', '.null'
            continue
        g = gs[gs_ind]
        pen = ReportLabPen(gs, Path(fillColor=colors.black, strokeWidth=5))
        g.draw(pen)
        w, h = g.width, g.width
        # w = 4048
        # h = 4048
        g = Group(pen.path)
        g.translate(0, 170)

        d = Drawing(w, h)
        d.add(g)
        imageFile = imagePath + "/" + thechr + ".png"
        renderPM.drawToFile(d, imageFile, fmt)


#         break;


ttfToImage(fontName="FZSONG_ZhongHuaSongPlane00_2021120120211201171438.ttf", imagePath="font")
# ttfToImage(fontName="FZSONG_ZhongHuaSongPlane02_2021120120211201171459.ttf", imagePath="font")