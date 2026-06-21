"""Generate thesis_diagrams.drawio — valid XML with unique cell IDs."""
import html
import os
import uuid
import xml.etree.ElementTree as ET

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_diagrams.drawio")

S = {
    "title":   "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=18;fontStyle=1;fontFamily=Arial;whiteSpace=wrap;",
    "start":   "ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=12;fontFamily=Arial;",
    "end":     "ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=12;fontFamily=Arial;",
    "process": "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;fontFamily=Arial;",
    "data":    "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=12;fontFamily=Arial;",
    "decision":"rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;fontFamily=Arial;",
    "model_r": "rounded=1;whiteSpace=wrap;html=1;fillColor=#2E86AB;fontColor=#ffffff;strokeColor=#1a5276;fontSize=12;fontFamily=Arial;",
    "model_c": "rounded=1;whiteSpace=wrap;html=1;fillColor=#3BB273;fontColor=#ffffff;strokeColor=#1e8449;fontSize=12;fontFamily=Arial;",
    "layer":   "rounded=0;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;fontFamily=Arial;",
    "layer_in":"rounded=0;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;fontFamily=Arial;",
    "layer_out":"rounded=0;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;fontFamily=Arial;",
    "arrow":   "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;fontSize=10;fontFamily=Arial;",
    "arrow_d": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;dashed=1;fontSize=10;fontFamily=Arial;",
    "group":   "swimlane;startSize=30;fillColor=#f5f5f5;strokeColor=#666666;fontSize=13;fontStyle=1;fontFamily=Arial;whiteSpace=wrap;",
    "note":    "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;fillColor=#ffffcc;strokeColor=#999999;fontSize=10;fontFamily=Arial;size=14;",
}


class DiagramBuilder:
    def __init__(self, prefix, page_w=1200, page_h=800):
        self.prefix = prefix
        self.page_w = page_w
        self.page_h = page_h
        self.cells = []
        self._n = 0

    def _id(self, tag):
        return f"{self.prefix}_{tag}"

    def _next_edge_id(self):
        self._n += 1
        return self._id(f"edge{self._n}")

    def add_vertex(self, tag, value, style, x, y, w, h):
        self.cells.append({
            "id": self._id(tag),
            "value": value,
            "style": style,
            "vertex": "1",
            "edge": None,
            "source": None,
            "target": None,
            "x": x, "y": y, "w": w, "h": h,
        })
        return self._id(tag)

    def add_edge(self, src, tgt, label="", dashed=False):
        self.cells.append({
            "id": self._next_edge_id(),
            "value": label,
            "style": S["arrow_d"] if dashed else S["arrow"],
            "vertex": "0",
            "edge": "1",
            "source": src,
            "target": tgt,
            "x": 0, "y": 0, "w": 0, "h": 0,
        })

    def to_root_xml(self):
        lines = [
            '        <mxCell id="0"/>',
            '        <mxCell id="1" parent="0"/>',
        ]
        for c in self.cells:
            val = html.escape(c["value"], quote=True).replace("\n", "&#xa;")
            attrs = (
                f'id="{c["id"]}" value="{val}" style="{c["style"]}" '
                f'vertex="{c["vertex"]}" parent="1"'
            )
            if c["edge"]:
                attrs += f' edge="{c["edge"]}" source="{c["source"]}" target="{c["target"]}"'
            lines.append(f"        <mxCell {attrs}>")
            if c["edge"]:
                lines.append('          <mxGeometry relative="1" as="geometry"/>')
            else:
                lines.append(
                    f'          <mxGeometry x="{c["x"]}" y="{c["y"]}" '
                    f'width="{c["w"]}" height="{c["h"]}" as="geometry"/>'
                )
            lines.append("        </mxCell>")
        return "\n".join(lines)


def build_main_flow():
    d = DiagramBuilder("main", 1100, 1350)
    d.add_vertex("title", "المخطط الرئيسي — خط أنابيب التدريب (EuroSAT)", S["title"], 250, 20, 600, 40)
    d.add_vertex("start", "البداية", S["start"], 470, 80, 160, 50)
    d.add_vertex("ds", "اختيار نوع البيانات\nRGB (3 قنوات) | Multispectral (13 قناة)", S["decision"], 420, 160, 260, 90)
    d.add_vertex("csv", "تحميل ملفات CSV\n(train / validation / test)", S["data"], 420, 280, 260, 60)
    d.add_vertex("gen", "EuroSATGenerator\nقراءة الصور + تغيير الحجم 64×64", S["process"], 420, 370, 260, 60)
    d.add_vertex("prep", "معالجة مسبقة\nRGB: ÷255 | MS: تطبيع (mean/std)", S["process"], 420, 460, 260, 60)
    d.add_vertex("aug", "Augmentation\n(قلب أفقي/عمودي — التدريب فقط)", S["process"], 420, 550, 260, 60)
    d.add_vertex("build", "بناء نموذجين للمقارنة", S["process"], 420, 640, 260, 50)
    d.add_vertex("resnet", "ResNet50V2\n(ImageNet — Transfer Learning)", S["model_r"], 180, 730, 230, 60)
    d.add_vertex("custom", "Custom CNN\n(من الصفر)", S["model_c"], 690, 730, 230, 60)
    d.add_vertex("compile", "Compile\nAdam + Sparse Categorical Crossentropy", S["process"], 420, 820, 260, 50)
    d.add_vertex("train", "التدريب + Callbacks\nCheckpoint | EarlyStopping | ReduceLR", S["process"], 420, 900, 260, 60)
    d.add_vertex("test", "تقييم على Test Set", S["process"], 420, 990, 260, 50)
    d.add_vertex("save", "حفظ المخرجات\nbest_model_*.h5 | history_*.json | curves_*.png", S["data"], 420, 1070, 260, 60)
    d.add_vertex("cmp", "مقارنة النماذج\ncomparison_accuracy / loss / bar", S["process"], 420, 1160, 260, 60)
    d.add_vertex("end", "النهاية — طباعة النتائج النهائية", S["end"], 440, 1250, 220, 50)
    d.add_vertex("note", "10 فئات:\nAnnualCrop | Forest | HerbaceousVegetation\nHighway | Industrial | Pasture\nPermanentCrop | Residential | River | SeaLake", S["note"], 780, 280, 280, 100)

    chain = ["start","ds","csv","gen","prep","aug","build","compile","train","test","save","cmp","end"]
    for i in range(len(chain) - 1):
        d.add_edge(d._id(chain[i]), d._id(chain[i+1]))
    d.add_edge(d._id("build"), d._id("resnet"), "نموذج 1")
    d.add_edge(d._id("build"), d._id("custom"), "نموذج 2")
    d.add_edge(d._id("resnet"), d._id("compile"), dashed=True)
    d.add_edge(d._id("custom"), d._id("compile"), dashed=True)
    return "1 - المخطط الرئيسي", d


def build_custom_cnn():
    d = DiagramBuilder("cnn", 1100, 1650)
    d.add_vertex("title", "بنية Custom CNN — الطبقات", S["title"], 300, 20, 500, 40)
    layers = [
        ("in",   "Input\n64 × 64 × C\n(C=3 RGB | C=13 Multispectral)", S["layer_in"],  430, 80),
        ("c1",   "Conv2D — 32 filters\nkernel=3×3, padding=same",      S["layer"],     430, 180),
        ("bn1",  "BatchNormalization",                                   S["layer"],     430, 255),
        ("r1",   "ReLU",                                                 S["layer"],     430, 320),
        ("p1",   "MaxPooling2D\n(32×32×32)",                             S["layer"],     430, 380),
        ("c2",   "Conv2D — 64 filters\nkernel=3×3, padding=same",      S["layer"],     430, 460),
        ("bn2",  "BatchNormalization",                                   S["layer"],     430, 535),
        ("r2",   "ReLU",                                                 S["layer"],     430, 600),
        ("p2",   "MaxPooling2D\n(16×16×64)",                             S["layer"],     430, 660),
        ("c3",   "Conv2D — 128 filters\nkernel=3×3, padding=same",     S["layer"],     430, 740),
        ("bn3",  "BatchNormalization",                                   S["layer"],     430, 815),
        ("r3",   "ReLU",                                                 S["layer"],     430, 880),
        ("p3",   "MaxPooling2D\n(8×8×128)",                              S["layer"],     430, 940),
        ("c4",   "Conv2D — 256 filters\nkernel=3×3, padding=same",     S["layer"],     430, 1020),
        ("bn4",  "BatchNormalization",                                   S["layer"],     430, 1095),
        ("r4",   "ReLU",                                                 S["layer"],     430, 1160),
        ("gap",  "GlobalAveragePooling2D\n(256 features)",             S["layer"],     430, 1220),
        ("d1",   "Dense — 256 units\nactivation=relu",                 S["layer"],     430, 1300),
        ("drop", "Dropout — 0.4",                                        S["layer"],     430, 1370),
        ("d2",   "Dense — 10 units\nactivation=softmax",               S["layer_out"], 430, 1440),
        ("out",  "Output\nاحتمالية لكل فئة (10)",                        S["layer_out"], 430, 1520),
    ]
    ids = []
    for tag, val, sty, x, y in layers:
        h = 70 if tag == "in" else (55 if tag.startswith("c") or tag == "d2" else (50 if tag in ("p1","p2","p3","gap","d1","out") else 45))
        ids.append(d.add_vertex(tag, val, sty, x, y, 240, h))
    for i in range(len(ids) - 1):
        d.add_edge(ids[i], ids[i+1])

    d.add_vertex("blk1", "Block 1\nFeature Extraction", S["group"], 150, 180, 160, 250)
    d.add_vertex("blk2", "Block 2", S["group"], 150, 460, 160, 250)
    d.add_vertex("blk3", "Block 3", S["group"], 150, 740, 160, 250)
    d.add_vertex("blk4", "Block 4\n+ Classifier", S["group"], 150, 1020, 160, 550)
    d.add_vertex("note", "إجمالي المعاملات: أقل بكثير من ResNet50V2\nمناسب للمقارنة مع Transfer Learning", S["note"], 780, 700, 260, 70)
    return "2 - طبقات Custom CNN", d


def build_resnet():
    d = DiagramBuilder("resnet", 1100, 750)
    d.add_vertex("title", "بنية ResNet50V2 — Transfer Learning", S["title"], 300, 20, 500, 40)
    layers = [
        ("in",  "Input\n64 × 64 × C", S["layer_in"], 430, 80, 240, 55),
        ("bb",  "ResNet50V2 Backbone\n(weights=ImageNet)\ntrainable = False", S["model_r"], 430, 170, 240, 70),
        ("gap", "GlobalAveragePooling2D", S["layer"], 430, 270, 240, 45),
        ("d1",  "Dropout — 0.4", S["layer"], 430, 340, 240, 45),
        ("d2",  "Dense — 512 units\nactivation=relu", S["layer"], 430, 410, 240, 50),
        ("d3",  "Dropout — 0.3", S["layer"], 430, 480, 240, 45),
        ("d4",  "Dense — 10 units\nactivation=softmax", S["layer_out"], 430, 550, 240, 55),
        ("out", "Output\n10 فئات EuroSAT", S["layer_out"], 430, 630, 240, 50),
    ]
    ids = []
    for tag, val, sty, x, y, w, h in layers:
        ids.append(d.add_vertex(tag, val, sty, x, y, w, h))
    for i in range(len(ids) - 1):
        d.add_edge(ids[i], ids[i+1])
    d.add_vertex("n1", "الطبقات المجمّدة: لا تُدرّب\nيُدرّب فقط رأس التصنيف (Classifier Head)", S["note"], 780, 200, 280, 70)
    d.add_vertex("n2", "ميزة: استفادة من ميزات\nمدرّبة مسبقاً على ImageNet", S["note"], 780, 400, 280, 60)
    return "3 - بنية ResNet50V2", d


def build_data_flow():
    d = DiagramBuilder("data", 1100, 800)
    d.add_vertex("title", "مخطط معالجة البيانات — EuroSATGenerator", S["title"], 250, 20, 550, 40)
    d.add_vertex("csv", "CSV File\n(Filename + Label)", S["data"], 430, 80, 240, 50)
    d.add_vertex("type", "نوع الصورة؟", S["decision"], 450, 160, 200, 80)
    d.add_vertex("rgb", "RGB (JPEG/PNG)\nload_img → 64×64\n÷ 255.0", S["process"], 170, 280, 220, 70)
    d.add_vertex("ms", "Multispectral (TIFF)\nrasterio.read → 13 bands\n(mean-std normalization)", S["process"], 690, 280, 260, 70)
    d.add_vertex("augq", "Augmentation?\n(التدريب فقط)", S["decision"], 450, 390, 200, 80)
    d.add_vertex("aug", "Flip Horizontal / Vertical", S["process"], 430, 510, 240, 45)
    d.add_vertex("batch", "Batch\n(images, labels)", S["data"], 430, 590, 240, 50)
    d.add_vertex("send", "إرسال للنموذج", S["end"], 450, 670, 200, 50)

    d.add_edge(d._id("csv"), d._id("type"))
    d.add_edge(d._id("type"), d._id("rgb"), "RGB")
    d.add_edge(d._id("type"), d._id("ms"), "MS")
    d.add_edge(d._id("rgb"), d._id("augq"))
    d.add_edge(d._id("ms"), d._id("augq"))
    d.add_edge(d._id("augq"), d._id("aug"), "نعم")
    d.add_edge(d._id("augq"), d._id("batch"), "لا")
    d.add_edge(d._id("aug"), d._id("batch"))
    d.add_edge(d._id("batch"), d._id("send"))
    return "4 - معالجة البيانات", d


def build_comparison():
    d = DiagramBuilder("cmp", 1100, 650)
    d.add_vertex("title", "مخطط المقارنة والمخرجات", S["title"], 330, 20, 450, 40)
    d.add_vertex("r", "ResNet50V2\nbest_model_resnet.h5", S["model_r"], 150, 100, 220, 60)
    d.add_vertex("c", "Custom CNN\nbest_model_custom.h5", S["model_c"], 530, 100, 220, 60)
    d.add_vertex("eval", "Test Set Evaluation", S["process"], 330, 210, 240, 50)
    d.add_vertex("cr", "curves_resnet.png\ncurves_custom.png", S["data"], 150, 310, 220, 55)
    d.add_vertex("hc", "history_resnet.json\nhistory_custom.json", S["data"], 530, 310, 220, 55)
    d.add_vertex("plots", "comparison_accuracy.png\ncomparison_loss.png\ncomparison_bar.png", S["process"], 290, 420, 320, 70)
    d.add_vertex("gui", "اختيار أفضل نموذج\nللاستخدام في GUI", S["end"], 330, 530, 240, 55)

    d.add_edge(d._id("r"), d._id("eval"))
    d.add_edge(d._id("c"), d._id("eval"))
    d.add_edge(d._id("r"), d._id("cr"), dashed=True)
    d.add_edge(d._id("c"), d._id("hc"), dashed=True)
    d.add_edge(d._id("eval"), d._id("plots"))
    d.add_edge(d._id("cr"), d._id("plots"), dashed=True)
    d.add_edge(d._id("hc"), d._id("plots"), dashed=True)
    d.add_edge(d._id("plots"), d._id("gui"))
    return "5 - المقارنة والمخرجات", d


def make_page(name, builder):
    return f"""  <diagram id="{uuid.uuid4().hex[:8]}" name="{html.escape(name, quote=True)}">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{builder.page_w}" pageHeight="{builder.page_h}" math="0" shadow="0">
      <root>
{builder.to_root_xml()}
      </root>
    </mxGraphModel>
  </diagram>"""


def main():
    pages = [build_main_flow(), build_custom_cnn(), build_resnet(), build_data_flow(), build_comparison()]
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" agent="EuroSAT-Thesis" version="24.0.0" '
        f'type="device" pages="{len(pages)}">\n'
        + "\n".join(make_page(name, b) for name, b in pages)
        + "\n</mxfile>\n"
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(content)

    # Validate XML
    ET.parse(OUT)
    print(f"Created & validated: {OUT}")


if __name__ == "__main__":
    main()
