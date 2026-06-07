import os
from datetime import datetime
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from ultralytics import YOLO

# Путь к папке в которой находятся данные
# Path to the folder where the data is located
IMAGE_DIR = r"C:\Users\path\in\image\Train"

# Имя аннотации
# Аnnotation name
OUTPUT_XML = "cvat_perspective_cuboids.xml"

# Имя модели
# Models name
MODEL_NAME = "yolov8x.pt"

CONF_THRESHOLD = 0.35
CUBOID_ROOF_RATIO = 0.12

# Пропуск обьектов на границах изображение (если хотите оставить поменяй те на False)
# Skipping objects on the borders of the image (if you want to leave them, change them to False)
SKIP_EDGE_TRUNCATED = True
EDGE_MARGIN = 5

# Классы для YOLO
# Yolo classes
TARGET_CLASSES = {
    2: "Car",
    5: "Bus",
    7: "Truck",
}

LABELS = [
    ("Car", "#4727e6"),
    ("Bus", "#6349ea"),
    ("Truck", "#a295e6"),
]

# Установка YOLO
# Download YOLO
model = YOLO(MODEL_NAME)


# Итоговая аннотация будет в формате CVAT 1.1
# The final annotation will be in CVAt 1.1 format
annotations = ET.Element("annotations")

version = ET.SubElement(annotations, "version")
version.text = "1.1"

meta = ET.SubElement(annotations, "meta")
job = ET.SubElement(meta, "job")

ET.SubElement(job, "id").text = "1"
size_el = ET.SubElement(job, "size")
size_el.text = "0"
ET.SubElement(job, "mode").text = "annotation"
ET.SubElement(job, "overlap").text = "0"
ET.SubElement(job, "created").text = datetime.now().isoformat()
ET.SubElement(job, "updated").text = datetime.now().isoformat()
ET.SubElement(job, "subset").text = "Train"
ET.SubElement(job, "start_frame").text = "0"
stop_frame_el = ET.SubElement(job, "stop_frame")
stop_frame_el.text = "0"

labels = ET.SubElement(job, "labels")

for name, color in LABELS:
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = name
    ET.SubElement(label, "color").text = color
    ET.SubElement(label, "type").text = "cuboid"
    ET.SubElement(label, "attributes")

# Постройка cuboid-а по bbox-у
# Make cuboid from bbox
def clamp_point(pt, img_w, img_h):
    return (
        int(np.clip(pt[0], 0, img_w - 1)),
        int(np.clip(pt[1], 0, img_h - 1)),
    )


def build_perspective_cuboid(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1
    h = y2 - y1

    if w <= 1 or h <= 1:
        return None

    # Ищем центр кадра для постройки cuboid-а по правильной перспективе
    # We are looking for the center of the frame to build a cuboid from the right perspective
    vp_x = img_w * 0.52
    vp_y = img_h * 0.35

    obj_x = (x1 + x2) / 2
    obj_y = y2

    direction = np.array([vp_x - obj_x, vp_y - obj_y], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1:
        norm = 1
    direction /= norm

    # Убераем параллелепипеды у границ изображения
    # Removing the parallelepipeds at the borders of the image
    scale = y2 / img_h
    depth = w * (0.18 + 0.12 * scale)
    depth = float(np.clip(depth, 8, min(w * 0.32, h * 0.22, 70)))
    offset = direction * depth

    # Увеличиваем высоту cuboid-ов
    # Increasing the height of the cuboids
    roof_y = y1 + h * CUBOID_ROOF_RATIO

    f_tl = np.array([x1, roof_y], dtype=np.float32)
    f_tr = np.array([x2, roof_y], dtype=np.float32)
    f_bl = np.array([x1, y2], dtype=np.float32)
    f_br = np.array([x2, y2], dtype=np.float32)

    b_tl = f_tl + offset
    b_tr = f_tr + offset
    b_bl = f_bl + offset
    b_br = f_br + offset

    shrink = w * 0.12
    b_tl[0] += shrink / 2
    b_bl[0] += shrink / 2
    b_tr[0] -= shrink / 2
    b_br[0] -= shrink / 2

    f_tl = clamp_point(f_tl, img_w, img_h)
    f_tr = clamp_point(f_tr, img_w, img_h)
    f_bl = clamp_point(f_bl, img_w, img_h)
    f_br = clamp_point(f_br, img_w, img_h)
    b_tl = clamp_point(b_tl, img_w, img_h)
    b_tr = clamp_point(b_tr, img_w, img_h)
    b_bl = clamp_point(b_bl, img_w, img_h)
    b_br = clamp_point(b_br, img_w, img_h)

    '''
    =====================================!!!ВАЖНО!!!=====================================
    ===================================!!!IMPORTANT!!!===================================
    
    Если поочередность переменных поменять местами то cuboid не будет построен правильно!
    If the alternation of variables is reversed, the cuboid will not be built correctly!
    
    ===================================!!!IMPORTANT!!!===================================
    =====================================!!!ВАЖНО!!!=====================================
    '''
    return {
        # Передняя часть cuboid-a
        # Face cuboid
        "xtl1": f_tl[0],
        "ytl1": f_tl[1],
        "xbl1": f_bl[0],
        "ybl1": f_bl[1],
        "xtr1": f_tr[0],
        "ytr1": f_tr[1],
        "xbr1": f_br[0],
        "ybr1": f_br[1],

        # Задняя часть cuboid-а
        # Back cuboid
        "xtl2": b_tr[0],
        "ytl2": b_tr[1],
        "xbl2": b_br[0],
        "ybl2": b_br[1],
        "xtr2": b_tl[0],
        "ytr2": b_tl[1],
        "xbr2": b_bl[0],
        "ybr2": b_bl[1],
    }

def is_edge_truncated(x1, x2, img_w):
    return x1 <= EDGE_MARGIN or x2 >= img_w - EDGE_MARGIN

# Запуск
# Start
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

size_el.text = str(len(image_files))
stop_frame_el.text = str(max(len(image_files) - 1, 0))

for idx, image_name in enumerate(image_files):
    image_path = os.path.join(IMAGE_DIR, image_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"[skip] cannot read image: {image_path}")
        continue

    img_h, img_w = img.shape[:2]

    image_tag = ET.SubElement(
        annotations,
        "image",
        {
            "id": str(idx),
            "name": image_name,
            "width": str(img_w),
            "height": str(img_h),
        },
    )

    results = model(img, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])

            if cls_id not in TARGET_CLASSES:
                continue

            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            label_name = TARGET_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if SKIP_EDGE_TRUNCATED and is_edge_truncated(x1, x2, img_w):
                continue

            cuboid = build_perspective_cuboid(x1, y1, x2, y2, img_w, img_h)
            if cuboid is None:
                continue

            attrs = {
                "label": label_name,
                "source": "auto",
                "occluded": "0",
                "z_order": "0",
            }
            attrs.update({k: str(v) for k, v in cuboid.items()})

            ET.SubElement(image_tag, "cuboid", attrs)

    print(f"[{idx + 1}/{len(image_files)}] {image_name}")

# Сохраняем аннотацию
# Save annotation
tree = ET.ElementTree(annotations)
ET.indent(tree, space="  ")
tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)

print("\nDONE")
print("XML:", OUTPUT_XML)
