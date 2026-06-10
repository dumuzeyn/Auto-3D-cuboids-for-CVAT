import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

XML_PATH = r"C:\Users\path\in\annotations.xml"
OUTPUT_DIR = "annotations"

CATEGORY_MAPPING = {
    "Car": 1,
    "Bus": 2,
    "Truck": 3
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
tree = ET.parse(XML_PATH)
root = tree.getroot()

datasets = {
    "Train": {
        "images": [],
        "annotations": [],
        "categories": []
    },
    "Validation": {
        "images": [],
        "annotations": [],
        "categories": []
    },
    "Test": {
        "images": [],
        "annotations": [],
        "categories": []
    }
}

# categories
for name, cid in CATEGORY_MAPPING.items():
    category = {
        "id": cid,
        "name": name
    }

    for subset in datasets:
        datasets[subset]["categories"].append(category)

annotation_id = 1
global_image_id = 0

for image in root.findall("image"):

    image_id = global_image_id
    global_image_id += 1

    subset = image.attrib["subset"]

    file_name = image.attrib["name"]

    width = int(image.attrib["width"])
    height = int(image.attrib["height"])

    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

    datasets[subset]["images"].append(image_info)
    for cuboid in image.findall("cuboid"):

        label = cuboid.attrib["label"]

        if label not in CATEGORY_MAPPING:
            continue

        category_id = CATEGORY_MAPPING[label]

        # 8 points
        points = [
            [float(cuboid.attrib["xtl1"]), float(cuboid.attrib["ytl1"])],
            [float(cuboid.attrib["xbl1"]), float(cuboid.attrib["ybl1"])],
            [float(cuboid.attrib["xtr1"]), float(cuboid.attrib["ytr1"])],
            [float(cuboid.attrib["xbr1"]), float(cuboid.attrib["ybr1"])],

            [float(cuboid.attrib["xtl2"]), float(cuboid.attrib["ytl2"])],
            [float(cuboid.attrib["xbl2"]), float(cuboid.attrib["ybl2"])],
            [float(cuboid.attrib["xtr2"]), float(cuboid.attrib["ytr2"])],
            [float(cuboid.attrib["xbr2"]), float(cuboid.attrib["ybr2"])]
        ]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min = min(xs)
        y_min = min(ys)

        x_max = max(xs)
        y_max = max(ys)

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,

            "bbox": [
                x_min,
                y_min,
                bbox_width,
                bbox_height
            ],

            "area": bbox_width * bbox_height,

            "iscrowd": 0,

            "segmentation": [],

            "cuboid": points
        }

        datasets[subset]["annotations"].append(annotation)

        annotation_id += 1

output_mapping = {
    "Train": "train.json",
    "Validation": "val.json",
    "Test": "test.json"
}

for subset, filename in output_mapping.items():

    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w") as f:
        json.dump(datasets[subset], f, indent=4)

    print(f"Saved: {output_path}")

print("DONE")

