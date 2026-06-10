# Автоматическое создание 3D кубоидов для транспортных средств на дорожной сцене в CVAT

**if you are an English speaker click [that](#english-version)**

Данный проект поможет автоматизировать процесс разметки машин, автобусов и грузовиков в CVAT. Проект содержит один скрипт `auto_cvat_cuboids.py`, который автоматически размечает автомобили на изображениях с помощью YOLOv8 и с не большими махинациями создает кубоиды, а далее записывает в XML-разметку в формате CVAT 1.1.А также скрипт `convert_cvat2coco.py` который меняет формат аннотации из `CVAT for images 1.1` в `COCO`.

Его можно импортировать в CVAT как разметку формата `CVAT for images 1.1`.

## Что делает скрипт

Скрипт работает следующим образом:

1. Берёт все изображения из папки `IMAGE_DIR`.

2. Загружает модель YOLOv8.

3. Для каждого изображения запускает детекцию объектов.

4. Оставляет только нужные классы, в моем случае это были `Car`, `Bus`, `Truck`.

5. Отсекает слабые детекции по порогу уверенности.

6. Из каждого 2D bounding box (`bbox`) строится 3D-кубоид (`cuboid`).

7. Записывает все кубоиды в XML-файл форматом CVAT 1.1.

8. Импортировать аннотацию в CVAT.

Идея простая: YOLO даёт обычный прямоугольник вокруг машины, а скрипт достраивает к нему вторую грань, чтобы получился псевдо-3D кубоид.

## Настройки

### Базовые переменные

В начале файла находятся __основные параметры__:

```python
IMAGE_DIR = r"C:\Users\path\in\image\Train"
OUTPUT_XML = "cvat_perspective_cuboids.xml"
MODEL_NAME = "yolov8x.pt"
CONF_THRESHOLD = 0.35
CUBOID_ROOF_RATIO = 0.12
```

`IMAGE_DIR` - папка с изображениями.

`OUTPUT_XML` - имя XML-файла, который будет создан.

`MODEL_NAME` - модель YOLO. Сейчас используется `yolov8x.pt`.

`CONF_THRESHOLD` - минимальная уверенность YOLO. Если значение ниже `0.35`,
объект не попадёт в разметку.

`CUBOID_ROOF_RATIO` - высота верхней грани кубоида внутри YOLO-бокса. Чем
меньше значение, тем выше кубоид:

### Классы YOLO

В проекте скрипт берёт только три класса YOLO:

```python
TARGET_CLASSES = {
    2: "Car",
    5: "Bus",
    7: "Truck",
}
```
```python
LABELS = [
    ("Car", "#4727e6"), #Рядом c именами это цвета, выбранные в CVAT
    ("Bus", "#6349ea"),
    ("Truck", "#a295e6"),
]
```
Если нужны другие классы, их нужно добавить и в `TARGET_CLASSES`, и в `LABELS`, а их `id` можете узнать на сайте [YOLO](https://docs.ultralytics.com/datasets/detect/coco#usage).

### Фильтр машин у края кадра

В коде есть настройка:

```python
SKIP_EDGE_TRUNCATED = True
EDGE_MARGIN = 5
```

Если `SKIP_EDGE_TRUNCATED = True`, скрипт пропускает машины, которые почти
касаются левого или правого края изображения. Это сделано из-за того что обрезанные
машины часто дают плохие кубоиды: часть объекта не видна, и автоматически
построить правильный объём сложно.

Если нужно размечать вообще все машины, можно поставить:

```python
SKIP_EDGE_TRUNCATED = False
```

Но тогда у крайних объектов кубоиды могут выглядеть хуже.




## Как создаётся XML CVAT 1.1

Скрипт создаёт корневой тег:

```xml
<annotations>
```

Затем добавляет версию:

```xml
<version>1.1</version>
```

После этого создаётся блок `meta/job`, где лежит служебная информация для CVAT:

```xml
<meta>
  <job>
    <id>1</id>
    <size>...</size>
    <mode>annotation</mode>
    <overlap>0</overlap>
    <created>...</created>
    <updated>...</updated>
    <subset>Train</subset>
    <start_frame>0</start_frame>
    <stop_frame>...</stop_frame>
    <labels>...</labels>
  </job>
</meta>
```

`size` - количество изображений.

`start_frame` - первый кадр, сейчас всегда `0`.

`stop_frame` - последний кадр, равен `количество изображений - 1`.

`labels` - список классов, которые CVAT должен знать заранее.

Для каждого класса создаётся label с типом `cuboid`:
ц
```xml
<label>
  <name>Car</name>
  <color>#4727e6</color>
  <type>cuboid</type>
  <attributes />
</label>
```

## Как обрабатываются изображения

Список изображений собирается так:

```python
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])
```

Скрипт берёт только файлы с расширениями:

- `.jpg`
- `.jpeg`
- `.png`

Файлы сортируются по имени. Это важно, потому что `id` изображения в XML
назначается по порядку после сортировки.

Для каждого изображения создаётся XML-тег:

```xml
<image id="0" name="frame.jpg" width="1920" height="1080">
  ...
</image>
```

`width` и `height` берутся из самого изображения через OpenCV:

```python
img_h, img_w = img.shape[:2]
```

## Как работает YOLO-детекция

Модель загружается один раз:

```python
model = YOLO(MODEL_NAME)
```

Для каждого изображения вызывается:

```python
results = model(img, verbose=False)
```

YOLO возвращает список найденных объектов. У каждого объекта есть:

- `box.cls` - ID класса;
- `box.conf` - уверенность модели;
- `box.xyxy` - координаты прямоугольника.

Координаты `xyxy` означают:

```text
x1, y1 - левый верхний угол
x2, y2 - правый нижний угол
```

Если класс не входит в `TARGET_CLASSES`, объект пропускается.

Если уверенность ниже `CONF_THRESHOLD`, объект тоже пропускается.

## Как из bbox строится cuboid

Главная функция:

```python
build_perspective_cuboid(x1, y1, x2, y2, img_w, img_h)
```

На вход она получает:

- координаты YOLO-бокса;
- ширину изображения;
- высоту изображения.

Сначала считается ширина и высота бокса:

```python
w = x2 - x1
h = y2 - y1
```

Если бокс слишком маленький или неправильный, функция возвращает `None`.

### Точка схода

Для перспективы используется условная точка схода дороги:

```python
vp_x = img_w * 0.52
vp_y = img_h * 0.35
```

Это точка в кадре, куда примерно сходятся линии дороги. От неё зависит
направление глубины кубоида.

Эти два значения можно подбирать:

```python
vp_x = img_w * 0.50
vp_y = img_h * 0.30
```

`vp_x` отвечает за горизонтальное положение точки схода.

`vp_y` отвечает за вертикальное положение точки схода.

### Направление глубины

Скрипт берёт нижний центр объекта:

```python
obj_x = (x1 + x2) / 2
obj_y = y2
```

Потом строит вектор от объекта к точке схода:

```python
direction = np.array([vp_x - obj_x, vp_y - obj_y], dtype=np.float32)
```

Этот вектор нормализуется, то есть превращается в направление длиной `1`.

Именно по этому направлению будет сдвинута дальняя грань кубоида.

### Глубина кубоида

Глубина считается так:

```python
scale = y2 / img_h
depth = w * (0.18 + 0.12 * scale)
depth = float(np.clip(depth, 8, min(w * 0.32, h * 0.22, 70)))
```

Логика такая:

- чем шире YOLO-бокс, тем больше глубина;
- чем ниже объект в кадре, тем он ближе к камере, поэтому глубина чуть больше;
- `np.clip` ограничивает глубину, чтобы кубоид не улетал слишком далеко.

Минимальная глубина сейчас `8` пикселей.

Максимальная глубина ограничивается меньшим из значений:

- `w * 0.32`
- `h * 0.22`
- `70`

Это защищает от слишком длинных и некрасивых кубоидов.

### Передняя грань

Передняя грань строится внутри YOLO-бокса:

```python
roof_y = y1 + h * CUBOID_ROOF_RATIO
```

Затем создаются четыре точки:

```text
f_tl - front top left
f_tr - front top right
f_bl - front bottom left
f_br - front bottom right
```

Нижняя часть передней грани стоит на `y2`, то есть на нижней границе
YOLO-бокса.

Верхняя часть стоит на `roof_y`.

### Задняя грань

Задняя грань получается сдвигом передней грани:

```python
b_tl = f_tl + offset
b_tr = f_tr + offset
b_bl = f_bl + offset
b_br = f_br + offset
```

`offset` - это направление к точке схода, умноженное на глубину.

Потом задняя грань немного сужается:

```python
shrink = w * 0.12
```

Это нужно, чтобы дальняя грань выглядела перспективно и не была такой же широкой,
как ближняя.

### Ограничение точек границами изображения

Все точки ограничиваются размерами изображения:

```python
np.clip(pt[0], 0, img_w - 1)
np.clip(pt[1], 0, img_h - 1)
```

Это значит, что координаты не выйдут за пределы кадра.

## Важный порядок точек для CVAT

Обычная геометрия задней грани имеет точки:

```text
b_tl, b_tr, b_bl, b_br
```

Но для CVAT 1.1 вторую грань нужно записывать в зеркальном порядке:

```python
"xtl2": b_tr[0],
"ytl2": b_tr[1],

"xbl2": b_br[0],
"ybl2": b_br[1],

"xtr2": b_tl[0],
"ytr2": b_tl[1],

"xbr2": b_bl[0],
"ybr2": b_bl[1],
```

Если записать вторую грань в обычном порядке, CVAT может нарисовать длинные
перекрученные линии. Именно поэтому порядок отличается от визуально ожидаемого.

 Правильный кубоид

![right](https://github.com/dumuzeyn/Auto-3D-cuboids-for-CVAT/blob/main/image/right3Dcuboid.png)

 Не правильный кубоид

![wrong](https://github.com/dumuzeyn/Auto-3D-cuboids-for-CVAT/blob/main/image/wrong3Dcuboid.png)

## Как кубоид записывается в XML

Для каждого найденного объекта создаётся тег:

```xml
<cuboid
  label="Car"
  source="auto"
  occluded="0"
  z_order="0"
  xtl1="..."
  ytl1="..."
  xbl1="..."
  ybl1="..."
  xtr1="..."
  ytr1="..."
  xbr1="..."
  ybr1="..."
  xtl2="..."
  ytl2="..."
  xbl2="..."
  ybl2="..."
  xtr2="..."
  ytr2="..."
  xbr2="..."
  ybr2="..." />
```

`label` - класс объекта.

`source="auto"` показывает, что разметка создана автоматически.

`occluded="0"` означает, что объект не отмечен как перекрытый.

`z_order="0"` - порядок отрисовки объекта в CVAT.

## Запуск

Перед запуском должны быть установлены зависимости:

```powershell
pip install ultralytics opencv-python numpy
```

Запуск:

```powershell
python .\auto_cvat_cuboids.py
```

После завершения скрипт напечатает:

```text
DONE
XML: cvat_perspective_cuboids.xml
```

## Что делать с XML

После генерации нужно открыть CVAT и импортировать файл:

```text
cvat_perspective_cuboids.xml
```

Формат импорта:

```text
CVAT for images 1.1
```

Важно, чтобы имена изображений в CVAT совпадали с именами файлов, которые были
в папке `IMAGE_DIR`. Например, если в XML указано:

```xml
<image name="frame_0001.jpg">
```

то в задаче CVAT тоже должен быть файл `frame_0001.jpg`.

## Что можно настраивать

Если кубоиды слишком низкие:

```python
CUBOID_ROOF_RATIO = 0.08
```

Если кубоиды слишком высокие:

```python
CUBOID_ROOF_RATIO = 0.16
```

Если кубоиды слишком длинные:

```python
depth = w * (0.14 + 0.08 * scale)
```

Если кубоиды слишком плоские:

```python
depth = w * (0.22 + 0.14 * scale)
```

Если много плохих детекций:

```python
CONF_THRESHOLD = 0.45
```

Если YOLO пропускает нужные машины:

```python
CONF_THRESHOLD = 0.25
```

## Ограничения

Этот скрипт не делает настоящую 3D-реконструкцию. Он строит приближённый
перспективный кубоид на основе 2D-бокса YOLO.

Поэтому результат зависит от:

- качества YOLO-детекции;
- положения точки схода;
- угла камеры;
- того, видна ли машина целиком;
- насколько объект перекрыт другими объектами.

Лучше всего скрипт работает на дорожных сценах с фиксированной камерой или
похожим ракурсом, где направление дороги стабильно.

## Частые проблемы

Если кубоиды снова становятся длинными, нужно уменьшить глубину:

```python
depth = w * (0.14 + 0.08 * scale)
```

Если кубоиды смотрят не в ту сторону, нужно менять точку схода:

```python
vp_x = img_w * 0.50
vp_y = img_h * 0.30
```

Если крайние машины не попадают в XML, это работает фильтр:

```python
SKIP_EDGE_TRUNCATED = True
```

Чтобы отключить его:

```python
SKIP_EDGE_TRUNCATED = False
```

Если CVAT не импортирует XML, нужно проверить:

- формат импорта выбран `CVAT for images 1.1`;
- имена изображений совпадают;
- labels `Car`, `Bus`, `Truck` есть в XML;
- файл XML не был переименован или повреждён.

> **Автор проекта :Зейналов У.Р.о.**

---

<h1 id="english-version">
Automatic creation of 3D cuboids for vehicles on the road scene in CVAT
</h1>

This project will help automate the process of marking cars, buses and trucks in CVAT. The project contains one script `auto_cvat_cuboids.py `, which automatically marks cars in images using YOLOv8 and creates cuboids with little manipulation, and then writes them to XML markup in CVAT 1.1 format. And also the script `convert_cvat2coco.py` which changes the annotation format from `CVAT for images 1.1` to `COCO`.

It can be imported into CVAT as markup in the `CVAT for images 1.1` format.

## What does the script do

The script works as follows:

1. Takes all images from the folder `IMAGE_DIR'.

2. Loads the YOLOv8 model.

3. Starts object detection for each image.

4. Leaves only the necessary classes, in my case they were `Car`, `Bus`, `Truck'.

5. Cuts off weak detections based on the confidence threshold.

6. A `3D cuboid` is constructed from each 2D bounding box (`bbox`).

7. Writes all cuboids to an XML file in CVAT 1.1 format.

8. Download annotation in CVAT.

The idea is simple: YOLO gives you a regular rectangle around the car, and the script adds a second face to it to make a pseudo-3D cuboid.

## Settings

### Basic variables

At the beginning of the file are the __main parameters__:

```python
IMAGE_DIR = r"C:\Users\path\in\image\Train"
OUTPUT_XML = "cvat_perspective_cuboids.xml"
MODEL_NAME = "yolov8x.pt"
CONF_THRESHOLD = 0.35
CUBOID_ROOF_RATIO = 0.12
```

`IMAGE_DIR` - folder with images.

`OUTPUT_XML`- is the name of the XML file to be created.

`MODEL_NAME` - is the YOLO model. Currently in use `yolov8x.pt `.

`CONF_THRESHOLD` - minimum confidence of YOLO. If the value is lower than `0.35`,
the object will not be included in the markup.

`CUBOID_ROOF_RATIO` - height of the upper face of the cuboid inside the YOLO box. Than
The lower the value, the higher the cuboid.:

### YOLO Classes

In the project, the script takes only three classes of YOLO:

```python
TARGET_CLASSES = {
    2: "Car",
    5: "Bus",
    7: "Truck",
}
```
```python
LABELS = [
    ("Car", "#4727e6"), #Next to the names are the colors selected in CVAT
    ("Bus", "#6349ea"),
    ("Truck", "#a295e6"),
]
```
If other classes are needed, they need to be added to both `TARGET_CLASSES` and `LABELS`, and their `id` can be found on the [YOLO](https://docs.ultralytics.com/datasets/detect/coco#usage ).

### Filter the machines at the edge of the frame

There is a setting in the code:

```python
SKIP_EDGE_TRUNCATED = True
EDGE_MARGIN = 5
```

If `SKIP_EDGE_TRUNCATED = True`, the script skips cars that almost
touch the left or right edge of the image. This is done because cropped
machines often produce poor cuboids: part of the object is not visible, and it is difficult to automatically
build the correct volume.

If you need to mark up all cars at all, you can put:

```python
SKIP_EDGE_TRUNCATED = False
```

But then the cuboids of the extreme objects may look worse.

## How XML CVAT 1 is created.1

The script creates the root tag:

```xml
<annotations>
```

Then adds the version:

```xml
<version>1.1</version>
```

After that, a `meta/job` block is created, where the service information for CVAT is located.:

```xml
<meta>
  <job>
    <id>1</id>
    <size>...</size>
    <mode>annotation</mode>
    <overlap>0</overlap>
    <created>...</created>
    <updated>...</updated>
    <subset>Train</subset>
    <start_frame>0</start_frame>
    <stop_frame>...</stop_frame>
    <labels>...</labels>
  </job>
</meta>
```

`size' - the number of images.

`start_frame` is the first frame, it is always `0` now.

`stop_frame` - the last frame is equal to `number of images - 1`.

`labels` - is a list of classes that the CVAT must know in advance.

A label with the type `cuboid` is created for each class:
c
```xml
<label>
  <name>Car</name>
  <color>#4727e6</color>
  <type>cuboid</type>
  <attributes />
</label>
```

## How images are processed

The list of images is collected as follows:

```python
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])
```

The script takes only files with extensions:

- `.jpg`
- `.jpeg`
- `.png`

The files are sorted by name. This is important because the `id` of the image in XML
is assigned in order after sorting.

An XML tag is created for each image.:

```xml
<image id="0" name="frame.jpg" width="1920" height="1080">
  ...
</image>
```

The `width` and `height` are taken from the image itself via OpenCV:

```python
img_h, img_w = img.shape[:2]
```

## How YOLO detection works

The model is loaded once:

```python
model = YOLO(MODEL_NAME)
```

For each image, it is called:

```python
results = model(img, verbose=False)
```

YOLO returns a list of found objects. Each object has:

- `box.cls` - Class ID;
- `box.conf` - confidence of the model;
- `box.xyxy` - coordinates of the rectangle.

The coordinates of `xyxy` mean:

```text
x1, y1 - upper left corner
x2, y2 - lower right corner
```

If the class is not included in the `TARGET_CLASSES`, the object is skipped.

If the confidence is lower than `CONF_THRESHOLD`, the object is also skipped.

## How a cuboid is built from a bbox

Main function:

```python
build_perspective_cuboid(x1, y1, x2, y2, img_w, img_h)
```

At the entrance, she gets:

- coordinates of the YOLO box;
- the width of the image;
- the height of the image.

The width and height of the box are calculated first.:

```python
w = x2 - x1
h = y2 - y1
```

If the box is too small or incorrect, the function returns `None`.

### Vanishing point

A conditional exit point is used for perspective.:

```python
vp_x = img_w * 0.52
vp_y = img_h * 0.35
```

This is the point in the frame where the road lines roughly converge. 
The direction of the depth of the cuboid depends on it.

These two values can be matched:

```python
vp_x = img_w * 0.50
vp_y = img_h * 0.30
```

`vp_x` is responsible for the horizontal position of the vanishing point.

`vp_y` is responsible for the vertical position of the vanishing point.

### Depth direction

The script takes the lower center of the object:

```python
obj_x = (x1 + x2) / 2
obj_y = y2
```

Then it builds a vector from the object to the vanishing point.:

```python
direction = np.array([vp_x - obj_x, vp_y - obj_y], dtype=np.float32)
```

This vector is normalized, that is, it turns into a direction of length `1`.

It is in this direction that the far side of the cuboid will be shifted.

### Cuboid depth

The depth is calculated as follows:

```python
scale = y2 / img_h
depth = w * (0.18 + 0.12 * scale)
depth = float(np.clip(depth, 8, min(w * 0.32, h * 0.22, 70)))
```

The logic is this:

- the wider the YOLO box, the greater the depth;
- the lower the object in the frame, the closer it is to the camera, so the depth is slightly greater.;
- `np.clip` limits the depth so that the cuboid does not fly too far.

The minimum depth is currently `8` pixels.

The maximum depth is limited to the smallest of the values:

- `w * 0.32`
- `h * 0.22`
- `70`

This protects against overly long and ugly cuboids.

### The front face

The front face is built inside the YOLO box:

```python
roof_y = y1 + h * CUBOID_ROOF_RATIO
```

Then four points are created:

```text
f_tl - front top left
f_tr - front top right
f_bl - front bottom left
f_br - front bottom right
```

The lower part of the front face is located on `y2`, that is, on the lower border
of the YOLO box.

The upper part is set to `roof_y`.

### Back face

The back face is obtained by shifting the front face:

```python
b_tl = f_tl + offset
b_tr = f_tr + offset
b_bl = f_bl + offset
b_br = f_br + offset
```

The `offset` is the direction to the vanishing point multiplied by the depth.

Then the back edge narrows slightly:

```python
shrink = w * 0.12
```

This is necessary so that the far side looks promising and is not as wide
as the near side.

### Limiting points to image boundaries

All points are limited by the size of the image:

```python
np.clip(pt[0], 0, img_w - 1)
np.clip(pt[1], 0, img_h - 1)
```

This means that the coordinates will not go beyond the frame.

## Important point order for CVAT

The usual geometry of the back face has points:

```text
b_tl, b_tr, b_bl, b_br
```

But for CVAT 1.1, the second face must be recorded in the mirror order.:

```python
"xtl2": b_tr[0],
"ytl2": b_tr[1],

"xbl2": b_br[0],
"ybl2": b_br[1],

"xtr2": b_tl[0],
"ytr2": b_tl[1],

"xbr2": b_bl[0],
"ybr2": b_bl[1],
```

If you write the second face in the usual order, CVAT can draw long
twisted lines. That is why the order differs from what is visually expected.

 The correct cuboid

![right](https://github.com/dumuzeyn/Auto-3D-cuboids-for-CVAT/blob/main/image/right3Dcuboid.png)

 Wrong cuboid

![wrong](https://github.com/dumuzeyn/Auto-3D-cuboids-for-CVAT/blob/main/image/wrong3Dcuboid.png)

## How a cuboid is written in XML

An tag is created for each found object.:

```xml
<cuboid
  label="Car"
  source="auto"
  occluded="0"
  z_order="0"
  xtl1="..."
  ytl1="..."
  xbl1="..."
  ybl1="..."
  xtr1="..."
  ytr1="..."
  xbr1="..."
  ybr1="..."
  xtl2="..."
  ytl2="..."
  xbl2="..."
  ybl2="..."
  xtr2="..."
  ytr2="..."
  xbr2="..."
  ybr2="..." />
```

`label` is the class of the object.

`source="auto"` indicates that the markup was created automatically.

`occluded="0"` means that the object is not marked as occluded.

`z_order="0"` - the order in which the object is drawn in CVAT.

## Launch

Dependencies must be installed before launching:

```powershell
pip install ultralytics opencv-python numpy
```

Launch:

```powershell
python .\auto_cvat_cuboids.py
```

After completion, the script will print:

```text
DONE
XML: cvat_perspective_cuboids.xml
```

## What to do with XML

After generation, you need to open the CVAT and import the file.:

```text
cvat_perspective_cuboids.xml
```

Import format:

```text
CVAT for images 1.1
```

It is important that the names of the images in the CVAT match the names of the files that were
in the `IMAGE_DIR` folder. For example, if the XML states:

```xml
<image name="frame_0001.jpg">
```

then there should also be a file in the CVAT task `frame_0001.jpg `.

## What can be customized

If the cuboids are too low:

```python
CUBOID_ROOF_RATIO = 0.08
```

If the cuboids are too high:

```python
CUBOID_ROOF_RATIO = 0.16
```

If the cuboids are too long:

```python
depth = w * (0.14 + 0.08 * scale)
```

If the cuboids are too flat:

```python
depth = w * (0.22 + 0.14 * scale)
```

If there are a lot of bad detections:

```python
CONF_THRESHOLD = 0.45
```

If YOLO skips the right cars:

```python
CONF_THRESHOLD = 0.25
```

## Restrictions

This script does not do a real 3D reconstruction. He's building an approximate
a perspective cuboid based on the 2D YOLO box.

Therefore, the result depends on:

- the quality of YOLO detection;
- vanishing point positions;
- camera angle;
- whether the whole car is visible;
- how much the object is blocked by other objects.

The script works best on road scenes with a fixed camera or
a similar angle, where the direction of the road is stable.

## Common problems

If the cuboids become long again, you need to reduce the depth.:

```python
depth = w * (0.14 + 0.08 * scale)
```

If the cuboids are facing the wrong way, you need to change the vanishing point.:

```python
vp_x = img_w * 0.50
vp_y = img_h * 0.30
```

If the edge machines don't get into the XML, this is how the filter works.:

```python
SKIP_EDGE_TRUNCATED = True
```

To disable it:

```python
SKIP_EDGE_TRUNCATED = False
```

If CVAT does not import XML, you need to check:

- the import format is `CVAT for images 1.1`;
- the names of the images match;
- labels `Car`, `Bus`, `Truck` are in XML;
- The XML file has not been renamed or corrupted.

> **The author of the project : Zeynalov U.R.O.**