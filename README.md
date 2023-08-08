# YOLOv8 Object Detection on a Custom Dataset Tutorial

#What is YOLOv8

YOLOv8 is the latest iteration in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduces new features and optimizations that make it an ideal choice for various object detection tasks in a wide range of applications.

![yolov8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

#Key Features

- **Advanced Backbone and Neck Architectures:** YOLOv8 employs state-of-the-art backbone and neck architectures, resulting in improved feature extraction and object detection performance.
- **Anchor-free Split Ultralytics Head:** YOLOv8 adopts an anchor-free split Ultralytics head, which contributes to better accuracy and a more efficient detection process compared to anchor-based approaches.

- **Optimized Accuracy-Speed Tradeoff:** With a focus on maintaining an optimal balance between accuracy and speed, YOLOv8 is suitable for real-time object detection tasks in diverse application areas.

- **Variety of Pre-trained Models:** YOLOv8 offers a range of pre-trained models to cater to various tasks and performance requirements, making it easier to find the right model for your specific use case.

---

#How to Use YOLOv8

See the [YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8/) for full documentation on training, validation, prediction and deployment.

<details open>
<summary>Install</summary>
<br>
Install the ultralytics package using pip.

```
pip install ultralytics
```

For alternative installation methods, please check the [Quickstart Guide](https://docs.ultralytics.com/quickstart/).

</details>

<details open>
<summary>Usage</summary>
<br>

**CLI**
YOLOv8 may be used directly in the Command Line Interface (CLI) with a yolo command:

```
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

For more information, please check the [CLI Docs](hhttps://docs.ultralytics.com/usage/cli/).

**Python**
YOLOv8 may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```
#yolov8_pre-trained.py
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

Please check YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python/) for more examples.

</details>

---

#YOLOv8 Custom Dataset Tutorial

<details open>
<summary>Create a Custom Dataset</summary>
<br>
To train Yolov8 object detection in a special dataset, the first thing you need to do is collect data for to create your custom dataset.

Creating a dataset for training an object detection model like YOLO requires careful planning and data collection. Here are the general steps you can follow to collect data and create a dataset:

1.  **Define Object Classes:**
    Determine the specific object classes you want to detect with your YOLO model. This could be any category of objects, such as cars, pedestrians, animals, or custom classes relevant to your application.
2.  **Data Collection:**
    There are various methods to collect data for your dataset:

-  **Manual Collection:** Capture images or videos using a camera or smartphone. Ensure that the objects you want to detect are well-represented in different environments and lighting conditions.
-  **Public Datasets:** Consider using publicly available datasets that are relevant to your task. Popular datasets like [COCO](https://cocodataset.org/#home), [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), or [Open Images](https://storage.googleapis.com/openimages/web/index.html) contain diverse object categories and bounding box annotations.
- **Web Scraping:** If applicable, you can scrape images or videos from the web using tools like BeautifulSoup or [Scrapy](https://scrapy.org) (ensure you have permission to use the data).

3.  **Data Annotation:**
    Annotate the collected images or videos with bounding boxes and corresponding class labels. There are various annotation tools available, such as [LabelImg](https://pypi.org/project/labelImg/), [RectLabel](https://rectlabel.com), [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/), [CVAT](https://www.cvat.ai) etc.

4.  **Dataset Structure:**
    Before you train YOLOv8 with your dataset you need to be sure if your dataset file format is proper.
    The proper format is to have two directories: images and labels. In the images directory there are our annotated images (.jpg) that we download before and in the labels directory there are annotation label files (.txt) which has the same names with related images. Just like this:

    <details open><summary>data</summary><blockquote>
            <details open><summary>images</summary><blockquote>
            image_1.jpg <br>
            image_2.jpg <br>
            image_3.jpg <br>
            </blockquote></details>
            <details open><summary>labels</summary><blockquote>
            image_1.txt <br>
            image_2.txt <br>
            image_3.txt <br>
            </blockquote></details>
    </blockquote></details>

</details>
<details open>
<summary>Train YOLOv8</summary>
<br>

This is basically the same thing as we see at the [`usage`](#how-to-use-yolov8) section. But this time we need to change the original code a little and remove the 'pre-trained dataset' part to train our custom dataset:

```
# yolov8_custom-dataset.py
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=3)  # train the model

```
Epoch, in machine learning, refers to the one entire passing of training data through the algorithm. It's a hyperparameter that determines the process of training the machine learning model. You can check online to learn more about [epoch](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning).
We need a configuration (.yaml) file with the same directory as main.py. This simple configuration file have a few keys which are: path, train, val and names. You need to set this keys with the way you need:

```
# config.yaml
path: /home/user/Desktop/yolov8_tutorial/code/data      # dataset root directory
train: images   # train images (relative to 'path' directory)
val: images     # val images (relative to 'path' directory)

# Classes
names:   # these are the names of the classes that you have
0: person
1: headphones
2: backpack

```
After train the YOLOv8, we are finally ready to test it.

</details>
<details open>
<summary>Test YOLOv8</summary>
<br>

You can test it with many ways. There is a code part for predict on an image, you can check ultralytics's website for more information about YOLOv8 [predict mode](https://docs.ultralytics.com/modes/predict/).
```
# test_custom-dataset.py
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train2/weights/best.pt')  # load your custom model

# Predict with the model
results = model('test_image.jpg')  # predict on an image
```
</details>
