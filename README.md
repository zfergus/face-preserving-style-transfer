# Face Preserving Style Transfer

**Fast style transfer with facial preservation.**

#### Abstract

Style transfer is the act of stylizing an input image based on the style of another image [[Gatys et al. 2015]](https://arxiv.org/abs/1508.06576). We extend the [Fast Style Transfer](https://cs.stanford.edu/people/jcjohns/eccv16/) network created by Johnson et al. to include an extra perceptual loss term for the loss in facial details [2016]. The face loss is calculated by using [Multi-task Cascaded Convolutional Networks [Zhang et al. 2016]](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) to find faces and [OpenFace [Amos et al. 2016]](https://cmusatyalab.github.io/openface/) to compute a 128-dimension face descriptor. The loss is then the squared distance between face descriptors.

Created by Zachary Ferguson for CSCI-GA.2271: Computer Vision (Fall 2018) at New York University.

#### Report
* [Compressed version (30 MB): ferguson-zachary-report-small.pdf](ferguson-zachary-report-small.pdf)
* [Full Version (86 MB): ferguson-zachary-report.pdf](ferguson-zachary-report.pdf)

#### Model

<p align="center">
    <img src="images/figures/model.png" width="90%"><br>
    We train an image transformation network to stylize images based on the style of a target style image. The image transformation network is trained using the perceptual loss computed by neural network(s). We extend the original network of Johnson et al. (circled in red) by including an additional face loss term computed by a new face loss network (circled in grey).
</p>

## Usage

We implement our network in Python using PyTorch. Additionally, Pillow and NumPy are used to handle images and other miscellaneous tasks.

### Stylizing Images

```bash
python -m src.stylize --content-image [path/to/content.ext] --style-model [path/to/model.pth] --output [path/to/output.png]
```

#### Pretrained Model

We provide a number of different pretrained style model in the `models`
directory. The style models include:

* `models/grande-jatte.pth`: Style of *A Sunday Afternoon on the Island of La Grande Jatte*
by Georges Seurat
    * style image: `images/styles/A-Sunday-Afternoon-on-the-Island-of-La-Grande-Jatte.jpg`
    * example results: `images/results/grande-jatte/`
* `models/manga.pth`: Style of artwork from *Fullmetal Alchemist* by Hiromu Arakawa
    * style image: `images/styles/manga.png`
    * example results: `images/results/manga/`
* `models/manga-face.pth`: Style of artwork from *Fullmetal Alchemist* by Hiromu Arakawa with facial preservation
    * style image: `images/styles/manga.png`
    * example results: `images/results/manga-face/`
* `models/mosaic.pth`: Style of a mosaic tiling
    * style image: `images/styles/mosaic.jpg`
    * example results: `images/results/mosaic/`
* `models/mosaic-face.pth`: Style of a mosaic tiling with facial preservation
    * style image: `images/styles/mosaic.jpg`
    * example results: `images/results/mosaic-face/`
* `models/rains-rustle.pth`: Style of *Rain's Rustle* by Leonid Afremov
    * style image: `images/styles/Rain's-Rustle-by-Leonid-Afremov.jpg`
    * example results: `images/results/rains-rustle/`
* `models/stary-night.pth`: Style of *The Stary Night* by Vincent van Gogh
    * style image: `images/styles/Starry-Night-by-Vincent-Van-Gogh.jpg`
    * example results: `images/results/stary-night/`
* `models/wave.pth`: Style of *The Great Wave off Kanagawa* by Hokusai
    * style image: `images/styles/Great-Wave-off-Kanagawa.jpg`
    * example results: `images/results/wave/`

### Training for New Styles

To train a new style model you need to first download a image dataset
(the pretrained models were trained using the COCO 2017 Train Images
[118K/18GB], no need for the annotations). Then you can train a model using the
following command

```bash
python -m src.train --content-images [path/to/content/] --style-image [path/to/style.jpg] --output [path/to/output/] [--face]
```

where python is Python >=3.5, `path/to/content/` is the path to the root of the
training dataset, `path/to/style.jpg` is the image of the style to learn, and
`--face` turns on facial preservation.

## Results of Fast Style Transfer

We reimplemented the Fast Style Transfer network presented in ["Perceptual Losses for Real-Time Style Transfer
and Super-Resolution" [Johnson et al. 2016]](https://cs.stanford.edu/people/jcjohns/eccv16/).

### Mosaic Style
<p align="center">
    <img src="images/styles/mosaic.jpg" width="300px">
    <br>
    <img src="images/content/amber.jpg" width="24%">
    <img src="images/results/mosaic/amber.png" width="24%">

    <img src="images/results/mosaic/corgi-square.png" width="24%">
    <img src="images/content/corgi-square.png" width="24%">
    <br>
    <img src="images/content/tokyo-square.png" width="24%">
    <img src="images/results/mosaic/tokyo-square.png" width="24%">

    <img src="images/results/mosaic/golden-gate-bridge-square.png" width="24%">
    <img src="images/content/golden-gate-bridge-square.png" width="24%">
    <br>
    <img src="images/content/manga-square.png" width="24%">
    <img src="images/results/mosaic/restylized/manga-square.png" width="24%">

    <img src="images/results/mosaic/wave-square.png" width="24%">
    <img src="images/content/wave-square.png" width="24%">
</p>

### Manga Style
<p align="center">
    <img src="images/styles/manga.png" width="300px">
    <br>
    <img src="images/content/amber.jpg" width="24%">
    <img src="images/results/manga/amber.png" width="24%">

    <img src="images/results/manga/corgi-square.png" width="24%">
    <img src="images/content/corgi-square.png" width="24%">
    <br>
    <img src="images/content/tokyo-square.png" width="24%">
    <img src="images/results/manga/tokyo-square.png" width="24%">

    <img src="images/results/manga/golden-gate-bridge-square.png" width="24%">
    <img src="images/content/golden-gate-bridge-square.png" width="24%">
    <br>
    <img src="images/content/surfing-square.jpg" width="24%">
    <img src="images/results/manga/surfing-square.png" width="24%">

    <img src="images/results/manga/viking-square.png" width="24%">
    <img src="images/content/viking-square.png" width="24%">
</p>

### Rain's Rustle Style
<p align="center">
    <img src="images/styles/Rain's-Rustle-by-Leonid-Afremov.jpg" width="300px">
    <br>
    <img src="images/content/amber.jpg" width="24%">
    <img src="images/results/rains-rustle/amber.png" width="24%">

    <img src="images/results/rains-rustle/corgi-square.png" width="24%">
    <img src="images/content/corgi-square.png" width="24%">
    <br>
    <img src="images/content/tokyo-square.png" width="24%">
    <img src="images/results/rains-rustle/tokyo-square.png" width="24%">

    <img src="images/results/rains-rustle/golden-gate-bridge-square.png" width="24%">
    <img src="images/content/golden-gate-bridge-square.png" width="24%">
    <br>
    <img src="images/content/surfing-square.jpg" width="24%">
    <img src="images/results/rains-rustle/surfing-square.png" width="24%">

    <img src="images/results/rains-rustle/viking-square.png" width="24%">
    <img src="images/content/viking-square.png" width="24%">
</p>

### Great Wave off Kanagawa Style
<p align="center">
    <img src="images/styles/Great-Wave-off-Kanagawa.jpg" width="300px">
    <br>
    <img src="images/content/amber.jpg" width="24%">
    <img src="images/results/wave/amber.png" width="24%">

    <img src="images/results/wave/corgi-square.png" width="24%">
    <img src="images/content/corgi-square.png" width="24%">
    <br>
    <img src="images/content/tokyo-square.png" width="24%">
    <img src="images/results/wave/tokyo-square.png" width="24%">

    <img src="images/results/wave/golden-gate-bridge-square.png" width="24%">
    <img src="images/content/golden-gate-bridge-square.png" width="24%">
    <br>
    <img src="images/content/surfing-square.jpg" width="24%">
    <img src="images/results/wave/surfing-square.png" width="24%">

    <img src="images/results/wave/viking-square.png" width="24%">
    <img src="images/content/viking-square.png" width="24%">
</p>

## Results of Face Preserving Style Transfer

Our additional face loss allows us to train a image transformation network that
preserves facial details. We train a face preserving network for two styles
(the manga and mosaic style).

### Manga Style with Facial Preservation
| ![](images/styles/manga.png) | **Without Facial Preservation** | **With Facial Preservation**
|:----------------------------:|:-------------------------------:|:-----------------------------:|
| ![](images/faces/john-snow.jpg) | ![](images/results/manga-face/john-snow-face=false.png) | ![](images/results/manga-face/john-snow-face=true.png) |
| ![](images/faces/stranger-things.jpg) | ![](images/results/manga-face/stranger-things-face=false.png) | ![](images/results/manga-face/stranger-things-face=true.png) |
| ![](images/faces/elvis.jpg) | ![](images/results/manga-face/elvis-face=false.png) | ![](images/results/manga-face/elvis-face=true.png) |
| ![](images/faces/jackie-chan.jpg) | ![](images/results/manga-face/jackie-chan-face=false.png) | ![](images/results/manga-face/jackie-chan-face=true.png) |

### Mosaic Style with Facial Preservation
| ![](images/styles/mosaic.jpg) | Without Facial Preservation | With Facial Preservation
|:-----:|:-----:|:-----:|
| ![](images/faces/john-snow.jpg) | ![](images/results/mosaic-face/john-snow-face=false.png) | ![](images/results/mosaic-face/john-snow-face=true.png) |
| ![](images/faces/stranger-things.jpg) | ![](images/results/mosaic-face/stranger-things-face=false.png) | ![](images/results/mosaic-face/stranger-things-face=true.png) |
| ![](images/faces/elvis.jpg) | ![](images/results/mosaic-face/elvis-face=false.png) | ![](images/results/mosaic-face/elvis-face=true.png) |
| ![](images/faces/jackie-chan.jpg) | ![](images/results/mosaic-face/jackie-chan-face=false.png) | ![](images/results/mosaic-face/jackie-chan-face=true.png) |
