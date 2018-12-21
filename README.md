# Face Preserving Style Transfer

Zachary Ferguson (zfergus@nyu.edu)

Fast style transfer with facial preservation.

Extends the Fast Style Transfer by Johnson et al. to include an extra
perceptual loss term for the loss in facial details. The face loss is
calculated by using MTCNN to find faces and OpenFace to compute a 128-dimension
face descriptor. The loss is then the squared distance between face
descriptors.

**Report:**
* [ferguson-zachary-report-small.pdf (compressed, 30 MB)](ferguson-zachary-reposrt-small.pdf)
* [ferguson-zachary-report.pdf (full resolution, 271 MB)](ferguson-zachary-reposrt.pdf)

## Stylizing Images

```bash
python -m src.stylize --content-image [path/to/content.ext] --style-model [path/to/model.pth] --output [path/to/output.png]
```

### Pretrained Model

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

## Training for New Styles

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

|  | ![](images/styles/mosaic.jpg) | ![](images/styles/Great-Wave-off-Kanagawa.jpg)  | ![](images/styles/A-Sunday-Afternoon-on-the-Island-of-La-Grande-Jatte.jpg)  | ![](images/styles/Starry-Night-by-Vincent-Van-Gogh.jpg)  | ![](images/styles/Rain's-Rustle-by-Leonid-Afremov.jpg) | ![](images/styles/manga.png) |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![](images/content/amber.jpg) | ![](images/results/mosaic/amber.png) | ![](images/results/wave/amber.png) | ![](images/results/grande-jatte/amber.png) | ![](images/results/stary-night/amber.png) | ![](images/results/rains-rustle/amber.png) | ![](images/results/manga/amber.png) |
| ![](images/content/corgi.jpg) | ![](images/results/mosaic/corgi.png) | ![](images/results/wave/corgi.png) | ![](images/results/grande-jatte/corgi.png) | ![](images/results/stary-night/corgi.png) | ![](images/results/rains-rustle/corgi.png) | ![](images/results/manga/corgi.png) |
| ![](images/content/tokyo.jpg) | ![](images/results/mosaic/tokyo.png) | ![](images/results/wave/tokyo.png) | ![](images/results/grande-jatte/tokyo.png) | ![](images/results/stary-night/tokyo.png) | ![](images/results/rains-rustle/tokyo.png) | ![](images/results/manga/tokyo.png) |
| ![](images/content/golden-gate-bridge.jpg) | ![](images/results/mosaic/golden-gate-bridge.png) | ![](images/results/wave/golden-gate-bridge.png) | ![](images/results/grande-jatte/golden-gate-bridge.png) | ![](images/results/stary-night/golden-gate-bridge.png) | ![](images/results/rains-rustle/golden-gate-bridge.png) | ![](images/results/manga/golden-gate-bridge.png) |
| ![](images/content/surfing.jpg) | ![](images/results/mosaic/surfing.png) | ![](images/results/wave/surfing.png) | ![](images/results/grande-jatte/surfing.png) | ![](images/results/stary-night/surfing.png) | ![](images/results/rains-rustle/surfing.png) | ![](images/results/manga/surfing.png) |
| ![](images/content/viking.jpg) | ![](images/results/mosaic/viking.png) | ![](images/results/wave/viking.png) | ![](images/results/grande-jatte/viking.png) | ![](images/results/stary-night/viking.png) | ![](images/results/rains-rustle/viking.png) | ![](images/results/manga/viking.png) |

## Results of Face Preserving Style Transfer
