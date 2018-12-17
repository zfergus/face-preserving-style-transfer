# Facial Preserving Style Transfer

Fast style transfer with facial preservation.

## Training for new styles

```bash
python src/train.py --content-images [path/to/content/] --style-image [path/to/style.jpg] --output [path/to/output/] [--face]
```

## Stylizing images

```bash
python src/stylize.py --content-image [path/to/content.ext] --style-model [path/to/model.pth] --output [path/to/output.png]
```

## Results of base fast style transfer

|  | ![](images/styles/mosaic.jpg) | ![](images/styles/Great_Wave_off_Kanagawa.jpg)  | ![](images/styles/A-Sunday-Afternoon-on-the-Island-of-La-Grande-Jatte.jpg)  | ![](images/styles/Starry-Night-by-Vincent-Van-Gogh.jpg)  | ![](images/styles/Rain's-Rustle-by-Leonid-Afremov.jpg) | ![](images/styles/manga.png) |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![](images/content/corgi.jpg) | ![](images/results/mosaic/corgi.png) | ![](images/results/wave/corgi.png) | ![](images/results/grande-jatte/corgi.png) | ![](images/results/stary-night/corgi.png) | ![](images/results/rains-rustle/corgi.png) | ![](images/results/manga/corgi.png) |
