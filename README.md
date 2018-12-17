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
| ![](images/content/amber.jpg) | ![](images/results/mosaic/amber.png) | ![](images/results/wave/amber.png) | ![](images/results/grande-jatte/amber.png) | ![](images/results/stary-night/amber.png) | ![](images/results/rains-rustle/amber.png) | ![](images/results/manga/amber.png) |
| ![](images/content/corgi.jpg) | ![](images/results/mosaic/corgi.png) | ![](images/results/wave/corgi.png) | ![](images/results/grande-jatte/corgi.png) | ![](images/results/stary-night/corgi.png) | ![](images/results/rains-rustle/corgi.png) | ![](images/results/manga/corgi.png) |
| ![](images/content/tokyo.jpg) | ![](images/results/mosaic/tokyo.png) | ![](images/results/wave/tokyo.png) | ![](images/results/grande-jatte/tokyo.png) | ![](images/results/stary-night/tokyo.png) | ![](images/results/rains-rustle/tokyo.png) | ![](images/results/manga/tokyo.png) |
| ![](images/content/golden-gate-bridge.jpg) | ![](images/results/mosaic/golden-gate-bridge.png) | ![](images/results/wave/golden-gate-bridge.png) | ![](images/results/grande-jatte/golden-gate-bridge.png) | ![](images/results/stary-night/golden-gate-bridge.png) | ![](images/results/rains-rustle/golden-gate-bridge.png) | ![](images/results/manga/golden-gate-bridge.png) |
| ![](images/content/surfing.jpg) | ![](images/results/mosaic/surfing.png) | ![](images/results/wave/surfing.png) | ![](images/results/grande-jatte/surfing.png) | ![](images/results/stary-night/surfing.png) | ![](images/results/rains-rustle/surfing.png) | ![](images/results/manga/surfing.png) |
| ![](images/content/viking.jpg) | ![](images/results/mosaic/viking.png) | ![](images/results/wave/viking.png) | ![](images/results/grande-jatte/viking.png) | ![](images/results/stary-night/viking.png) | ![](images/results/rains-rustle/viking.png) | ![](images/results/manga/viking.png) |
