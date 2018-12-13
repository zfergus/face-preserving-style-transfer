#!/bin/bash
python3 ./stylize.py --style-model $1 --content-image ./images/content/amber.jpg --output $2/amber.png
python3 ./stylize.py --style-model $1 --content-image ./images/content/corgi.jpg --output $2/corgi.png
python3 ./stylize.py --style-model $1 --content-image ./images/content/tokyo.jpg --output $2/tokyo.png
python3 ./stylize.py --style-model $1 --content-image ./images/content/golden-gate-bridge.jpg --output $2/golden-gate-bridge.png
python3 ./stylize.py --style-model $1 --content-image ./images/content/surfing.jpg --output $2/surfing.png
python3 ./stylize.py --style-model $1 --content-image ./images/content/viking.jpg --output $2/viking.png
python3 ./stylize.py --style-model $1 --content-image ./images/styles/mosaic.jpg --output $2/mosaic.png
python3 ./stylize.py --style-model $1 --content-image ./images/styles/Great_Wave_off_Kanagawa.jpg --output $2/wave.png
python3 ./stylize.py --style-model $1 --content-image ./images/styles/A-Sunday-Afternoon-on-the-Island-of-La-Grande-Jatte.jpg --output $2/grande-jatte.png
python3 ./stylize.py --style-model $1 --content-image ./images/styles/Starry-Night-by-Vincent-Van-Gogh.jpg --output $2/stary-night.png
python3 ./stylize.py --style-model $1 --content-image ./images/styles/Rain\'s-Rustle-by-Leonid-Afremov.jpg --output $2/rains-rustle.png
