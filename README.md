# NeuralMaterials

This repository contains:
- spectrum autoencoder
- neural Phong illumination model
- neural thin film reflection function
- nothing else

Spectrum example:

![spectrum](https://github.com/RomanRodionov/NeuralMaterials/blob/main/spectrum_example.png?raw=true)

GT Phong:

![gt_phong](https://github.com/RomanRodionov/NeuralMaterials/blob/main/tests/phong/gt_2.png?raw=true)

Neural Phong:

![neural_phong](https://github.com/RomanRodionov/NeuralMaterials/blob/main/tests/phong/neural_2.png?raw=true)

Thin films reflection example:

![films](https://github.com/RomanRodionov/NeuralMaterials/blob/main/films_example.png?raw=true)

Install:

    git clone --recursive git@github.com:RomanRodionov/NeuralMaterials.git
    python3 -m venv ./venv
    source venv/bin/activate
    pip install -r requirements.txt

To run spectrum encoding demo use:

    python3 spectrum_demo.py

To run phong demo use:

    python3 phong_demo.py

To run thin_films demo use:

    python3 films_demo.py