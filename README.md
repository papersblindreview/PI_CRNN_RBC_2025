# PI_CRNN2025

Supplemental codes for "A Physics-Informed Convolutional Long Short Term Memory Statistical Model for Fluid Thermodynamics Simulations".

There are four scripts in the `code` folder:

1) `functions.py` contains helper functions to load and preprocess the data
2) `cae_model.py` contains code to run the CAE
3) `pi_crnn_model.py` contains code to run the physics-informed spatiotemporal model, conditional on the trained CAE
4) `uq.py` contains code to reproduce predictions intervals using conformal method

The `code` folder also contains a reproducible version able to run on a desktop machine. The files are the same as (1-4) and are in the `for_desktop` directory inside of `code`.

The DNS data for Rayleigh-Benard Convection used in this work is in the data folder, along with files containing the coordinates and physical constants. These can be reproduced from [this](https://git.uwaterloo.ca/SPINS/SPINS_main) public repository.

In the same folder as the data, a user should download the three files in the code folder. To reproduce the results from the manuscript, the user should first run `cae_model.py` to train the CAE portion of the model. Then, run `pi_crnn_model.py` to train and produce forecasts for the proposed PI-CRNN approach. The `uq.py` can be run after training the model to obtain predictions intervals and verify proper coverage.
