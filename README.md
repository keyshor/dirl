Implementation of the compositional RL (DiRL) approach presented in the paper "[Compositional Reinforcement Learning from Logical Specifications](https://arxiv.org/abs/2106.13906)", Kishor Jothimurugan, Suguman Bansal, Osbert Bastani and Rajeev Alur. In NeurIPS 2021.

# Dependencies

Requires Python version 3.7 or higher. Commands to install all dependencies:

```
pip install numpy
pip install gym
pip install torch torchvision
pip install matplotlib
pip install opencv-python
pip install mujoco-py==2.0.2.8
```

Mujoco-py requires mujoco physics simulator which can be installed following the instructions [here](https://github.com/openai/mujoco-py).

# Running the code

To run DiRL on 9Rooms,

```
python -m spectrl.examples.9rooms_dirl -e 2 -n {run_number} -d {directory} -s {spec_number}
```

Results are stored in the `{directory}` provided and `{run_number}` is an integer used to distinguish different runs for the same specification. `{spec_number}` is either 3, 4, 5, 6 or 7 corresponding to specs 1 to 5 in the paper.

Similarly, for 16Rooms (with all doors open), the command is

```
python -m spectrl.examples.16rooms_dirl -e 3 -n {run_number} -d {directory} -s {spec_number}
```

`{spec_number}` is either 9, 10, 11, 12 or 13 corresponding to specs 1 to 5 in the paper. To train on the 16Rooms environment with some blocked doors, use `-e 4` instead of `-e 3`.

For the fetch environment, the command is

```
python -m spectrl.examples.fpp_dirl -n {run_number} -d {directory} -s {spec_number}
```

Here `{spec_number}` is either 3 (PickAndPlace), 4 (PickAndPlaceStatic) or 5 (PickAndPlaceChoice). Use the option `-g` to run on GPU if available.

Replace `_dirl` in the above commands with (i) `_spectrl` to run spectrl baseline or (ii) `_tltl` to run tltl baseline.
