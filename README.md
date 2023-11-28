# Combinatorial Optimization via Memory Metropolis: Template Networks for Proposal Distributions in Simulated Annealing applied to Nanophotonic Inverse Design

Code for the paper [*Combinatorial Optimization via Memory Metropolis: Template Networks for Proposal Distributions in Simulated Annealing applied to Nanophotonic Inverse Design*](https://openreview.net/forum?id=Eu2k9La3RB) at AI4MAT workshop at NeurIPS 2023

## Installation

Install requirements using pip `pip -r requirements.txt`.
The simulation backend `simFrame` will be released soon. For now a dummy backend class is implemented instead.

## Execution

``` bash
python main.py --config configurations/algorithms/parallelMeme.yml --device_config configurations/devices/mdm.yml
```
