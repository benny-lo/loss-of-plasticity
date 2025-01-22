# Loss of Plasticity and Winning Lottery Tickets

## Overview
This repository contains the code and experiments for investigating the relationship between *Winning Lottery Tickets* (from the Lottery Ticket Hypothesis) and *Loss of Plasticity (LoP)* in neural networks. The project explores whether pruned subnetworks (winning tickets) retain their plasticity or suffer from LoP, particularly under distribution shifts.

Our findings aim to provide insights into:
- The correlation between the number of winning tickets and network plasticity.
- Whether winning tickets are more or less resistant to LoP.
- The impact of *Iterative Magnitude Pruning (IMP)* on plasticity.

This research was conducted as part of the ETH Zürich *Deep Learning 2024-2025* course.



## Repository Structure
```
├── README.md
├── requirements.txt
├── src
│   ├── datasets
│   │   ├── _init_.py
│   │   └── mnist.py
│   ├── plotting
│   │   ├── mask_accuracy_plot.py
│   │   ├── plot_config.yaml
│   │   ├── plot_util.py
│   │   ├── plotting_average_overlap_all.py
│   │   ├── plotting_overlap_parameters.py
│   │   └── plotting_winning_ticket_plasticity.py
│   └── wip
│       ├── empiricalNTK.py
│       ├── lop_on_tickets.py
│       ├── config.yaml
│       ├── experiments.py
│       ├── lottery_tickets.py
│       ├── models.py
│       ├── slurm_run.sh
│       ├── training.py
│       └── utils.py
├── .gitignore
```

## Installation & Setup

bash
git clone https://github.com/benny-lo/loss-of-plasticity.git
cd loss-of-plasticity
pip install -r requirements.txt


we use python=3.12.7

## Key Experiments

### Number of Winning Tickets vs. Plasticity  
- Measures the correlation between the number of winning ticket masks and the plasticity of the network.

### Loss of Plasticity in Winning Tickets  
- Evaluates how ticket accuracy changes under sequential task learning.

### Plastic Parameters and Winning Tickets  
- Compares plastic parameters with the identified winning tickets.

All the experiments can be run with 
bash
python src/experiments.py name_experiment


The experiments should be run sequentially, since the first produces the winning tickets and the data needed to compute the other two.
