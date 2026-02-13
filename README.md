## MLMI: Potential Energy Prediction for Cu–Zr

R. Chinna Rayudu 22CSB0A45(CSE)



This repository contains Molecular Dynamics (LAMMPS) simulation files and Machine Learning models (Linear Regression and Neural Network) for predicting the **potential energy (PE)** of Cu–Zr alloy systems.

---

##  Project Structure

```

CuZr/
├─ Cu-Zr_4.eam              # EAM potential file used by LAMMPS
├─ data5                    # MD output file (CSV format) used directly by ML models
├─ in.cuzr                  # LAMMPS input script for running MD at various temperatures
├─ Linear_Regression.py     # Linear Regression model implementation (runs directly on data5)
├─ log.lammps               # LAMMPS log file
├─ Neural_Network.py        # Neural Network model implementation (runs directly on data5)
└─ README.md                # This document

```
---

##  Overview

This project demonstrates:

* Running Molecular Dynamics simulations using **LAMMPS**
* Extracting thermodynamic properties of a Cu–Zr alloy
* Training ML models to predict **Potential Energy (PE)** from MD outputs using features such as:

  * Temperature (`temp`)
  * Pressure (`press`)
  * Volume (`vol`)


##  Expected Format of `data5`

data5 must be a CSV file with a header row. At minimum it should contain:

step,temp,press,vol,pe,ke,etotal

* Inputs: temp, press, vol
* Target:  pe

If your CSV uses different column names, either rename them or edit the scripts to match.

---

##  MD Simulation (Run this FIRST)

Before running ML scripts, run the MD simulation to generate `log.lammps` and `data5`.

1. Open a terminal and change to the project directory:

cd CuZr

2. Run LAMMPS (serial):
lmp_serial -in in.cuzr

Or, to run with MPI:

mpirun -np 4 lmp_mpi -in in.cuzr

###  Files produced by the MD run

* log.lammps — LAMMPS thermodynamic output (contains temp, press, vol, pe, ke, etotal if thermo_style is set accordingly)
* data5 — CSV file used by ML scripts (ensure columns match the expected format)

> If data5 is not created automatically by your LAMMPS input, extract the thermo block from log.lammps and save it to data5.

##  Running the ML models

> Make sure data5 exists and has the correct columns before running any ML script.

### 1) Linear Regression

python Linear_Regression.py

What it does (expected behavior):

* Reads data5
* Selects features (temp, press, vol) and target (pe)
* Trains a sklearn.linear_model.LinearRegression model on the entire dataset
* Prints metrics and/or sample predictions
* Saves model artifacts if implemented in the script

### 2) Neural Network

python Neural_Network.py

What it does (expected behavior):

* Reads data5
* Standardizes inputs (if implemented)
* Builds and trains a small feedforward MLP to predict pe
* Prints training progress and final evaluation (if test split provided)
* Saves the trained model if implemented


##  References

* LAMMPS documentation: [https://lammps.sandia.gov/](https://lammps.sandia.gov/)
* scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
* TensorFlow / PyTorch documentation if using NN frameworks
