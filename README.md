# Quantum Bayesian in circuit QED with moderate bandwidth

This python code calculates the evolution of a qubit that is being measured in a cQED setup with moderate readout resonator bandwidth. The theoretical work that this code uses is published in [Phys. Rev. A **94**, 042326 (2016)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.94.042326) (also here: [arXiv:1606.07162](https://arxiv.org/abs/1606.07162)).

Please refer to `examples.ipynb` jupyter notebook to see how the code can be used.

## Few notes

* This code assumes a cQED setup where the measurement tone reflects from the readout resonator (and not transmitted).
* This code is written for a phase sensitive measurement
* Code uses `python3.5.2`
* Was written with `numpy 1.12.1`. Other standard libraries used are `multiprocessing`, `math`, and `cmath`.
