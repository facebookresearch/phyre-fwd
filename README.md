[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)
# PHYRE Forward Agents

Code for reproducing [Forward Prediction for Physical Reasoning](https://arxiv.org/abs/2006.10734).


# Getting started

## Installation
A [Conda](https://docs.conda.io/en/latest/) virtual enviroment is provided contianing all necessary dependencies.
```(bash)
git clone https://github.com/facebookresearch/phyre-fwd.git
cd phyre-fwd
conda env create -f env.yml && conda activate phyre-fwd
pushd src/python && pip install -e . && popd
pip install -r requirements.agents.txt
```
Make a directory somewhere with enough space for trained models, and symlink it to `agents/outputs`.

## Methods

For training the methods, and available pre-trained models, see [agents](agents/).

# License
PHYRE forward agents are released under the Apache license. See [LICENSE](LICENSE) for additional details.


# Citation

If you use `phyre-fwd` or the baseline results, please cite it:

```bibtex
@article{girdhar2020forward,
    title={Forward Prediction for Physical Reasoning},
    author={Rohit Girdhar and Laura Gustafson and Aaron Adcock and Laurens van der Maaten},
    year={2020},
    journal={arXiv:2006.10734}
}
```
