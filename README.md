# Tensor-Based Modulation

Tensor-Based Modulation (TBM) is a modulation designed to handle massive over-the-air contention in multiple antenna wireless systems. As opposed to classical methods based on handling collisions through transmission redundancy, TBM relies on multi-linear spreading to enable the parallel decoding of most of the colliding signals, up to a high degree of contention. The method was introduced in [1].

This is an implementation of TBM in Python, for the case of a block-fading channel with multiple receiving antennas.

It also includes an implementation of the structured vector modulation adapted to non-coherent communications described in [2].


## Getting started

### Installation

Install the prerequisite libraries ([Numpy](https://numpy.org/), [Tensorly](http://tensorly.org/), [Statistics](https://docs.python.org/3/library/statistics.html) and [Graycode](https://gitlab.com/heikkiorsila/gray-code)) and clone this repository:
```
pip install numpy tensorly statistics graycode
git clone https://gitlab.inria.fr/maxguill/tensor-based-modulation.git
```

### Running the example

```
cd tensor-based-modulation
python3 ./tbm_test.py
```


## Implementation status

The present implementation is not optimized for performance (speed and/or other efficiency metrics). It intends to be didactical by adhering to the concepts and notations used in [1] and [2] (annotations in the code refer to equations in these articles) and maximally reusing off-the-shelf components (in particular, all tensor algebraic operations are performed using [Tensorly](http://tensorly.org/)).

### Features (implemented and to-do)
- [x] Tensor-based modulation from [1] over a block-fading multiuser Single-Input Multiple-Output channel
- [x] Vector codebook from [2] (including mapper and hard demapper)
- [ ] Vector codebook based on reference symbol+QAM modulation, and ZF equalization
- [ ] Binary channel code
- [ ] Receiver-side estimation of the number of active users (currently assumed known)
- [ ] Performance benchmark



## Author
The code was written by [Maxime Guillaud](http://research.mguillaud.net/). The theory behind tensor-based modulation, published in [1] and [2], was developed in collaboration with Alexis Decurninge, [Khac-Hoang Ngo](https://khachoang1412.github.io/), Ingmar Land and [Sheng Yang](https://l2s.centralesupelec.fr/en/u/yang-sheng/).


## License

This software is distributed under the [3-Clause BSD](https://opensource.org/license/bsd-3-clause/) license agreement.


## Bibliography

[1] [Tensor-Based Modulation for Unsourced Massive Random Access](https://dx.doi.org/10.1109/LWC.2020.3037523), by A. Decurninge, I. Land and M. Guillaud,  IEEE Wireless Communications Letters, vol. 10, no. 3, pp. 552-556, March 2021.

[2] [Cube-Split: A Structured Grassmannian Constellation for Non-Coherent SIMO Communications](https://doi.org/10.1109/TWC.2019.2959781), by K.-H. Ngo, A. Decurninge, M. Guillaud, S. Yang, IEEE Transactions on Wireless Communications, Vol. 19, No. 3, March 2020.
