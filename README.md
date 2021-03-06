![CDT_AAE](https://danielegrattarola.github.io/images/2018-06-07/scheme.png)

This is the official implementation of the paper "Change Detection in Graph Streams by Learning Graph Embeddings on Constant-Curvature Manifolds" by D. Grattarola, D. Zambon, C. Alippi, and L. Livi. (2018, [https://arxiv.org/abs/1805.06299](https://arxiv.org/abs/1805.06299)).  

This code provides a proof of concept for the experiments conducted in the paper, and contains all the necessary elements to apply the methodology on a generic problem.  

Please cite the paper if you use any of this code for your research:   

```
@article{grattarola2019change,
  title={Change detection in graph streams by learning graph embeddings on constant-curvature manifolds},
  author={Grattarola, Daniele and Zambon, Daniele and Livi, Lorenzo and Alippi, Cesare},
  journal={IEEE transactions on neural networks and learning systems},
  year={2019},
  publisher={IEEE}
}
```

## Setting up

The code is implemented for Python 3 and tested on Ubuntu 16.04.  
To run the code, you will need a number of libraries installed on your system:

- [Keras](https://keras.io/) (`pip install keras`), a high-level API for deep learning;
- [Spektral](https://danielegrattarola.github.io/spektral/) (`pip install spektral`), a Keras extension to build graph neural networks;
- [CDG](https://github.com/dan-zam/cdg) (see README on Github), a library for change detection tests and non-Euclidean geometry. 

The code also depends on Numpy, Scipy, Pandas, Scikit-learn, and Joblib.  

## Running experiments

The `src` folder includes several scripts to run the different versions of the algorithm proposed in the paper:

- `main_prior.py`, trains the AAE using the non-Euclidean prior;
- `main_geom.py`, trains the AAE using the geometric discriminator;
- `main_cdt.py`, runs the change detection tests on the data saved by the previous scripts;

To simplify the workflow, the repository includes a `main.sh` script to run a full experiment on Delaunay triangulations.

## About

For a simplified explanation of the paper, check out [this blog post](https://danielegrattarola.github.io/posts/2018-06-07/ccm-paper.html).
