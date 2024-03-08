# Fair Supervised Learning with A Simple Random Sampler of Sensitive Attributes

This work is accepted by __AISTATS 2024__ (https://arxiv.org/abs/2311.05866). 

__Abstract__: As the data-driven decision process becomes dominating for industrial applications, fairness-aware machine learning arouses great attention in various areas. This work proposes fairness penalties learned by neural networks with a simple random sampler of sensitive attributes for non-discriminatory supervised learning. In contrast to many existing works that critically rely on the discreteness of sensitive attributes and response variables, the proposed penalty is able to handle versatile formats of the sensitive attributes, so it is more extensively applicable in practice than many existing algorithms. This penalty enables us to build a computationally efficient group-level in-processing fairness-aware training framework. Empirical evidence shows that our framework enjoys better utility and fairness measures on popular benchmark data sets than competing methods. We also theoretically characterize estimation errors and loss of utility of the proposed neural-penalized risk minimization problem.

__Authors__: Jinwon Sohn, Qifan Song, and Guang Lin. 

### Description
- 'datasets' contains the real data sets used in the simulation section.
- 'models' includes all competing models.
- 'utils' includes functions to import the data sets and to calculate evaluation metrics. 


### Requirement
- Python 3.8
- Tensorflow 2.4.0

### Implementation 

Here is an example to implement the main script; running our model (fairSBP) for separation in Scenario II on Adult data where the trade-off parameter is 0.9. Note the 'datasets' and 'utils' directories should be appropriately placed. 

```console
my@com:~ $ python run_script.py --data-name adult --fair-type eo --scenario-type 2 --method fairSBP --plambda 0.9
```

To implement other models, refer to the scripts in 'models'. The figures and tables in the manuscript are based on the calculated scores. To reproduce the results, one may need to find the Pareto frontiers for the produced metrics.
