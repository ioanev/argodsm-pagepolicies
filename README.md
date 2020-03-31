# ArgoDSM

[ArgoDSM](https://www.it.uu.se/research/project/argo) is a software distributed
shared memory system which aims to provide great performance with a simplified
programming model. It is currently being developed in Uppsala University.

For more information please visit our [website](https://www.argodsm.com).
There is also a [quickstart guide](https://etascale.github.io/argodsm/) and a
[tutorial](https://etascale.github.io/argodsm/tutorial.html).

Please contact us at [contact@argodsm.com](mailto:contact@argodsm.com)

# PagePolicies

This branch consists of a stable version of ArgoDSM with the following incorporated
page-based memory allocation policies (implementation is under the data_distribution.cpp and data_distribution.hpp files):
- 0 : Bind-All
- 1 : Cyclic
- 2 : Cyclic-Block
- 3 : Skew-Mapp
- 4 : Skew-Mapp-Block
- 5 : Prime-Mapp
- 6 : Prime-Mapp-Block
- 7 : First-Touch (buggy)
