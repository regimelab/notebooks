# regimelab-notebooks

This is an area for exploratory data analysis mainly, and grappling with some concepts in applied math and statistics, mathematical finance and Bayesian machine learning. My belief is that this type of research has a lot to offer not just in the financial domain, but in codifying frameworks and best practices for working with non-stationary, stochastic time series problems. 

And importantly, having <b>falsifiable</b> theories that can be tested systematically. This is what these notebooks are about. 

(1) Non-parametric models offer flexibility to learn and generate (describe) data distributions that have not occurred yet, but may occur.

(2) Detecting regime shifts (concept shifts, data shifts, distribution shifts, higher moments, kurtosis, and so on) is a worthy enterprise for the purposes of having stable ML algorithms in production, as well as for modeling the most interesting problems themselves. 

(3) This all becomes useful in dealing with complex, risky phenomena that face us in the real world across many domains: the climate, supply chains, weather patterns, biodiversity of eco-systems, medical systems and medicine, financial systems, economies, and so on. 

(4) Sequential decision problems with uncertain outcomes and rewards don't require 'prediction' in the strictest sense, but decisions can still be optimized. This is central to successful business and operational planning. 

What are regimes?
https://regimelab.substack.com/p/what-are-regimes

$$
\huge p({\theta}|x) = \sum_{i=1}^n \mathcal{\omega_i} \mathcal{N}(\mu_i, \sigma^2_i)
$$

References & Inspiration
------------------------

I. [Basic properties of the Multivariate Fractional Brownian Motion](https://hal.science/hal-00497639/document)
Pierre-Olivier Amblard, Jean-François Coeurjolly, Frédéric Lavancier, Anne Philippe. Basic properties
of the Multivariate Fractional Brownian Motion. Séminaires et congrès, 2013, 28, pp.65-87. ffhal00497639v2f

II. [Learning Fractional White Noises in Neural Stochastic Differential Equations](https://openreview.net/pdf?id=lTZBRxm2q5)
Anh Tong KAIST
Thanh Nguyen-Tang (Johns Hopkins University)
Toan Tran (VinAI Research, Vietnam)
Jaesik Choi KAIST, INEEJI

III. [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
Marcos Lopez de Prado 

IV. [Tactical Investment Algorithms](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3459866)
Marcos Lopez de Prado

IV. [Statistical Arbitrage in the U.S. Equities Market (2008)](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf)
Marco Avellaneda & Jeong-Hyun Lee

V. [Principal Eigenportfolios for U.S Equities (2020)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3738769)
Marco Avallaneda, Brian Healy, Andrew Papanicolaou, George Papanicolaou

VI. [Infinite Mixture of Global Gaussian Processes](https://www.bell-labs.com/institute/publications/itd-15-55873g/#gref)
Fernando Perez-Cruz, Melanie Pradier

VII. [Dirichlet Process](https://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/dp.pdf)
Yee Whye Teh, University College London

VIII. [Managing Risks in a Risk-On/Risk-Off Environment](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2150877)
Marcos Lopez de Prado, Lawrence Berkeley National Laboratory

IX. [CS229 Lecture notes](http://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf)
Tengyu Ma and Andrew Ng
