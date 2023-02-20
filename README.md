# regimelab-notebooks

This is an area for exploratory data analysis mainly, and grappling with some concepts in applied math and statistics, mathematical finance and Bayesian machine learning. My belief is that this type of research has a lot to offer not just in the financial domain, but in codifying frameworks and best practices for working with non-stationary, stochastic time series problems. 

(1) Non-parametric models offer the flexibility to learn and generate (describe) data distributions that have not occurred yet, but may occur. They can learn new distributions on the fly (now-casting). 

(2) Detecting regime shifts (concept drift, dataset shifts, distributional shifts, higher moments and so on) is a worthy enterprise for the purpose of having stable algorithms in production, and regime shifts are a feature of many natural/physical phenomena worth modeling in themselves.

(3) This all becomes useful in dealing with complex, risky phenomena that face us in the real world across many domains: the climate, supply chains, weather patterns, financial systems, economies. 

(4) Sequential decision problems with uncertain outcomes and rewards don't require 'prediction' in the strictest sense, but decisions can still be optimized. This is central to successful business and operational planning, and risk management.

(5) Theories involving time series data should be falsifiable and interpretable. 

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
Anh Tong, Thanh Nguyen-Tang (Johns Hopkins University), Toan Tran (VinAI Research, Vietnam), Jaesik Choi

III. [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
Marcos Lopez de Prado 

IV. [Tactical Investment Algorithms](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3459866)
Marcos Lopez de Prado

V. [Statistical Arbitrage in the U.S. Equities Market](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf)
Marco Avellaneda & Jeong-Hyun Lee

VI. [Principal Eigenportfolios for U.S Equities](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3738769)
Marco Avallaneda, Brian Healy, Andrew Papanicolaou, George Papanicolaou

VII. [Infinite Mixture of Global Gaussian Processes](https://www.bell-labs.com/institute/publications/itd-15-55873g/#gref)
Fernando Perez-Cruz, Melanie Pradier

VIII. [Dirichlet Process](https://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/dp.pdf)
Yee Whye Teh, University College London

IX. [Managing Risks in a Risk-On/Risk-Off Environment](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2150877)
Marcos Lopez de Prado, Lawrence Berkeley National Laboratory

X. [CS229 Lecture notes](http://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf)
Tengyu Ma and Andrew Ng

XI. [Market Regime Detection via Realized Covariances](https://arxiv.org/pdf/2104.03667.pdf)
Andrea Bucci, Vito Ciciretti, Department of Economics, Universita degli Studi ”G. d’Annunzio” Chieti-Pescara, Independent Researcher

XII. [Rational Expectations Econometric Analysis Of Changes in Regime](https://www.bu.edu/econ/files/2014/01/Hamilton-Interest-Rates.pdf)
James D. Hamilton

XIII. [The Algebra of Probable Inference](https://bayes.wustl.edu/Manual/cox-algebra.pdf)
Richard T. Cox

XIV. [The Elements of Statistical Learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576)
Trevor Hastie, Robert Tibshirani, Jerome Friedman

XV. [Variational Bayes Estimation of Hidden Markov Models for Daily Precipitation with Semi-Continuous Emissions](http://hpcf-files.umbc.edu/research/papers/MajumderHPCF20218.pdf)
Department of Mathematics and Statistics, Joint Center for Earth Systems Technology, University of Maryland
