# regimelab-notebooks

Exploratory data analysis (EDA) notebooks for the substack, and a place to grapple with some concepts in applied math and statistics, mathematical finance, and Bayesian machine learning. My belief is that this type of research has a lot to offer not just in the financial domain, but in codifying frameworks and best practices for working with non-stationary time series problems. 

What are regimes?
https://regimelab.substack.com/p/what-are-regimes

Areas of Research
-----------------

(1) Non-parametric models offer the flexibility to learn and generate data distributions that have not occurred yet, but may occur. They can learn new distributions on the fly (now-casting) while simultaneously providing a quantity of uncertainty and entropy in the Bayesian sense. 

(2) Detecting regime shifts, including concept drift, dataset shift, distributional shift, higher moments and so on, is a worthy enterprise for the purpose of monitoring stability in production ML/data pipelines. Additionally, regime shifts are a feature of many natural and physical phenomena worth modeling in themselves.  
https://en.wikipedia.org/wiki/Regime_shift

Additional Reference: https://www.beringclimate.noaa.gov/regimes/rodionov_overview.pdf

(3) This all becomes useful in dealing with complex, risky phenomena that face us in the real world across many domains: the climate, supply chains, weather patterns, financial systems, economies. 

(4) Sequential decision problems with uncertain outcomes and rewards don't require 'prediction' in the strictest sense, but decisions can still be optimized. This is central to successful business and operational planning, and risk management.

(5) Causal theories involving data must be falsifiable and interpretable in the sense of Popper falsifiability or a Humean induction and the scientific method. 

(6) I study Long memory/auto-correlation in data generating processes, stationarity, and ergodicity assumptions. E.g. fractional Gaussian Noise & fractional Brownian Motion. 

<br/>

$$
\huge cov(s,t) = \frac{1}{2} \left( t^{2H} + s^{2H} - |t-s|^{2H} \right)
$$

References & Inspiration
------------------------

Mostly just interesting papers or books. 

<b> Climate & Ecosystem Regimes </b>

I. [Climate regime shift detection with a trans-dimensional, sequential Monte Carlo, variational Bayes method](https://onlinelibrary.wiley.com/doi/abs/10.1111/anzs.12265)
Clare A. McGrory, Daniel C. Ahfock, Ricardo T. Lemos, Australia & New Zealand Journal of Statistics

II. [The Theory of Parallel Climate Realizations](https://link.springer.com/article/10.1007/s10955-019-02445-7)
Journal of Statistical Physics

III. [Irreversibility of regime shifts in the North Sea](https://www.frontiersin.org/articles/10.3389/fmars.2022.945204/full)
Frontiers in Marine Science

IV. [Variational Bayes Estimation of Hidden Markov Models for Daily Precipitation with Semi-Continuous Emissions](http://hpcf-files.umbc.edu/research/papers/MajumderHPCF20218.pdf)
Department of Mathematics and Statistics, Joint Center for Earth Systems Technology, University of Maryland

V. [A Bayesian Deep Learning Approach to Near-Term Climate Prediction](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022MS003058?af=R)
Xihaier Luo, Balasubramanya T. Nadiga, Ji Hwan Park, Yihui Ren, Wei Xu, Shinjae Yoo, Journal of Advances in Modeling Earth Systems

<b> The Hurst Effect </b> <br/>

I. [Basic properties of the Multivariate Fractional Brownian Motion](https://hal.science/hal-00497639/document)
Pierre-Olivier Amblard, Jean-François Coeurjolly, Frédéric Lavancier, Anne Philippe. Basic properties
of the Multivariate Fractional Brownian Motion. Séminaires et congrès, 2013, 28, pp.65-87. ffhal00497639v2f

II. [Learning Fractional White Noises in Neural Stochastic Differential Equations](https://openreview.net/pdf?id=lTZBRxm2q5)
Anh Tong, Thanh Nguyen-Tang (Johns Hopkins University), Toan Tran (VinAI Research, Vietnam), Jaesik Choi

III. [A Dynamical Systems Explanation of the Hurst Effect and Atmospheric Low-Frequency Variability](https://www.nature.com/articles/srep09068)
Christian L. E. Franzke, Scott M. Osprey, Paolo Davini & Nicholas W. Watkins 

IV. [Long memory and regime switching](https://www.nber.org/papers/t0264)
Francis X. Diebold, Atsushi Inoue

V. [On the continuing relevance of Mandelbrot’s non-ergodic fractional renewal models of 1963 to 1967](https://link.springer.com/content/pdf/10.1140/epjb/e2017-80357-3.pdf) 
Nicholas W. Watkins, Centre for the Analysis of Time Series, London School of Economics and Political Science, London, UK, Centre for Fusion, Space and Astrophysics, University of Warwick, Coventry, UK, Faculty of Science, Technology, Engineering and Mathematics, Open University, Milton Keynes, UK

VI. [The Zumbach effect under rough Heston](https://arxiv.org/pdf/1809.02098.pdf)
Radoˇs Radoiˇci´c,, Mathieu Rosenbaum, Omar El Euch, Jim Gatheral, Baruch College, CUNY, Ecole Polytechnique

<b> Mathematical Finance </b> <br/>

I. [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
Marcos Lopez de Prado 

II. [Tactical Investment Algorithms](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3459866)
Marcos Lopez de Prado

III. [Statistical Arbitrage in the U.S. Equities Market](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf)
Marco Avellaneda & Jeong-Hyun Lee

IV. [Principal Eigenportfolios for U.S Equities](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3738769)
Marco Avallaneda, Brian Healy, Andrew Papanicolaou, George Papanicolaou

V. [Managing Risks in a Risk-On/Risk-Off Environment](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2150877)
Marcos Lopez de Prado, Lawrence Berkeley National Laboratory

VI. [Market Regime Detection via Realized Covariances](https://arxiv.org/pdf/2104.03667.pdf)
Andrea Bucci, Vito Ciciretti, Department of Economics, Universita degli Studi ”G. d’Annunzio” Chieti-Pescara, Independent Researcher

VII. [Can Factor Investing Become Scientific?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4205613)
Marcos Lopez de Prado 

VIII. [Rational Expectations Econometric Analysis Of Changes in Regime](https://www.bu.edu/econ/files/2014/01/Hamilton-Interest-Rates.pdf)
James D. Hamilton

<b> Mixture Models, Gaussian Processes, HMMs </b> <br/> 

I. [EM Algorithm - CS229 Lecture](http://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf)
Tengyu Ma and Andrew Ng

II. [Infinite Mixture of Global Gaussian Processes](https://melaniefp.github.io/contents/papers/DDP_regression_paper_BNPNG.pdf)
Fernando Perez-Cruz, Melanie Pradier

III. [Dirichlet Process](https://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/dp.pdf)
Yee Whye Teh, University College London

IV. [Variational Inference](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf)
David M. Blei

V. [Graphical Models, Exponential Families, and Variational Inference](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf)
Martin J. Wainwright, Michael I. Jordan, University of California, Berkeley

VI. [The Algebra of Probable Inference](https://bayes.wustl.edu/Manual/cox-algebra.pdf)
Richard T. Cox

VII. [The Elements of Statistical Learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576)
Trevor Hastie, Robert Tibshirani, Jerome Friedman

VIII. [A New Approach to Data Driven Clustering](https://mlg.eng.cam.ac.uk/zoubin/papers/AzrGhaICML06.pdf)
Arik Azran, Gatsby Computational Neuroscience Unit, University College London, Zoubin Ghahramani, Department of Engineering, University of Cambridge, Cambridge

IX. [Particle Learning for Bayesian Non-Parametric Markov Switching Stochastic Volatility Model](https://hedibert.org/wp-content/uploads/2016/07/2016-virbickaite-lopes-ausin-galeano.pdf)
Bayesian Analysis

X: [A Guide To Monte Carlo Simulations In Statistical Physics](https://el.us.edu.pl/ekonofizyka/images/6/6b/A_guide_to_monte_carlo_simulations_in_statistical_physics.pdf)
David P. Landau, Kurt Binder 

<b> The Ergodicity Problem </b> <br/> 

I. [The ergodicity problem in economics](https://www.nature.com/articles/s41567-019-0732-0)
Ole Peters, Nature Physics

II. [Time to move beyond average thinking](https://www.nature.com/articles/s41567-019-0758-3)
Nature Physics

III. [Ergodicity-breaking reveals time optimal decision making in humans](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009217)
David Meder, Finn Rabe, Tobias Morville, Kristoffer H. Madsen, Magnus T. Koudahl, Ray J. Dolan, Hartwig R. Siebner, Oliver J. Hulme 

IV. [Autocorrelation functions and ergodicity in diffusion with stochastic resetting](https://arxiv.org/abs/2107.11686)
Viktor Stojkoski, Trifce Sandev, Ljupco Kocarev, Arnab Pal

V. [Self-fulfilling Prophecies, Quasi Non-Ergodicity & Wealth Inequality](https://www.nber.org/system/files/working_papers/w28261/w28261.pdf)
NBER Working Paper

VI. [A misconception in ergodicity: Identify ergodic regime not ergodic process](http://science-memo.blogspot.com/2022/05/ergodic-regime-not-process.html)
Mehmet Süzen

VII. [Effective ergodicity in single-spin-flip dynamics](https://www.researchgate.net/publication/266945166_Effective_ergodicity_in_single-spin-flip_dynamics/link/5783964408ae3f355b4a1a02/download)
Mehmet Süzen

VIII. [Wealth Inequality and the Ergodic Hypothesis: Evidence from the United States](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2794830)
Yonatan Berman, London Mathematical Laboratory, Ole Peters, London Mathematical Laboratory; Santa Fe Institute, Alexander Adamou, London Mathematical Laboratory

IX. [Non-ergodic extended regime in random matrix ensembles: insights from eigenvalue spectra](https://www.nature.com/articles/s41598-023-27751-9) Scientific Reports, Nature, Wang‐Fang Xu, W. J. Rao


