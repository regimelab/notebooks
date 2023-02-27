# regimelab-notebooks

Explanatory notebooks for the substack, and a place to grapple with some concepts in applied math and statistics, mathematical finance, and Bayesian machine learning. My belief is that this type of research has a lot to offer not just in the financial domain, but in codifying frameworks and best practices for working with non-stationary time series problems. 

What are regimes?
https://regimelab.substack.com/p/what-are-regimes

Areas of Research
-----------------

(1) Non-parametric models offer the flexibility to learn and generate data distributions that have not occurred yet, but may occur. They can learn new distributions on the fly (nowcasting). 

(2) Detecting regime shifts (concept drift, dataset shifts, distributional shifts, higher moments and so on) is a worthy enterprise for the purposes of having stable algorithms in production. Additionally, regime shifts are a feature of many natural/physical phenomena worth modeling in themselves.  
https://en.wikipedia.org/wiki/Regime_shift

(3) This all becomes useful in dealing with complex, risky phenomena that face us in the real world across many domains: the climate, supply chains, weather patterns, financial systems, economies. 

(4) Sequential decision problems with uncertain outcomes and rewards don't require 'prediction' in the strictest sense, but decisions can still be optimized. This is central to successful business and operational planning, and risk management.

(5) Causal theories involving data must be falsifiable and interpretable, in the sense of Popper falsifiability and the scientific method. 

(6) I study the importance of long memory/auto-correlation in data generating processes, stationarity, and ergodicity assumptions. 

<b> Eg. Fractional Gaussian Noise & Fractional Brownian Motion </b> <br/>

$$
\huge cov(s,t) = \frac{1}{2} \left( t^{2H} + s^{2H} - |t-s|^{2H} \right)
$$

References & Inspiration
------------------------

<b> FGn/FBm, The Hurst Effect </b> <br/>

I. [Basic properties of the Multivariate Fractional Brownian Motion](https://hal.science/hal-00497639/document)
Pierre-Olivier Amblard, Jean-François Coeurjolly, Frédéric Lavancier, Anne Philippe. Basic properties
of the Multivariate Fractional Brownian Motion. Séminaires et congrès, 2013, 28, pp.65-87. ffhal00497639v2f

II. [Learning Fractional White Noises in Neural Stochastic Differential Equations](https://openreview.net/pdf?id=lTZBRxm2q5)
Anh Tong, Thanh Nguyen-Tang (Johns Hopkins University), Toan Tran (VinAI Research, Vietnam), Jaesik Choi

III. [A Dynamical Systems Explanation of the Hurst Effect and Atmospheric Low-Frequency Variability](https://www.nature.com/articles/srep09068)
Christian L. E. Franzke, Scott M. Osprey, Paolo Davini & Nicholas W. Watkins 

IV. [Long memory and regime switching](https://www.sciencedirect.com/science/article/abs/pii/S0304407601000732?via%3Dihub)
Francis X. Diebold, Atsushi Inoue

V. [On the continuing relevance of Mandelbrot’s non-ergodic fractional renewal models of 1963 to 1967](https://link.springer.com/content/pdf/10.1140/epjb/e2017-80357-3.pdf) 
Nicholas W. Watkins, Centre for the Analysis of Time Series, London School of Economics and Political Science, London, UK, Centre for Fusion, Space and Astrophysics, University of Warwick, Coventry, UK, Faculty of Science, Technology, Engineering and Mathematics, Open University, Milton Keynes, UK

<b> ML + Quant Finance </b> <br/>

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

IV. [Variational Bayes Estimation of Hidden Markov Models for Daily Precipitation with Semi-Continuous Emissions](http://hpcf-files.umbc.edu/research/papers/MajumderHPCF20218.pdf)
Department of Mathematics and Statistics, Joint Center for Earth Systems Technology, University of Maryland

VI. [A Bayesian Deep Learning Approach to Near-Term Climate Prediction](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022MS003058?af=R)
Xihaier Luo, Balasubramanya T. Nadiga, Ji Hwan Park, Yihui Ren, Wei Xu, Shinjae Yoo, Journal of Advances in Modeling Earth Systems

VII. [Variational Inference](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf)
David M. Blei

VIII. [Graphical Models, Exponential Families, and Variational Inference](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf)
Martin J. Wainwright, Michael I. Jordan, University of California, Berkeley

IX. [Climate regime shift detection with a trans-dimensional, sequential Monte Carlo, variational Bayes method](https://onlinelibrary.wiley.com/doi/abs/10.1111/anzs.12265)
Clare A. McGrory, Daniel C. Ahfock, Ricardo T. Lemos, Australia & New Zealand Journal of Statistics

X. [The Algebra of Probable Inference](https://bayes.wustl.edu/Manual/cox-algebra.pdf)
Richard T. Cox

XI. [The Elements of Statistical Learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576)
Trevor Hastie, Robert Tibshirani, Jerome Friedman

XII. [A New Approach to Data Driven Clustering](https://mlg.eng.cam.ac.uk/zoubin/papers/AzrGhaICML06.pdf)
Arik Azran, Gatsby Computational Neuroscience Unit, University College London, Zoubin Ghahramani, Department of Engineering, University of Cambridge, Cambridge

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

VI. [Ergodic Markov Chains](https://gregorygundersen.com/blog/2019/10/28/ergodic-markov-chains/)
Gregory Gunderson
