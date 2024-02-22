import scipy.stats as stats

rolling_zscore = lambda serie, wlen: [ stats.zscore(serie[x - wlen : x]).values[-1] for x in range(wlen, len(serie) + 1) ]
