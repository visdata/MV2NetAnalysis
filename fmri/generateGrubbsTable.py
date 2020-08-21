import numpy as np
from scipy import stats
from math import sqrt

# two side
def get_t_significance_level(alpha, n):
    return alpha / (2*n)
    # return alpha / n

def get_g_test(n, alpha):
        """Compute a significant value score following these steps, being alpha
        the requested significance level:
        1. Find the upper critical value of the t-distribution with n-2
           degrees of freedom and a significance level of alpha/2n
           (for two-sided tests) or alpha/n (for one-sided tests).
        2. Use this t value to find the score with the following formula:
           ((n-1) / sqrt(n)) * (sqrt(t**2 / (n-2 + t**2)))
        :param n: length of data set
        :param float alpha: significance level
        :return: G_test score
        """
        significance_level = get_t_significance_level(alpha, n)
        t = stats.t.isf(significance_level, n-2)
        # print(t)
        return ((n-1) / sqrt(n)) * (sqrt(t**2 / (n-2 + t**2)))

if __name__ == '__main__':  
    # alpha = np.array([0.0001,0.00025,0.0005,0.00075,0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1])
    # alpha = np.array([0.00000001,0.00000025,0.0000005,0.000001,0.000025,0.00005,0.0001,0.00025,0.0005,0.00075,0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1])
    alpha = np.array([0.0001,0.00025,0.0005,0.00075,0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,0.9,0.95,0.975,0.99,0.995,1])
    alphaline = [str(a) for a in alpha]
    with open('grubbsTable.txt', 'w') as ifile:
        ifile.write(' '.join(alphaline)+'\n')
        for n in range(3, 200):
            grubbsTable = []
            for a in alpha:
                grubbsTable.append(str(get_g_test(n, a)))
            ifile.write(' '.join(grubbsTable)+'\n')
    
        
    