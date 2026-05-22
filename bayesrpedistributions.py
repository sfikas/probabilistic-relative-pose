import numpy as np
import torch
import normflows as nf
from matplotlib import pyplot as plt


class TargetWPlot(nf.distributions.Target):
    def __init__(self):
        super().__init__()

    def plot(self, plot_gt=None, faux_normalize=True, savefile=None, xmin=-np.pi/2, xmax=+np.pi/2, grid_size = 60, plotsize = (9,6)):
        '''
        Create a 1D plot of the distribution
        '''
        #xx, yy = torch.meshgrid(
        xx = torch.meshgrid(
            torch.linspace(xmin, xmax, grid_size)
        ); xx = xx[0]
        #torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        prob = torch.exp(self.log_prob(xx))
        prob[torch.isnan(prob)] = 0
        if faux_normalize:
            # This normalizes the pdf by normalization factor "Z",
            # which is approximated via Monte Carlo.
            invnormalization_factor = grid_size / torch.sum(prob)
            prob = prob * invnormalization_factor
        if type(plotsize) is tuple:
            plt.figure(figsize=plotsize)
        else:
            plt.figure(figsize=(plotsize, plotsize))
        #plt.figure(figsize=(5, 5))
        plt.plot(xx, prob, label='likelihood')
        if plot_gt is not None:
            gtpoint_likelihood = torch.exp(self.log_prob(torch.Tensor([plot_gt])))
            if faux_normalize:
                gtpoint_likelihood = gtpoint_likelihood * invnormalization_factor
            plt.stem(plot_gt, gtpoint_likelihood, 'ro--', label='ground truth')
        plt.legend()
        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight')
        else:
            plt.show()


class VonMises(TargetWPlot):
    def __init__(self, theta0, m):
        super().__init__()
        self.m = m
        self.theta0 = theta0
        self.log_const = -np.log(2 * np.pi) - np.log(np.i0(self.m))
        self.max_log_prob = np.log(1 + np.exp(self.log_const))

    def log_prob(self, theta):
        return(self.m * torch.cos(theta - self.theta0) + self.log_const)

    def __plot(self, faux_normalize=True):
        grid_size = 60
        xx = torch.meshgrid(
            torch.linspace(-np.pi, np.pi, grid_size)
            #torch.linspace(-np.pi/2, np.pi/2, grid_size)
        ); xx = xx[0]
        prob = torch.exp(self.log_prob(xx))
        prob[torch.isnan(prob)] = 0
        if faux_normalize:
            invnormalization_factor = grid_size / torch.sum(prob)
            prob = prob * invnormalization_factor
        plt.figure(figsize=(5, 5))
        plt.plot(xx, prob, label='likelihood')
        plt.legend()
        plt.show()



class EngelsNister(TargetWPlot):
    #TODO: The name of this class is misnomer, as it implements two possible likelihoods. Engels-Nister is one of them.
    '''
    This implements one of possible likelihood terms.
    Options are
      * Engels & Nister 2005 (equation 8).
      * Generalized Gaussian (novel work, have computed the normalizing constant for this one)


    'sampson_denominator' can be True or False.
                    If True, the Sampson distance will be used, as it appears e.g. in Engels-Nister 2005 or Zhong 2019.
                    If False, the Sampson distance will be approximated with [x' E x]^2, i.e. omitting the denominator.
    'likelihood_formula':
                    'engels-nister'
                    'generalized-gaussian'
                    'generalized-gaussian-nosigma':       Does not scale by .5σ^{-2}. Useful in the σ update.
    '''
    def __init__(self, x2_A, x2_B,
                 z = None,      # Weights in [0, 1], to be applied to each x2_A-x2_B pair. Default is to weight all by the same weight (x1)
                 k = 0.1,       # cf. Engels & Nister section 6.4. This parameter should be between 0 and 1 (they suggest k = 0.5)
                 sigma2 = .1,
                 sampson_denominator = True,
                 likelihood_formula = None,
                 sum_over_x = True,
                 n_dims = 1,    # Ideally this should be inferred when running "log_prob". n_dims = 1 -> 1DOF(azimuth), n_dims = 5 -> 5DOF(full Essential)
                 ):
        super().__init__()
        if likelihood_formula is None:
            return ValueError('You must specify likelihood expliticly.')
        self.n_dims = n_dims
        # TODO: Probably this need fixing, to check
        self.max_log_prob = 0 # This is correct, albeit for this (unnormalized) form (Note the "propto" sign in Engels & Nister, eq.8)

        enable_cuda = True
        self.device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self.x2_A = torch.from_numpy(x2_A).to(self.device)
        self.x2_B = torch.from_numpy(x2_B).to(self.device)

        d, N = x2_A.shape
        self.N = N
        self.k = k        
        self.sigma2 = torch.Tensor([sigma2]).to(self.device)
        #self.sigma2 = sigma2
        
        self.sum_over_x = sum_over_x        
        self.z = z
        self.sampson_denominator = sampson_denominator
        self.likelihood_formula = likelihood_formula
        assert(d == 3)      # Must be 2D homogeneous coordinates
        assert(N == x2_B.shape[1])

    def essential_matrix(self, angle):
        #TODO
        #   Create an essential matrix up given >1DOF.
        #
        #       p(t):   Azimuth, elevation. Make sure to take into account ambiguities (e.g., can't discern btw t and -t)
        #       p(R|t):
        #               Loosely based on Liu et al. 2023.
        #               Breaks down as p(c1,c2|t) = p(c2|c1,t)p(c1|t)
        #                   where c1: One column of Rotation matrix.
        #                       c2: Second column of Rotation matrix. (only 1DOF here!)
        #                       c3: Third column of Rotation matrix. Zero DOF left; this is deteministically c1 x c2.
        #         
        N_angles = angle.shape[0]
        angle = torch.squeeze(angle)
        x = torch.cos(angle)
        y = torch.sin(angle)
        essential_matrix = torch.zeros([9, N_angles])
        essential_matrix[2, :] = y
        essential_matrix[5, :] = -x
        essential_matrix[6, :] = -y
        essential_matrix[7, :] = x
        essential_matrix = torch.reshape(essential_matrix.T, [N_angles, 3, 3])
        essential_matrix = torch.reshape(torch.transpose(essential_matrix, 0, 1), [3, N_angles*3]).double()
        essential_matrix = essential_matrix.to(self.device)
        return(essential_matrix)
    
    def log_prob(self, angle, batchmode=None): #=10
        '''
        This will return the log-probability of likelihood p(X|angle).
        X is a set of corresponding pairs of the form: {x,x'}.
        
        Arguments:
        'batchmode' can be None, or have an integer value.
            None        Use all available corresponding pairs. This may be slower to evaluate.
            num_x       Use num_x corresponding pairs, sampled at random from X.
                        This will be aking to the difference of SGD vs GD: The objective will be different in each evaluation.
        '''
        if batchmode is not None:
            assert(batchmode <= self.N) # self.N is the total number of available pairs
            N = batchmode
            cc = torch.randperm(self.N)
            x2_A = self.x2_A[:, cc[:N]]
            x2_B = self.x2_B[:, cc[:N]]
            if self.z is not None:
                z = self.z[cc[:N]]
        else:
            x2_A = self.x2_A
            x2_B = self.x2_B
            N = self.N
            if self.z is not None:
                z = self.z

        N_angles = angle.shape[0]
        res = torch.empty_like(torch.squeeze(angle))
        #res = torch.empty_like([N_angles])
        
        
        # We need to compute a number of terms of the form: (x^T E x')^2. 
        # If we have a single "E" matrix (corresponding to a single angle), we can write this as
        #  * multiplication of a N x 3 matrix (groups the "x" data, X) by a 3x3 matrix = N x 3 (each row of result is = x_i^T @ E)
        #  * hadamard multiplication of previous result by a N x 3 matrix (groups the "x'" data, X')
        #  * Sum each row, then take the result as a N x 1 matrix, and take the square of each variate.
        #  * Compute the denominators in a similar manner, then sum all terms and rescale them accordingly.
        #
        # If we have multiple "E" matrices (corresponding to multiple angles), we can write this as:
        #  * multiplication of a N_angles x 3 matrix (x2_A) by a 3x (N_angles*3) matrix
        essential_matrix = self.essential_matrix(angle)

        #  * hadamard multiplication of previous result by a N x (N_angles*3) matrix. This latter is constructed as a copy of the X' matrix, N_angles times.
        tt = torch.matmul(x2_A.T, essential_matrix) * x2_B.T.repeat(1, N_angles)
        #  * Reshape as a N x N_angles x 3 tensor.
        tt2 = torch.reshape(tt, [N, N_angles, 3])
        #  * Sum each row, then take the result as a N x N_angles matrix, and take the square of each variate.
        tt3 = torch.sum(tt2, dim=2) ** 2
        if self.sampson_denominator:
            #  * Compute the denominators in a similar manner, then sum all terms over rows and rescale them accordingly.
            ex2A = torch.reshape(torch.matmul(x2_A.T, essential_matrix), [N, N_angles, 3])
            ex2B = torch.reshape(torch.matmul(x2_B.T, essential_matrix), [N, N_angles, 3])
            tt4 = torch.linalg.norm(ex2A, ord=1, dim=2)**2 + torch.linalg.norm(ex2A, ord=2, dim=2)**2 + torch.linalg.norm(ex2B, ord=1, dim=2)**2 + torch.linalg.norm(ex2B, ord=2, dim=2)**2
            samsonterm = tt3 / tt4
        else:
            samsonterm = tt3
        if self.likelihood_formula == 'engels-nister':
            # This corresponds to the original likelihood defined in Engels&Nister 2005.
            tt5 = torch.log(self.sigma2) - torch.log(self.sigma2 + samsonterm)
        elif self.likelihood_formula == 'generalized-gaussian':
            tt5 = -(.5 / self.sigma2) * samsonterm
        elif self.likelihood_formula == 'generalized-gaussian-nosigma':
            tt5 = samsonterm
        else:
            raise NotImplementedError(f'Unknown likelihood form: {self.likelihood_formula}')
        ###
        # At this point, tt5 will be a N x N_angles matrix of "loss" terms.
        # Row-wise, we have per-x loss terms;
        # Column-wise, we have per-z loss terms.
        if self.z is not None:
            # For each n \in [1,N], multiply each row with a weight given by the correct index of matrix "z".
            tt5 = tt5 * torch.outer(z, torch.ones(N_angles,).to(self.device))
        ## Sum or just return result individually over pairs?
        if(not self.sum_over_x):
            return(tt5)
        else:
            res = torch.sum(tt5, dim=0)
        ###
        if self.likelihood_formula == 'engels-nister':
            #  * The result is a N_angles x 1 vector of log-probabilities for each input angle.
            res = res * (N ** -self.k)
        return(res)




if __name__ == '__main__':
    pass
