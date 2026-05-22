import numpy as np
import torch
import normflows as nf
from tqdm import tqdm
from matplotlib import pyplot as plt

from bayesrpedistributions import EngelsNister
from bayesrpeflows import Antimoebius
# createFreshFlows
# trainFlow
# plot_flows
# "EngelsNister"
# countparameters
# createSyntheticDataset
#
#
# Giorgos Sfikas 2025



def createFreshFlows(num_kernels, num_layers = 2, num_hidden_channels = 8, num_bins = 6):
    flow_layers = [ [], [] ]
    flow_layers = []
    for k in range(num_kernels):
        flow_layers.append([])
        for _ in range(num_layers):
            flow_layers[k] += [nf.flows.CircularAutoregressiveRationalQuadraticSpline(
                num_input_channels = 1,
                num_blocks = 1,
                num_hidden_channels = num_hidden_channels,
                ind_circ = [0],
                num_bins = num_bins,
                tail_bound = np.pi / 2,
                activation = torch.nn.ReLU)
            ]
        flow_layers[k] += [Antimoebius(xmin=-np.pi/2, xmax=+np.pi/2)]
    return(flow_layers)

#print(f'The model has {count_parameters(model)} learnable parameters.')
def trainFlow(x2_A, x2_B, z, sigma2, 
              E_prior, base, 
              flow_layers, 
              likelihood_formula = 'generalized-gaussian',
              initialization = None, max_iter = 100, num_samples = 2**8, enable_cuda = True,
              use_tqdm = True,
              learning_rate = 5e4
              ):
    '''
    x2_A, x2_B:             Input data (matches).
    z:                      Z matrix. This is about responsibility of each flow for each datum (match).
    sigma2:
    E_prior:                The prior distribution over E.
    base:                   NF Base distribution. Example:
                                nf.distributions.UniformGaussian(1, [0], torch.tensor([np.pi]))
    flow_layers:            A list of flow layers. Each element of the list must be an object inheriting from the "Flow" class.
    initialization:         If None, initialize using a randomly initalized flow.
                            Otherwise, use the provided flow definition (should be a "flow" model)
    max_iter:               Maximum number of NF training iterations.
    num_samples:            Use this many samples for training in each iteration.
    enable_cuda:
    use_tqdm:               Use tqdm on the sampling iteration loop.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    target = EngelsNister(
                        x2_A=x2_A, x2_B=x2_B, 
                        #z=z[k, :] * u[k, :], sigma2=sigma2[k],
                        z=z, sigma2=sigma2,
                        likelihood_formula=likelihood_formula,
                        )
    target = target.to(device)
    base = base.to(device)

    if E_prior is not None:
        raise NotImplementedError('Using a custom prior distribution over E is not implemented yet. A Uniform prior is assumed.')
    
    if initialization is None:
        model = nf.NormalizingFlow(base, flow_layers, target)
    else:
        model = initialization
    model.train()
    model = model.to(device)    

    loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)
    if use_tqdm:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)
    for _ in loop:
        optimizer.zero_grad()
        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        #     record_shapes=True,
        #     with_stack=True
        # ) as prof:
        #     loss = model.reverse_kld(num_samples)
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
        loss = model.reverse_kld(num_samples)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        scheduler.step()
    return(model)

def plot_flows(flowmodels, plotsize = (9,6), xmin = -np.pi/2, xmax = +np.pi/2, nomax = False, wait = False, disp = None):
    '''
    Plot a set of normalizing flows fit over 1D data.
    Input:
        A list of lists comprising "Flow" objects.
    '''
    grid_size = 60
    xx = torch.meshgrid(
        torch.linspace(xmin, xmax, grid_size)
    ); xx = xx[0].unsqueeze(1)
    if type(plotsize) is tuple:
        plt.figure(figsize=plotsize)
    else:
        plt.figure(figsize=(plotsize, plotsize))
    if type(flowmodels) is list:
        num_kernels = len(flowmodels)
    else:
        num_kernels = 1
        flowmodels = [flowmodels]
    prob = []
    for k in range(num_kernels):
        current_model = flowmodels[k]
        current_model = current_model.to('cpu')
        current_model.eval()
        with torch.no_grad():
            log_prob = current_model.log_prob(xx)
            tt = torch.exp(log_prob)
            tt[torch.isnan(tt)] = 0
            prob.append(tt)

    maxprob = torch.max(torch.stack(prob))
    for k in range(num_kernels):
        plt.subplot(1, num_kernels, k+1)
        plt.plot(xx, prob[k], label=f'Model {k}')
        if not nomax:
            plt.ylim([0, maxprob])
        plt.legend()
        if disp is not None:
            disp.clear_output(wait=wait)
            disp.display(plt.gcf()) 
    plt.show()        


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# baseline_angle = [np.pi/4, -np.pi/3]
def createSyntheticDataset(baseline_length, baseline_angle, spatialPointPlaneDistance = 10, spatialPointVariance = 6, N_points = 100):
    '''
    We'll create a 3D point cloud.
    One major difference with Notebook 1 is that half the points will move according to some rotation-translation pair, 
    and the other half according to another pair.

    The inputs "baseline_length, baseline_angle, CM_rotation" describe the properties of the Composing Motions (CMs). 
    Each variate corresponds to one CM.
    (For now) we assume *no* rotation, hence all K Essential Matrices turn out of the form [t]_x @ I aka just a 2DOF skew-matrix each.

    Inputs:
        baseline_length:                        Default value [10, 10]. Shaped as 1xK. 
        baseline_angle:                         Default value [np.pi/4, -np.pi/3]. Shaped as 1xK.
        CM_rotation (not implemented yet):      Shaped as 3x3xK.
        K (implied):                            The number of ground-truth (and assumed) composing motions.
    Outputs:
        X3:   3D points, shaped as 4xK*N_points (in homogeneous coords)
        x2_A: 2D points as projected in view A, shaped as 3xK*N_points (in homogeneous coords)
        x2_B: 2D points as projected in view B, shaped as 3xK*N_points (in homogeneous coords)

    
    '''
    def projectPoints(cameraMatrix, homogeneousCoords, forceUnity=True):
        '''
        Inputs:
            cameraMatrix:           A camera matrix, shaped as 3x4.
            homogeneousCoords:      Input 3D homogeneous coordinates, shaped as 4xN, where N is the number of input points.
            forceUnity:             This will project the output so that the last coordinate is always equal to 1.
        Output:
                                    2D homogeneous coordinates of the projected points, shaped as 3xN.
        '''
        _, N = homogeneousCoords.shape
        tt = cameraMatrix @ homogeneousCoords
        if not forceUnity:
            return(tt)
        res = np.zeros([3, N])
        res[0, :] = tt[0, :] / tt[2, :]
        res[1, :] = tt[1, :] / tt[2, :]
        res[2, :] = np.ones([1,N])
        return(res)
    ###################################################################################################
    X3 = []
    x2_B = []
    K = len(baseline_length)
    ###################################################################################################
    cameraMatrix_A = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]])
    cameraMatrix_B = np.zeros([K, 3, 4])
    for k in range(K):
        cameraMatrix_B[k, :, :] = np.array(
                            [[1, 0, 0, np.cos(baseline_angle[k])*baseline_length[k]],
                            [0, 1, 0, np.sin(baseline_angle[k])*baseline_length[k]],
                            [0, 0, 1, 0]]
        )
    for k in range(K):
        X3part = np.random.randn(3, N_points)
        if False:
            '''
            Initial, "manual" noise addition
            '''
            if k == 0:
                X3part[-1, :] += 10
            elif k == 1:
                # TODO: Run another experiment where the 3D point clusters are not identical 
                # to the two composing motion clusters (I was just mentally lazy here)
                X3part[0, :] += 3
                X3part[1, :] += -4
                X3part[2, :] += 5
                #X3part[-1, :] += 10
            else: 
                raise ValueError
        else:
            X3part[-1, :] += spatialPointPlaneDistance                                              # Move everything away from the camera
            X3part[0, :] += spatialPointVariance * np.random.rand() - spatialPointVariance / 2      # And add random noise to the cluster
            X3part[1, :] += spatialPointVariance * np.random.rand() - spatialPointVariance / 2
            X3part[2, :] += spatialPointVariance * np.random.rand() - spatialPointVariance / 2

        # Convert to homogeneous coordinates
        X3part = np.vstack([X3part, np.ones([1,N_points])])
        x2_Bpart = projectPoints(cameraMatrix_B[k, : ,:], X3part)
        x2_B.append(x2_Bpart)
        X3.append(X3part)

    X3 = np.concatenate(X3, axis=1)
    x2_B = np.concatenate(x2_B, axis=-1)
    x2_A = projectPoints(cameraMatrix_A, X3)
    return(X3, x2_A, x2_B)
