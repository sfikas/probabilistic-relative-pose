import numpy as np
import torch
import unittest
from unittest import TestCase
from normflows.flows.flow_test import FlowTest
from normflows.flows.base import Flow, zero_log_det_like_z
from torch import nn
import normflows as nf
from normflows.flows.neural_spline import autoregressive
from normflows.flows.flow_test import FlowTest


class VerySimpleFlow(Flow):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(2,))

    def forward(self, z):
        '''
        Είναι η κατεύθυνση από την βασική προς την περίπλοκη κατανομή.
        Tο χρειαζόμαστε για την εκπαίδευση (reverse_kld)
        '''
        a = self.weights[0]
        b = self.weights[1]
        # Αυτός είναι ο μετασχηματισμός f: dom(z) -> dom(x)
        x = torch.exp(a)*z + b
        # Αυτός είναι λογάριθμος της παραγώγου του μετασχηματισμού f: dom(z) -> dom(x).
        # Δηλαδή log grad x ως προς z.
        logdetgrad = a
        return(x, logdetgrad)

    def inverse(self, x):
        '''
        Είναι η κατεύθυνση από την περίπλοκη προς την βασική κατανομή (η "κανονικοποιητική" κατεύθυνση)
        Αυτό το χρειαζόμαστε για να αποτιμήσουμε την πιθανότητα ενός νέου δείγματος.
        '''
        a = self.weights[0]
        b = self.weights[1]
        z = (x-b)/torch.exp(a)
        logdetgrad = -a
        return(z, logdetgrad)

class Antimoebius(Flow):
    '''
    Υλοποίηση της συνάρτησης "g" στο paper των Rezende et al. 2020.
    Ισχύει η ιδιότητα g^{-1} = g, για την ίδια παράμετρο.
    '''
    def __init__(self, xmin=0, xmax=2*torch.pi):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.weights = nn.Parameter(torch.randn(1, 1)) #[None])             # Πρόκειται για βάρη που αντιστοιχούν σε σύνθεση Moebius -- δεν μας ενδιαφέρει! (Λάθος: ~~Βάρη που δεν παίζουν ρόλο όταν έχουμε S1.~~)
        self.w = nn.Parameter(torch.randn(1, 1, 2))                         # Οι παράμετροι μας, σε καρτεσιανές συντεταγμένες.
        # "Register_buffer" σημαίνει ότι ορίζουμε "βοηθητικές" σταθερές ως τμήματα του μοντέλου, οι οποίες όμως δεν προορίζονται προς βελτιστοποίηση.
        # Αυτή η μεταβλητή αντιστοιχεί στις καρτεσιανές συντεταγμένες εισόδου-γωνίας ίσης με 0.
        self.register_buffer('_zero_radian', torch.tensor([[1, 0]]), persistent=False)
        # Αυτή η μεταβλητή αντιστοιχεί 
        self.register_buffer('_I', torch.eye(2), persistent=False)

    def _h(self, z, w):
        w_norm = torch.norm(w, dim=-1, keepdim=True) # n x k x 1
        h_z = (1 - w_norm ** 2) / (torch.norm((z.reshape(-1, 1, 2) - w), dim=-1, keepdim=True) ** 2) * (z.reshape(-1, 1, 2) - w) - w
        return h_z

    def _g(self, z, w):
        '''
        According to the Figure in 2.2.1 of Rezende 2020, g() = -h().
        g() can be understood as a "reflection" of input z, with respect to w.
        '''
        res = -self._h(z, w)
        return(res)

    def reparametrize_weights(self, w, p=0.99):
        '''
        Αναπαραμετροποίηση που σκοπό έχει να κρατήσει το w μακριά από το να έχει νόρμα = 1 (σελ.7). 
        Θα είναι χρήσιμο να ρυθμίσουμε το p σε ακόμα μικρότερες τιμές, αν θέλουμε να επιβάλλουμε όρια στην καμπυλότητα της εξόδου
        (μικρότερο p συνεπάγεται λιγότερο ομοιόμορφη την κατανομή στην έξοδο)

        Προσοχή ότι *κάποια* αναπαραμετροποίηση είναι υποχρεωτική από τη στιγμή που το input w είναι σε απεριόριστο domain (R2)
        '''
        return(p / (1 + torch.norm(w, dim=-1, keepdim=True)) * w)
        #return(p / (1 + torch.norm(self.w, dim=-1, keepdim=True)) * w)  # Erratum? self.w should be w
        #return(0.8*p * w / torch.norm(self.w, dim=-1, keepdim=True))

    def forward(self, x):
        '''
        Είναι η κατεύθυνση από την βασική προς την περίπλοκη κατανομή.
        Tο χρειαζόμαστε για την εκπαίδευση (reverse_kld)

        Adapted from "flows-on-spheres" repo.
        '''
        #### Transform to range [0,2π)
        b = self.xmax - self.xmin
        x = x - self.xmin
        x = x * (2*torch.pi)/b
        ################################################################################
        # Compute transformation
        ################################################################################
        weights = torch.softmax(self.weights, dim=1)                            # οπότε τα weights είναι στο [0,1] και αθροίζουν σε μονάδα (αρα αν έχουμε S1 - κύκλο, δεν παίζουν ρόλο)
        w = self.reparametrize_weights(self.w)
        z = torch.hstack([torch.cos(x), torch.sin(x)]) # n x 2                  # αυτή είναι η παραμετροποίηση της εισόδου-γωνίας σε καρτεσιανές συντεταγμένες
        #h_z = self._h(z, w)                                                          # ***Moebius transform*** -- εξίσωση 8 στο paper
        g_z = self._g(z, w)
        #h_z = self._h(h_z, w)

        #h_zero_radian = self._h(self._zero_radian, w)
        #h_zero_radian = self._h(h_zero_radian, w)

        #radians = torch.atan2(h_z[..., 1], h_z[..., 0])
        radians = torch.atan2(g_z[..., 1], g_z[..., 0])
        #shifts = torch.atan2(h_zero_radian[..., 1], h_zero_radian[..., 0])
        
        tx = radians
        #tx = radians - shifts
        tx = torch.where(tx >= 0, tx, tx + torch.pi * 2)                               # Βεβαιώνουμε ότι η έξοδος είναι στο [0,2π]
        #tx = torch.where(tx <= 2*torch.pi, tx, tx - torch.pi * 2)                               # Βεβαιώνουμε ότι η έξοδος είναι στο [0,2π]
        #tx = torch.where(tx >= -torch.pi, tx, tx + 2*torch.pi)                          # Βεβαιώνουμε ότι η έξοδος είναι στο [-π,π]
        tx = torch.sum(weights * tx, dim=1, keepdim=True)                               # Έχει νόημα μόνο όταν έχουμε S2 και πάνω (αν έχουμε S1-κύκλο, δεν παίζει ρόλο)
        ################################################################################
        # Compute log-determinant of transformation gradient
        ################################################################################
        z_w = z[:, None, :] - w
        z_w_norm = torch.norm(z_w, dim=-1)
        z_w_unit = z_w / z_w_norm[..., None]
        # n x 2
        dz_dtheta = torch.hstack([-torch.sin(x), torch.cos(x)]) 
        # n x k x 2 x 2
        dh_dz = (1 - torch.norm(w, dim=-1) ** 2)[..., None, None] * \
                    (self._I[None, None, ...] - 2 * torch.einsum('nki,nkj->nkij', z_w_unit, z_w_unit)) / \
                        (z_w_norm[..., None, None] ** 2)
        
        dh_dtheta = torch.einsum('nkpq,nq->nkp',dh_dz, dz_dtheta)
        dtx = torch.sum(torch.norm(dh_dtheta, dim=-1) * weights, dim=1)

        ## Transform back to original range
        tx = tx * b/(2*torch.pi)
        tx = tx + self.xmin
        #
        return tx, torch.log(dtx) #+ torch.log(torch.Tensor([2*torch.pi/b]))


    def inverse(self, x):
        '''
        Είναι η κατεύθυνση από την περίπλοκη προς την βασική κατανομή (η "κανονικοποιητική" κατεύθυνση)
        Αυτό το χρειαζόμαστε για να αποτιμήσουμε την πιθανότητα ενός νέου δείγματος.
        '''
        tt, dtt = self.forward(x)
        return tt, dtt


from normflows.flows.neural_spline.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from normflows.flows import CircularAutoregressiveRationalQuadraticSpline

from tt_flowtest import AutoregressiveSplineAntimoebiusFlow


def randomize_flow_network(myflow):
    # Randomize parameters of the MADE neural network inside the flow
    made = myflow.flows[0].mprqat.autoregressive_net
    for param in made.parameters():
        param.data = torch.randn_like(param)        

class Test_Bounded_Densities(FlowTest):
    '''
    This is a bunch of tests 
    '''

    def setUp(self):
        self.xmin = -torch.pi
        self.xmax = +torch.pi
        self.bound_size = self.xmax - self.xmin
        self.nsteps = 1000
        self.stepsize = self.bound_size / self.nsteps
        # Create a uniform distribution, with domain over -pi .. +pi.
        self.base = nf.distributions.UniformGaussian(1, [0], torch.tensor([self.bound_size]))
        self.base2 = nf.distributions.UniformGaussian(2, [0,1], torch.tensor([self.bound_size, self.bound_size]))

    def test_density_uniform(self):
        #a = base.sample(batch_size)
        logprobs = self.base.log_prob(torch.linspace(self.xmin, self.xmax, self.nsteps))
        # Compute Riemannian integral approximation
        total_mass = torch.sum(torch.exp(logprobs)) * self.stepsize
        self.assertAlmostEqual(total_mass.numpy(), 1.0, places=5)

    def test_density_antimoebius(self):
        myflow = nf.NormalizingFlow(self.base, [Antimoebius()], p=None)
        logprobs = myflow.log_prob(torch.unsqueeze(torch.linspace(self.xmin, self.xmax, self.nsteps), dim=1))
        # Compute Riemannian integral approximation
        total_mass = torch.sum(torch.exp(logprobs)) * self.stepsize
        self.assertAlmostEqual(total_mass.detach().numpy(), 1.0, places=2)

    def test_density_autoregressive_circularonlyspline_original(self):
        myflow = nf.NormalizingFlow(self.base, [
            CircularAutoregressiveRationalQuadraticSpline(1, 1, 512, [0], num_bins=10, tail_bound=torch.tensor([np.pi]), permute_mask=True)
            ], p=None)
        logprobs = myflow.log_prob(torch.unsqueeze(torch.linspace(self.xmin, self.xmax, self.nsteps), dim=1))
        total_mass = torch.sum(torch.exp(logprobs)) * self.stepsize
        self.assertAlmostEqual(total_mass.detach().numpy(), 1.0, places=2)

    def test_density_autoregressive_moebius_deactivated(self):
        myflow = nf.NormalizingFlow(self.base, [
            AutoregressiveSplineAntimoebiusFlow(1, [0], spline_tail_bound=np.pi, xmin=[None], xmax=[None])
            #CircularAutoregressiveRationalQuadraticSpline(1, 1, 512, [0], num_bins=10, tail_bound=torch.tensor([np.pi]), permute_mask=True)
            ], p=None)
        logprobs = myflow.log_prob(torch.unsqueeze(torch.linspace(self.xmin, self.xmax, self.nsteps), dim=1))
        total_mass = torch.sum(torch.exp(logprobs)) * self.stepsize
        self.assertAlmostEqual(total_mass.detach().numpy(), 1.0, places=2)

    '''
    def test_density_autoregressive_moebius(self):
        myflow = nf.NormalizingFlow(self.base, [
            AutoregressiveSplineAntimoebiusFlow(1, [0], spline_tail_bound=np.pi, xmin=[self.xmin], xmax=[self.xmax])
            ], p=None)
        randomize_flow_network(myflow)
        #
        logprobs = myflow.log_prob(torch.unsqueeze(torch.linspace(self.xmin, self.xmax, self.nsteps), dim=1))
        total_mass = torch.sum(torch.exp(logprobs)) * self.stepsize
        self.assertAlmostEqual(total_mass.detach().numpy(), 1.0, places=2)

    def test_density_autoregressive_moebius_twodimensional_circular(self):
        myflow = nf.NormalizingFlow(self.base2, [
            AutoregressiveSplineAntimoebiusFlow(2, [0,1], spline_tail_bound=np.pi, xmin=[self.xmin, self.xmin], xmax=[self.xmax, self.xmax])
            ], p=None)
        randomize_flow_network(myflow)
        #
        x = torch.linspace(self.xmin, self.xmax, self.nsteps)
        y = torch.linspace(self.xmin, self.xmax, self.nsteps)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        logprobs = myflow.log_prob(points)
        total_mass = torch.sum(torch.exp(logprobs)) * (self.stepsize ** 2)
        self.assertAlmostEqual(total_mass.detach().numpy(), 1.0, places=2)
    '''

class Moebius_mostly(FlowTest):
    def test_verysimpleflow(self):
        batch_size = 100
        for id_init in [True, False]:
            with self.subTest(id_init=id_init):
                flow = VerySimpleFlow()
                inputs = torch.randn((batch_size, 1))
                self.checkForwardInverse(flow, inputs)

    def test_reflectivity(self):
        '''
        Αν χρησιμοποιήσουμε την g αντί την αντίθετη της, h (βλ. paper Rezende 2020),
        τότε το flow πρέπει να έχει την ιδιότητα το forward να είναι ίδιο με το inverse, given ίσες παραμέτρους.
        '''
        batch_size = 1
        base = nf.distributions.UniformGaussian(1, [0], torch.tensor([2*np.pi]))
        myflow = Antimoebius()
        #model = nf.NormalizingFlow(base, [myflow], VonMises(0, 1))
        inputs = 2*torch.pi * torch.rand((batch_size, 1)) #- torch.pi
        outputs, detx = myflow(inputs)
        #print(f'Output for input {inputs} is: {outputs}')
        #print(f'Logdet for the aforementioned is: {detx}')
        #print('\n\n=============================================================\n\n')
        outputs_outputs, detx2 = myflow.inverse(outputs)
        #print(f'Output with previous output {outputs} as input is: {outputs_outputs}')
        #print(f'Logdet for the aforementioned is: {detx2}')
        t1 = torch.squeeze(outputs_outputs).detach().numpy()
        t2 = torch.squeeze(inputs).numpy()
        self.assertAlmostEqual(t1, t2, places=5)
        self.assertAlmostEqual(torch.squeeze(detx).detach().numpy(), - torch.squeeze(detx2).detach().numpy(), places=5)

    def test_antimoebius(self):
        batch_size = 100
        for id_init in [True, False]:
            with self.subTest(id_init=id_init):
                flow = Antimoebius()
                inputs = 2*torch.pi * torch.rand((batch_size, 1)) #- torch.pi
                self.checkForwardInverse(flow, inputs)

    def test_antimoebius_different_range(self):
        batch_size = 100
        for id_init in [True, False]:
            with self.subTest(id_init=id_init):
                flow = Antimoebius(xmin=-np.pi,xmax=np.pi)
                inputs = 2*torch.pi * torch.rand((batch_size, 1)) - torch.pi
                self.checkForwardInverse(flow, inputs)

    def test_antimoebius_different_range_2(self):
        batch_size = 100
        for id_init in [True, False]:
            with self.subTest(id_init=id_init):
                flow = Antimoebius(xmin=-np.pi/2,xmax=np.pi/2)
                inputs = torch.pi * torch.rand((batch_size, 1)) - torch.pi/2
                self.checkForwardInverse(flow, inputs)

    def test_antimoebius_x(self):
        for _ in range(100):
            a = Antimoebius()
            # Χωρίς αναπαραμετροποίηση
            weights = a.w
            # Με αναπαραμετροποίηση
            weights_reparam = a.reparametrize_weights(a.w)
            self.assertLess(torch.norm(weights_reparam), 1.0)

    def test_outputrange(self):
        for _ in range(100):
            a = Antimoebius()
            batch_size = 1
            #res = a.forward(torch.Tensor([[3]]))
            res, _ = a.forward(2*torch.pi * torch.rand((batch_size, 1)))
            self.assertLessEqual(res, torch.Tensor([[2*torch.pi]]))
            self.assertGreaterEqual(res, torch.Tensor([[0]]))

    def test_outputrange_2a(self):
        for _ in range(100):
            a = Antimoebius(xmin=-torch.pi, xmax=+torch.pi)
            batch_size = 1
            #res = a.forward(torch.Tensor([[3]]))
            res, _ = a.forward(2*torch.pi * torch.rand((batch_size, 1)) - torch.pi)
            self.assertLessEqual(res, torch.Tensor([[torch.pi]]))
            self.assertGreaterEqual(res, torch.Tensor([[-torch.pi]]))
          
    def test_outputrange_2(self):
        for _ in range(100):
            a = Antimoebius(xmin=-torch.pi/2, xmax=+torch.pi/2)
            batch_size = 1
            #res = a.forward(torch.Tensor([[3]]))
            res, _ = a.forward(torch.pi * torch.rand((batch_size, 1)) - torch.pi/2)
            self.assertLessEqual(res, torch.Tensor([[torch.pi/2]]))
            self.assertGreaterEqual(res, torch.Tensor([[-torch.pi/2]]))

    def test_batchsize(self):
        for _ in range(100):
            a = Antimoebius(xmin=-torch.pi/2, xmax=+torch.pi/2)
            batch_size = 10
            res, _ = a.forward(torch.pi * torch.rand((batch_size, 1)) - torch.pi/2)


class MaskedPiecewiseRationalQuadraticAutoregressiveFlowTest(FlowTest):
    def test_autoregressivebase_2(self):
        # Create normalizing flow
        flow = nf.flows.CircularAutoregressiveRationalQuadraticSpline(
            num_input_channels= 2, 
            num_blocks=1, 
            num_hidden_channels=128, 
            ind_circ=[1],                   # Indices of circular variates
            tail_bound=torch.tensor([5., np.pi]),
            permute_mask=True)
        # (The above actually calls MaskedPiecewiseRationalQuadraticAutoregressive)
        base = nf.distributions.UniformGaussian(2, [1], torch.tensor([1., 2 * np.pi]))
        # Visualize base
        inputs = base.sample(1)
        self.checkForwardInverse(flow, inputs)
        #model = nf.NormalizingFlow(base, flow_layers)

    def test_autoregressivebase_3(self):
        # Create normalizing flow
        flow = nf.flows.CircularAutoregressiveRationalQuadraticSpline(
            num_input_channels= 3, 
            num_blocks=1, 
            num_hidden_channels=128, 
            ind_circ=[1,2],                   # Indices of circular variates
            tail_bound=torch.tensor([5., np.pi, np.pi]),
            permute_mask=True)
        base = nf.distributions.UniformGaussian(
            ndim=3, 
            ind=[1,2], 
            scale=torch.tensor([1., 2 * np.pi, 2 * np.pi])) #TODO these values should be corrected
        inputs = base.sample(1)
        self.checkForwardInverse(flow, inputs)


if __name__ == "__main__":
    unittest.main()