import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from normflows.flows import Flow
from normflows import utils
from normflows.flows.affine.autoregressive import Autoregressive
from normflows.nets import made as made_module
from normflows.utils import splines
from normflows.utils.nn import PeriodicFeaturesElementwise





class MAF_Antimoebius(Autoregressive):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,             # Your spline will have bounds at [-tail_bound, +tail_bound] GS
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        permute_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
        xmin = None, #-torch.pi,               # If set to non-Tensor, this value will be copied 'feature'- times.
        xmax = None #+torch.pi
    ):
        self.moebius_parameters = 2 # A constant, really
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.xmin = xmin
        self.xmax = xmax

        if isinstance(self.tails, list) or isinstance(self.tails, tuple):
            ind_circ = []
            for i in range(features):
                if self.tails[i] == "circular":
                    ind_circ += [i]
            if torch.is_tensor(tail_bound):
                scale_pf = np.pi / tail_bound[ind_circ]
            else:
                scale_pf = np.pi / tail_bound
            preprocessing = PeriodicFeaturesElementwise(features, ind_circ, scale_pf)
        else:
            raise NotImplementedError("Please define 'self.tails' as a list or tuple. Use this as a 'pure' autoregression only, ie pdf_dimension must be at least 2")
            preprocessing = None

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            permute_mask=permute_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            preprocessing=preprocessing,
        )

        if init_identity:
            torch.nn.init.constant_(autoregressive_net.final_layer.weight, 0.0)
            torch.nn.init.constant_(
                autoregressive_net.final_layer.bias,
                np.log(np.exp(1 - min_derivative) - 1),
            )

        super().__init__(autoregressive_net)

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound)
        else:
            self.tail_bound = tail_bound

        #########################
        # This is required for Moebius...
        self.register_buffer('_zero_radian', torch.tensor([[1, 0]]), persistent=False)
        self.register_buffer('_I', torch.eye(2), persistent=False)


    def _output_dim_multiplier(self):
        # This is equal to: how many parameters for the flow, for each per pdf_dimension ("features")
        numparams_spline = self.num_bins * 3 + 1
        # For the moebius/antimoebius flow, we only need two parameters per dimension.
        # This is the cartesian coordinates of the point inside the unit sphere, acting as a parameter.
        numparams_antimoebius = self.moebius_parameters
        res = numparams_spline + numparams_antimoebius
        return res
        '''
        if self.tails == "linear":
            # We need one less parameter for the linear flow vs the circular one...
            return self.num_bins * 3 - 1
        elif self.tails == "circular":
            # because circular = there is a "wrap-around"
            return self.num_bins * 3
        else:
            ## ??? if self.tails is None or it is actually a list, pdf_dimension >= 2
        '''

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

    def _moebius_scalar(self, 
                        x,          # Δεδομένα 
                        w,          # Παράμετροι. Μπορεί να είναι διαφορετικές για κάθε datum.
                        xmin = -torch.pi,
                        xmax = +torch.pi,
                        zero_w_means_identity = True,       # True = the standard Moebius, defined in Rezende 2020.
                                                            # TODO: Run more tests on "False", not sure if debugged 100%
        ):
        '''
        Αυτή η συνάρτηση θεωρεί ότι τα inputs είναι στο εύρος [-π,π) και τα w είναι σε R2.
        Πρώτα θα τα μετατρέψει σε [0,2π) και μετα ξανά στο αρχικό εύρος.

        Shapes that are accepted
            x       batch_size x 1
            w       batch_size x 1 x 2
        '''
        b = xmax - xmin
        x = x - xmin
        x = x * (2*torch.pi)/b
        # So at this point x is in [0, 2π)
        if(zero_w_means_identity):
            # Add π. Same as x = -x
            x = x + torch.pi

        w = self.reparametrize_weights(w)                                       # was: είσοδος ήταν self.w
        print(w)
        z = torch.hstack([torch.cos(x), torch.sin(x)]) # n x 2                  # αυτή είναι η παραμετροποίηση της εισόδου-γωνίας σε καρτεσιανές συντεταγμένες
        g_z = self._g(z, w)
        tx = torch.atan2(g_z[..., 1], g_z[..., 0])
        tx = torch.where(tx >= 0, tx, tx + torch.pi * 2)         # Βεβαιώνουμε ότι η έξοδος είναι στο [0,2π]:
        tx = torch.where(tx < torch.pi*2, tx, tx - torch.pi * 2)      
        #tx = torch.sum(tx, dim=1, keepdim=True)                               
        ################################################################################
        # Compute log-determinant of transformation gradient
        ################################################################################
        z_w = z[:, None, :] - w
        z_w_norm = torch.norm(z_w, dim=-1)
        z_w_unit = z_w / z_w_norm[..., None]
        dz_dtheta = torch.hstack([-torch.sin(x), torch.cos(x)]) 
        dh_dz = (1 - torch.norm(w, dim=-1) ** 2)[..., None, None] * \
                    (self._I[None, None, ...] - 2 * torch.einsum('nki,nkj->nkij', z_w_unit, z_w_unit)) / \
                        (z_w_norm[..., None, None] ** 2)
        dh_dtheta = torch.einsum('nkpq,nq->nkp',dh_dz, dz_dtheta)
        dtx = torch.sum(torch.norm(dh_dtheta, dim=-1), dim=1)
        logdtx = torch.squeeze(torch.log(torch.abs(dtx))) #  Confirmed (GS).
        # Change inputs back to original bounds.
        # (normally there is a transformation gradient related to this of course,
        #   but it cancels out with the initial inverse transform)
        tx = tx * b/(2*torch.pi)
        tx = tx + xmin
        return(tx, logdtx)

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins : 3*self.num_bins + 1]
        # Antimoebius params
        antimoebius_params = transform_params[..., (3*self.num_bins+1):]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        def antimoebius_fn(x, inverse, w):
            #   x:      The input sample. Shaped as batch_size x features (ie = pdf dimensionality)
            #   w:      The parameters. Shaped as batch_size x features x 2
            #                   (Σημείωση: Το μέγεθος των "params" μπορεί να μπερδέψει -- πως γίνεται
            #                    να έχουμε ένα σετ παραμέτρων για κάθε δείγμα; Η απάντηση είναι απλά ότι
            #                    params είναι οι έξοδοι του νευρωνικού, το οποίο παράγει "params" βάσει του κάθε sample.
            #                    Δηλαδή αυτό που βελτιστοποιείται δεν είναι τα "params"-- που όντως είναι και πρέπει να είναι διαφορετικά για κάθε δείγμα--
            #                    αλλά οι παράμετροι του νευρωνικού δικτύου (autoregressive_net))
            #   xmin, xmax: The range of the input. Each shaped as 'features'-sized vector
            #

            #   logabsdet:  Shaped as batch_size x features
            #
            #
            #### Transform to range [0,2π)
            if inverse:
                w = -w
            tx = torch.zeros_like(x)
            logdtx = torch.zeros_like(x)
            for i in range(features):
                if(self.xmin[i] is not None and self.xmax[i] is not None):
                #if False:
                    xx, logdtx[:, i] = self._moebius_scalar(
                        x[:, i:i+1], w[:, i:i+1, :], xmin=self.xmin[i], xmax=self.xmax[i])
                    tx[:, i] = torch.squeeze(xx)
                else:
                    tx[:, i] = x[:, i]
            return tx, logdtx


        outputs, logabsdet_antimoebius = antimoebius_fn(
            x=outputs,                         # Use the outputs of the "previous" flow
            inverse=inverse,                    
            w=antimoebius_params,
        )
        # Add to the previous logabsdet
        logabsdet -= logabsdet_antimoebius
        return outputs, utils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class AutoregressiveSplineAntimoebiusFlow(Flow):
    def __init__(
        self,
        num_input_channels,                 # Data dimensionality.
        ind_circ,
        spline_tail_bound=3,                # Για περιοδικές + γωνιακές συνιστώσες αυτό πρέπει να είναι ίσο με pi.
        xmin=None,
        xmax=None,
    ):
        """

        All inputs are considered to represent angles.
        This is a hard constraint, so *all variates are considered to be bounded*.
        The real difference is between "circular" and "linear".
        The ones that are "circular" (ie periodic) have their index as part of the list "ind_circ".


        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          ind_circ (Iterable): Indices of the circular coordinates
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          spline_tail_bound (int): Bound of the spline tails. (GS): For periodic/angular variates this should be equal to pi.
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        tails = [
            "circular" if i in ind_circ else "linear" for i in range(num_input_channels)
        ]

        num_blocks=1
        num_bins=8
        num_hidden_channels = 512
        dropout_probability=0.0
        num_context_channels=None
        permute_mask=True
        init_identity=True
        
        # This is a variable that regards only the "Antimoebius" part.
        #xmin = [-torch.pi] * num_input_channels
        #xmax = [+torch.pi] * num_input_channels

        self.mprqat = MAF_Antimoebius(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=num_context_channels,
            num_bins=num_bins,
            tails=tails,
            tail_bound=spline_tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=torch.nn.ReLU(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity,
            xmin=xmin,
            xmax=xmax,
        )

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        return z, log_det.view(-1)




import unittest
from unittest import TestCase
from normflows.flows.flow_test import FlowTest
from normflows.flows.neural_spline import autoregressive
from normflows.flows.flow_test import FlowTest



class Moebius_mostly(FlowTest):
    def test_mprqas(self):
        batch_size = 5
        features = 10
        inputs = torch.rand(batch_size, features)

        flow = autoregressive.MaskedPiecewiseRationalQuadraticAutoregressive(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        #    xmin = [-torch.pi] * features,
        #    xmax = [+torch.pi] * features,
        )

        self.checkForwardInverse(flow, inputs)

    def test_antimoebius_0_alllinear_nomoebius(self):
        batch_size = 50
        features = 3
        inputs = torch.rand(batch_size, features)
        flow = MAF_Antimoebius(
            features=features,
            hidden_features=30,
            tails=['linear']*features,
            xmin = [None] * features,
            xmax = [None] * features,            
        )
        self.checkForwardInverse(flow, inputs)

    def test_antimoebius_0_allcircular_nomoebius(self):
        batch_size = 50
        features = 3
        inputs = torch.rand(batch_size, features)
        flow = MAF_Antimoebius(
            features=features,
            hidden_features=30,
            tails=['circular']*features,
            xmin = [None] * features,
            xmax = [None] * features,            
        )
        self.checkForwardInverse(flow, inputs)

    def test_antimoebius_0_allcircular_nomoebius_range_minusone_to_plusone(self):
        batch_size = 50
        features = 3
        inputs = 2*torch.rand(batch_size, features) - 1.
        flow = MAF_Antimoebius(
            features=features,
            hidden_features=30,
            tails=['circular']*features,
            xmin = [None] * features,
            xmax = [None] * features,            
        )
        self.checkForwardInverse(flow, inputs)

    def test_antimoebius_0_allcircular_nomoebius_range_minuspi_to_pluspi(self):
        batch_size = 50
        features = 3
        inputs = torch.pi * (2*torch.rand(batch_size, features) - 1.)
        flow = MAF_Antimoebius(
            features=features,
            hidden_features=30,
            tail_bound=torch.pi,                    # Necessary if you want your spline to have a range more than [-1,+1]
            tails=['circular']*features,
            xmin = [None] * features,
            xmax = [None] * features,            
        )
        self.checkForwardInverse(flow, inputs)

    def test_antimoebius_0_allcircular_30_features(self):
        batch_size = 50
        features = 30
        inputs = torch.pi * (2*torch.rand(batch_size, features) - 1.)
        flow = MAF_Antimoebius(
            features=features,
            hidden_features=30,
            tail_bound=torch.pi,                    # Necessary if you want your spline to have a range more than [-1,+1]
            tails=['circular']*features,
            xmin = [-torch.pi] * features,
            xmax = [+torch.pi] * features,            
        )
        self.checkForwardInverse(flow, inputs)

    def test_wrapper_function_all_circular(self):
        batch_size = 50
        features = 30
        bound = torch.pi
        inputs = bound * (2*torch.rand(batch_size, features) - 1.)
        flow = AutoregressiveSplineAntimoebiusFlow(
            num_input_channels = features,
            ind_circ = list(range(features)),
            spline_tail_bound = bound,
            xmin = [-bound] * features,
            xmax = [+bound] * features,
        )
        self.checkForwardInverse(flow, inputs)

    def test_wrapper_function_part_circular(self):
        batch_size = 50
        features = 30
        bound = torch.pi
        inputs = bound * (2*torch.rand(batch_size, features) - 1.)
        flow = AutoregressiveSplineAntimoebiusFlow(
            num_input_channels = features,
            ind_circ = list(range(features // 2)),
            spline_tail_bound = bound,
            xmin = [-bound] * features,
            xmax = [+bound] * features,
        )
        self.checkForwardInverse(flow, inputs)

    def test_wrapper_function_part_circular_and_bigbound(self):
        batch_size = 50
        features = 30
        bound = 10
        inputs = bound * (2*torch.rand(batch_size, features) - 1.)
        flow = AutoregressiveSplineAntimoebiusFlow(
            num_input_channels = features,
            ind_circ = list(range(features // 2)),
            spline_tail_bound = bound,
            xmin = [-bound] * features,
            xmax = [+bound] * features,
        )
        self.checkForwardInverse(flow, inputs)


    def test_0(self):
        batch_size = 50
        features = 2
        bound = 2.5
        inputs = bound * (2*torch.rand(batch_size, features) - 1.)
        inputs[:, 1] = torch.pi * (2*torch.rand(batch_size) - 1.)
        flow = AutoregressiveSplineAntimoebiusFlow(
            num_input_channels=features, 
            ind_circ=[1], 
            spline_tail_bound=torch.Tensor([bound, np.pi]),
            xmin = [None, -np.pi],
            xmax = [None, +np.pi],
        )
        self.checkForwardInverse(flow, inputs)

    def test_0_only_second_is_circular(self):
        batch_size = 50
        features = 2
        bound = 2.5
        inputs = bound * (2*torch.rand(batch_size, features) - 1.)
        inputs[:, 1] = torch.pi * (2*torch.rand(batch_size) - 1.)
        flow = AutoregressiveSplineAntimoebiusFlow(
            num_input_channels=features, 
            ind_circ=[1], 
            spline_tail_bound=torch.Tensor([bound, np.pi]),
            xmin = [None, -np.pi],
            xmax = [None, +np.pi],
        )
        self.checkForwardInverse(flow, inputs)

if __name__ == "__main__":
    unittest.main()