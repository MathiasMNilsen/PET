# External imports
import numpy as np
import time
import pprint
import warnings

from numpy import linalg as la
from scipy.optimize import line_search

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize

# ignore line_search did not converge message
warnings.filterwarnings('ignore', message='The line search algorithm did not converge')


class LineSearch(Optimize):

    def __init__(self, fun, x, args, jac, hess, bounds=None, **options):

        # init PETEnsemble
        super(LineSearch, self).__init__(**options)

        # Set input as class variables
        self.options    = options   # options
        self.fun        = fun       # objective function
        self.cov        = args[0]   # initial covariance
        self.jac        = jac       # gradient function
        self.hess       = hess      # hessian function
        self.bounds     = bounds    # parameter bounds
        self.mean_state = x         # initial mean state
        self.pk_from_ls = None
        
        # Set other optimization parameters
        self.alpha_iter_max  = options.get('alpha_maxiter', 5)
        self.alpha_cov       = options.get('alpha_cov', 0.01)
        self.normalize       = options.get('normalize', True)
        self.iter_resamp_max = options.get('resample', 0)
        self.shrink_factor   = options.get('shrink_factor', 0.25)
        self.alpha           = 0.0

        # Initialize line-search parameters (scipy defaults for c1, and c2)
        self.alpha_max  = options.get('alpha_max', 1.0)
        self.ls_options = {'c1': options.get('c1', 0.0001),
                           'c2': options.get('c2', 0.9)}
        
        
        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()
            self.obj_func_values = self.fun(self.mean_state)
            self.nfev += 1
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)
            if self.logger is not None:
                self.logger.info('       ====== Running optimization - EnOpt ======')
                self.logger.info('\n'+pprint.pformat(self.options))
                info_str = '       {:<10} {:<10} {:<15} {:<15} {:<15} '.format('iter', 'alpha_iter',
                                                                        'obj_func', 'step-size', 'cov[0,0]')
                self.logger.info(info_str)
                self.logger.info('       {:<21} {:<15.4e}'.format(self.iteration, np.mean(self.obj_func_values)))

        
        self.run_loop() 
                
    def calc_update(self, iter_resamp=0):

        # Initialize variables for this step
        success = False

        # Dummy functions for scipy.line_search
        def _jac(x):
            self.njev += 1
            x = ot.clip_state(x, self.bounds) # ensure bounds are respected
            g = self.jac(x, self.cov)
            return g
        
        def _fun(x):
            self.nfev += 1
            x = ot.clip_state(x, self.bounds) # ensure bounds are respected
            f = self.fun(x, self.cov).mean()
            return f

        #Compute gradient. If a line_search is already done, the new grdient is alread returned as slope by the function
        if self.pk_from_ls is None:
            pk = _jac(self.mean_state)
        else:
            pk = self.pk_from_ls
    
        # Compute the hessian (Not Used Currently)
        hessian = self.hess()

        if self.normalize:
            hessian /= np.maximum(la.norm(hessian, np.inf), 1e-12)  # scale the hessian with inf-norm
            pk_norm = la.norm(pk, np.inf)
        else:
            pk_norm = 1
            
        # Perform Line Search
        self.logger.info('Performing line search...')
        line_search_kwargs = {'f'        : _fun,
                              'myfprime' : _jac,
                              'xk'       : self.mean_state,
                              'pk'       : -pk/pk_norm,
                              'gfk'      : pk,
                              'old_fval' : self.obj_func_values.mean(),
                              'c1'       : self.ls_options['c1'],
                              'c2'       : self.ls_options['c2'],
                              'amax'     : self.alpha_max,
                              'maxiter'  : self.alpha_iter_max}
        
        step_size, _, _, fnew, _, slope = line_search(**line_search_kwargs)
        
        if isinstance(step_size, float):
            self.logger.info('Strong Wolfie conditions satisfied')

            # Update state
            self.mean_state      = ot.clip_state(self.mean_state - step_size*pk/pk_norm, self.bounds)
            self.obj_func_values = fnew
            self.alpha           = step_size
            self.pk_from_ls      = slope

            # Update covariance 
            #TODO: This sould be mande into an callback function for generality 
            #      (in case of non ensemble gradients or GenOpt gradient)
            self.cov = self.cov - self.alpha_cov * hessian
            self.cov = ot.get_sym_pos_semidef(self.cov)

            # Update status
            success = True
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)

            # Write logging info
            if self.logger is not None:
                info_str_iter = '       {:<10} {:<10} {:<15.4e} {:<15.2e} {:<15.2e}'.\
                    format(self.iteration, 0, np.mean(self.obj_func_values),
                            self.alpha, self.cov[0, 0])
                self.logger.info(info_str_iter)

            # Update iteration
            self.iteration += 1
        
        else:
            self.logger.info('Strong Wolfie conditions not satisfied')

            if iter_resamp < self.iter_resamp_max:

                self.logger.info('Resampling Gradient')
                iter_resamp += 1
                self.pk_from_ls = None

                # Shrink cov matrix
                self.cov = self.cov*self.shrink_factor

                # Recursivly call function
                success = self.calc_update(iter_resamp=iter_resamp)

            else:
                success = False
    
        return success

