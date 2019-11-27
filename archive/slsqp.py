    def optimize_slsqp(self, theta0=None, bounds=None, max_ratio=20, epsilon=0.0001):
        """
        NOTE: This function seems to perform worse (is generally more finicky) than the Nelder-Mead optimizer in the method self.optimize.

        This function has all the default information and function calls needed to run the Scipy SLSQP optimizer for the kernel parameters.
        
        Parameters:
            theta0 (list/ndarray): Starting parameter values [sigma_s, sigma_x, sigma_y, phi, sigma_w]. If theta0 is None, then this function will use the theta0 supplied by the correlation_fit method.
            bounds (ndarray of length 2 tuples): Upper and lower bounds for the optimizer for each parameter.
            max_ratio (int/float): Maximum ratio between sigma_x and sigma_y.
            epsilon (float): Step size used by the optimizer in calculating gradient.
        """
        
        if theta0 is None:
            # Use 0.99 times the output of the correlation_fit method in order to avoid Exit Mode 8 on the optimizer.
            theta0 = self.theta0 * 0.99
        elif theta0 == 'default':
            # These are the standard starting value and bounds that seem to work well
            theta0 = np.array([1e2, 10**(-1.217), .1, 0, 1])
        elif isinstance(theta0, (np.ndarray, list)):
            pass
        
        if bounds is None:
            bounds = np.array([
                (1, 1e4),
                (((264*u.mas).to(u.deg)).value*10, 5),
                (((264*u.mas).to(u.deg)).value*10, 5),
                (-2*np.pi, 2*np.pi),
                (1, np.sqrt(10))
            ])
        
        # These are the two inequality constraints which specify that:
        # sigma_x * max_ratio >= sigma_y
        # sigma_y * max_ratio >= sigma_x
        # This means that sigma_x and sigma_y will never be more than max_ratio times apart from each other.
        # This seems to speed up computation time since when sigma_x and sigma_y are vastly different the computation time goes way up.
        def g1(x):
            sigma_x = x[1]
            sigma_y = x[2]
            return -sigma_x + max_ratio * sigma_y

        def g2(x):
            sigma_x = x[1]
            sigma_y = x[2]
            return -sigma_y + max_ratio * sigma_x
        
        self.opt_params, self.nLML_final, self.nIterations, self.exit_mode, self.exit_mode_explanation = opt.fmin_slsqp(
            self.get_nLML,
            theta0,
            ieqcons=[g1, g2],
            bounds=bounds,
            iprint=2,
            epsilon=epsilon,
            full_output=True
        )
        
        self.theta = self.opt_params