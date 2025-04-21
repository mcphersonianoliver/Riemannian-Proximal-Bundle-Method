import numpy as np

class RProximalBundle(object):
    '''Riemannian proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
    This class implements the proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
    The algorithm is based on the proximal bundle method for convex optimization, which is a generalization of the bundle method for non-convex optimization.
    This implementation centers on a two-cut surrogate model, however, one may easily replace with other convex surrogates, satisfying certain properties.'''

    def __init__(self, manifold, retraction, retraction_error = 0,
                 transport_map = transport_map, transport_error = 0, proximal_parameter = 1,
                 trust_parameter = 0.1,
                 objective_function, initial_point = initial_point, initial_objective = initial_objective,
                 initial_subgradient = initial_subgradient, minimizer = minimizer,
                 max_iter = max_iter, tolerance = tolerance):

        # parameters and tools
        self.manifold = manifold
        self.retraction_map = retraction 
        self.transport_map = transport_map

        self.current_proximal_center = initial_point
        self.proximal_parameter = proximal_parameter

        self.trust_parameter = trust_parameter
        self.objective_function = objective_function
        self.initial_objective = initial_objective

        # flag for if the current surrogate is a single-cut (first iterate at the proximal center)
        self.single_cut = True

        # parameters for first-order retraction/transport
        self.retraction_error = retraction_error
        self.transport_error = transport_error
        
        self.subgradient_at_center = initial_subgradient

        # storage for two-cut model - setting aside memory
        self.model_subg = [initial_subgradient] # s_{k}, initialize together to line-up indexes, dummy entry
        self.untransported_subgradients = [initial_subgradient] # g_{k}
        self.transported_subgradients = [initial_subgradient] # \hat{g}_{k}
        self.candidate_directions = [] # d_{k}
        self.candidate_obj_history = [initial_objective] # f(R_x(x_{k}))
        self.candidate_model_obj_history = [] # f_k(d_{k})
        self.error_shifts = [0] # e_{f_k}(\rho_{k})

        # storage for algorithm run
        self.proximal_parameter_history = []
        self.indices_of_descent_steps = []
    
        # algorithm run parameters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.initial_point = initial_point
        self.initial_subgradient = initial_subgradient
        self.minimizer = minimizer
        

    def run(self, T):
        ''' run the proximal bundle algorithm'''
        for i in range(self.max_iter):
            # compute the candidate direction and convert to a point on the manifold using retraction map
            candidate_direction = self.cand_prox_direction() 
            self.candidate_directions.append(candidate_direction)
            candidate_point = self.direction_to_point(candidate_direction) # unnecessary

            # compute true objective and predicted objective
            true_objective = self.objective_function(candidate_point)
            self.candidate_obj_history.append(true_objective)
            model_objective = self.model_evaluation(candidate_direction, i) # check index
            self.candidate_model_obj_history.append(model_objective)
            
            # query new subgradient
            new_subgradient = self.manifold.subgradient(candidate_point) # TODO: IMPLEMENT

            self.single_cut = False # no longer single-cut model after taking steps

            # compute the model's predicted objective gap versus the true objective gap
            ratio = self.model_versus_true(candidate_direction, candidate_point, i)

            if ratio > self.trust_parameter: # DESCENT STEP
                self.current_proximal_center = candidate_point # moves model

                # update proximal center and set initial subgradient 
                self.current_proximal_center = candidate_point   
                self.subgradient_at_center = new_subgradient # updates subgradient at center

                # update model information - one-cut model now!
                self.untransported_subgradients.append(new_subgradient) # g_{k+1}
                self.transported_subgradients.append(new_subgradient) # no transport is done
                self.model_subg.append(new_subgradient) # one-cut model - placeholder for index purposes
                self.error_shifts.append(0) # e_{f_k}(\rho_{k+1}) = 0, no transport is done
                
                # for visualization
                self.indices_of_descent_steps.append(i)
                self.proximal_parameter_history.append(self.proximal_parameter)
            else:
                if self.proximal_parameter_check(candidate_direction): # NULL STEP 
                    # don't move - just update model

                    self.untransported_subgradients.append(new_subgradient) # g_{k+1}
                    self.transported_subgradients.append(self.transport_map(new_subgradient, self.current_proximal_center)) # \hat{g}_{k+1}
                    
                    model_subg = 1 # TODO: implement this
                    self.model_subg.append(model_subg) # one-cut model - placeholder for index purposes
                    
                    error_shift = self.compute_shift_adjustment(new_subgradient)
                    self.error_shifts.append(error_shift) # conservative shift adjustment

                    # for visualization
                    self.proximal_parameter_history.append(self.proximal_parameter)

                else: # PROXIMAL PARAMETER DOUBLING STEP
                    self.proximal_parameter *= 2 # double proximal parameter

                    # don't update model and don't move - repeat previous model
                    prev_untransport_subg = self.untransported_subgradients[-1] # g_{k+1}
                    prev_transport_subg = self.transported_subgradients[-1] # \hat{g}_{k+1}
                    prev_model_subg = self.model_subg[-1] # s_{k+1}
                    prev_error_shift = self.error_shifts[-1] # e_{f_k}(\rho_{k+1})

                    self.untransported_subgradients.append(prev_untransport_subg) # g_{k+1}
                    self.transported_subgradients.append(prev_transport_subg) # \hat{g}_{k+1}
                    self.model_subg.append(prev_model_subg) # s_{k+1}
                    self.error_shifts.append(prev_error_shift) # e_{f_k}(\rho_{k+1})

                    # for visualization
                    self.proximal_parameter_history.append(self.proximal_parameter)

## helper functions for proximal bundle algorithm run
    def model_evaluation(self, direction, iter):
        ''' compute the cut surrogate model
        Paramters
        ----------
        direction : np.array
            tangent vector on the manifold to be assessed
        iter : int
            which model to be used for evaluation
            
        Returns
        -------
        model_objective : float
            the objective value of the model at the candidate point'''
        # TODO: CURRENT VERSION ASSUMES TWO-CUT MODEL
        # recalls model information for given iteration
        candidate_dir = self.candidate_directions[iter]
        transp_subg = self.transported_subgradients[iter]
        mod_subg = self.prev_model_info[iter]
        mod_shift = self.error_shifts[iter]
        true_obj_cand = self.candidate_obj_history[iter]
        model_obj_cand = self.candidate_model_obj_history[iter]

        # computes on "new" cut
        affine_new_shift = true_obj_cand - np.dot(transp_subg, candidate_dir) - mod_shift
        new_inner = np.dot(transp_subg, direction)
        new_cut_obj = affine_new_shift + new_inner 
        
        # computes on "old" cut
        affine_old_shift = model_obj_cand - np.dot(mod_subg, candidate_dir)
        old_inner = np.dot(mod_subg, direction)
        old_cut_obj = affine_old_shift + old_inner

        # returns the maximum of the two cuts
        return max(new_cut_obj, old_cut_obj)


    def cand_prox_direction(self):
        ''' compute proximal direction due to model '''

        if self.single_cut:
            # computes the proximal direction using the single-cut model: - (g_{k}/\rho_{k})
            return - (self.untransported_subgradients[-1]/self.proximal_parameter)
        else:
            # computes form from Lemma 10 of paper - compute \theta_{t+1}
            prev_true_obj_cand = self.candidate_obj_history[-1]
            prev_error_shift = self.error_shifts[-1]
            prev_model_obj_cand = self.candidate_model_obj_history[-1]
            prev_model_subg = self.model_subg[-1]
            prev_transported_subg = self.transported_subgradients[-1]
            numerator = self.proximal_parameter * (prev_true_obj_cand - prev_error_shift - prev_model_obj_cand)
            denominator = np.linalg.norm(prev_transported_subg - prev_model_subg)**2
            convex_comb_arg = numerator / denominator
            convex_comb = min(1, convex_comb_arg)

            # computes the proxmial direction - convex combination of two subg
            return - ((convex_comb * prev_transported_subg + (1 - convex_comb) * prev_model_subg) / self.proximal_parameter)

    def direction_to_point(self, candidate_direction):
        ''' conver the tangent vector to a point on the manifold'''
        candidate_point = self.retraction_map(candidate_direction)
        self.candidate_point_history.append(candidate_point)
        return candidate_point

    def model_versus_true(self, candidate_direction, candidate_point, iter):
        ''' computes the model's predicted objective gap versus the true objective gap'''

        numerator = self.objective_function(self.proximal_center_history[-1]) - self.objective_function(candidate_point) # computes true gap on the manifold
        denominator = self.objective_function(self.proximal_center_history[-1]) - self.model_evaluation(candidate_direction, iter) # computes model gap on the tangent space
        ratio = numerator / denominator # computes the ratio of the two gaps
        return ratio
    
    def proximal_parameter_check(self, candidate_direction, new_subgradient):
        ''' compare proximal gap with shift'''

        # compute the proximal gap and shift adjustment
        shift_adjustment = self.compute_shift_adjustment(new_subgradient)
        model_proximal_gap = self.compute_model_proximal_gap(candidate_direction)

        check_value = 0.5*model_proximal_gap - (shift_adjustment / (1-self.trust_parameter))

        if check_value >= 0:
            return True
        else:
            return False

    def compute_shift_adjustment(self, new_subgradient):
        # compute relevant subgradient norms
        norm_subgradient_center = np.linalg.norm(self.subgradient_at_center)
        norm_new_subgradient = np.linalg.norm(new_subgradient)

        radius_of_cand = ((2 * norm_subgradient_center) / self.proximal_parameter) + self.retraction_error * (((2 * norm_subgradient_center) / self.proximal_parameter)**2)
        shift_adjustment = self.transport_error * norm_new_subgradient * radius_of_cand**2
        return shift_adjustment
    
    def compute_model_proximal_gap(self, candidate_direction):
        current_location_objective = self.objective_function(self.proximal_center_history[-1])
        prox_obj_on_model = self.model_evaluation(candidate_direction) + (self.proximal_parameter/2)*(np.linalg.norm(candidate_direction)**2)
        
        prox_gap = current_location_objective - prox_obj_on_model
        return prox_gap



## helper functions for visualizations
    def plot_objective_gap_versus_iter(self):
        ''' plot the objective gap versus iteration number'''
        
        raise NotImplementedError("The method 'plot_objective_gap_versus_iter' is not yet implemented.")

    def plot_proximal_parameter_versus_iter(self):
        ''' plot the proximal parameter versus iteration number'''
        
        raise NotImplementedError("The method 'plot_proximal_parameter_versus_iter' is not yet implemented.")