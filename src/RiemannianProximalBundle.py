import matplotlib.pyplot as plt
import numpy as np

class RProximalBundle(object):
    '''Riemannian proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
    This class implements the proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
    The algorithm is based on the proximal bundle method for convex optimization, which is a generalization of the bundle method for non-convex optimization.
    This implementation centers on a two-cut surrogate model, however, one may easily replace with other convex surrogates, satisfying certain properties.'''

    def __init__(self, manifold, retraction_map, transport_map, objective_function, 
                 subgradient, initial_point, initial_objective, initial_subgradient, true_min_obj = 0, retraction_error = 0,
                 transport_error = 0, proximal_parameter = 0.5,
                 trust_parameter = 0.9,
                 max_iter = 200, tolerance = 1e-12):

        # parameters and tools
        self.manifold = manifold
        self.retraction_map = retraction_map
        self.transport_map = transport_map

        self.current_proximal_center = initial_point # initialization 
        self.proximal_parameter = proximal_parameter
        self.trust_parameter = trust_parameter
        self.initial_objective = initial_objective
        self.max_iter = max_iter
        self.tolerance = tolerance

        # functions to compute the objective and subgradient
        self.compute_objective = objective_function
        self.compute_subgradient = subgradient

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
        self.proximal_center_history = [initial_point] # x_{k}

        # model information for two-cut model evaluation
        self.prev_candidate_direction = None
        self.prev_true_obj = None
        self.prev_model_obj = None
        self.prev_transport_subg = None
        self.prev_model_subg = None
        self.prev_error_shift = None
        self.prev_prox_parameter = proximal_parameter

        # storage for algorithm run
        self.proximal_parameter_history = [proximal_parameter]
        self.indices_of_descent_steps = []
        self.objective_history = [initial_objective]
        self.true_min_obj = true_min_obj
        

    def run(self):
        ''' run the proximal bundle algorithm'''
        for i in range(self.max_iter):
            print('')
            print('Iteration:', i)
            # fix two-cut model information before any updates
            self.prev_candidate_direction = self.candidate_directions[-1] if len(self.candidate_directions) > 0 else None
            self.prev_true_obj = self.candidate_obj_history[-1] if len(self.candidate_obj_history) > 0 else None
            self.prev_model_obj = self.candidate_model_obj_history[-1] if len(self.candidate_model_obj_history) > 0 else None
            self.prev_transport_subg = self.transported_subgradients[-1] if len(self.transported_subgradients) > 0 else None
            self.prev_model_subg =  -self.prev_prox_parameter * self.prev_candidate_direction if self.prev_candidate_direction is not None else None
            self.model_subg.append(self.prev_model_subg) # s_{k+1} = - \rho_{k+1} d_{k+1}
            self.prev_error_shift = self.error_shifts[-1] if len(self.error_shifts) > 0 else None

            # compute the candidate direction and convert to a point on the manifold using retraction map
            candidate_direction = self.cand_prox_direction() 

            self.candidate_directions.append(candidate_direction)
            candidate_point = self.retraction_map(self.current_proximal_center, candidate_direction) # retraction map to get point on manifold

            # compute true objective and predicted objective
            model_objective = self.model_evaluation(candidate_direction) 
            
            self.candidate_model_obj_history.append(model_objective)
            true_objective = self.compute_objective(candidate_point)
            self.candidate_obj_history.append(true_objective)

            
            # query new subgradient
            new_subgradient = self.compute_subgradient(candidate_point) 

            # compute the model's predicted objective gap versus the true objective gap
            ratio = self.model_versus_true(true_objective, model_objective)

            if ratio > self.trust_parameter: # DESCENT STEP
                self.current_proximal_center = candidate_point # moves model
                self.proximal_center_history.append(candidate_point) # stores the new proximal center

                # update proximal center and set initial subgradient 
                self.current_proximal_center = candidate_point   
                self.subgradient_at_center = new_subgradient # updates subgradient at center

                # update model information - one-cut model now!
                self.untransported_subgradients.append(new_subgradient) # g_{k+1}
                self.transported_subgradients.append(new_subgradient) # no transport is done
                self.error_shifts.append(0) # e_{f_k}(\rho_{k+1}) = 0, no transport is done

                self.single_cut = True

                self.objective_history.append(self.compute_objective(self.current_proximal_center))
                print('Descent Step Taken.')
                
                # for visualization
                self.indices_of_descent_steps.append(i)
                self.proximal_parameter_history.append(self.proximal_parameter)
            else:
                if self.proximal_parameter_check(candidate_direction, new_subgradient, candidate_point): # NULL STEP 
                    # don't move - just update model
                    # transports subg to proximal center tangent space
                    transported_subg = self.transport_map(candidate_point, self.current_proximal_center, new_subgradient) 

                    self.untransported_subgradients.append(new_subgradient) # g_{k+1}
                    self.transported_subgradients.append(transported_subg) # \hat{g}_{k+1}
                    
                    error_shift = self.compute_shift_adjustment(new_subgradient, candidate_point)
                    self.error_shifts.append(error_shift) # conservative shift adjustment

                    self.single_cut = False # no longer single-cut model after taking steps unless we take a descent step

                    # for visualization
                    self.objective_history.append(self.compute_objective(self.current_proximal_center))
                    self.proximal_parameter_history.append(self.proximal_parameter)
                    print('Null Step Taken.')

                else: # PROXIMAL PARAMETER DOUBLING STEP
                    self.prev_prox_parameter = self.proximal_parameter
                    self.proximal_parameter *= 2 # double proximal parameter

                    # don't update model and don't move - repeat previous model
                    prev_untransport_subg = self.untransported_subgradients[-1] # g_{k+1}
                    prev_transport_subg = self.transported_subgradients[-1] # \hat{g}_{k+1}
                    prev_error_shift = self.error_shifts[-1] # e_{f_k}(\rho_{k+1})

                    self.untransported_subgradients.append(prev_untransport_subg) # g_{k+1}
                    self.transported_subgradients.append(prev_transport_subg) # \hat{g}_{k+1}
                    self.error_shifts.append(prev_error_shift) # e_{f_k}(\rho_{k+1})
                    print('Proximal Parameter Doubling Step Taken.')

                    # for visualization
                    self.objective_history.append(self.compute_objective(self.current_proximal_center))
                    self.proximal_parameter_history.append(self.proximal_parameter)
        
            # check for convergence
            if abs(self.objective_history[-1] - self.true_min_obj) < self.tolerance:
                print('Converged to true minimum.')
                break

## helper functions for proximal bundle algorithm run
    def model_evaluation(self, direction):
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
        if self.single_cut:
            # computes the model objective using the single-cut model: f_k(d_{k}) = f_k(x_{k}) + \langle g_{k}, d_{k} \rangle
            return self.candidate_obj_history[-1] + self.manifold.inner_product(self.current_proximal_center, self.untransported_subgradients[-1], direction)

        # computes on "new" cut
        affine_new_shift = self.prev_true_obj -  self.manifold.inner_product(self.current_proximal_center, self.prev_transport_subg, self.prev_candidate_direction) - self.prev_error_shift
        new_inner =  self.manifold.inner_product(self.current_proximal_center, self.prev_transport_subg, direction)
        new_cut_obj = affine_new_shift + new_inner 
        
        # computes on "old" cut
        affine_old_shift = self.prev_model_obj -  self.manifold.inner_product(self.current_proximal_center, self.prev_model_subg, self.prev_candidate_direction)
        old_inner =  self.manifold.inner_product(self.current_proximal_center, self.prev_model_subg, direction)
        old_cut_obj = affine_old_shift + old_inner

        # returns the maximum of the two cuts
        return max(new_cut_obj, old_cut_obj)


    def cand_prox_direction(self):
        ''' compute proximal direction due to model '''
        if self.single_cut:
            # computes the proximal direction using the single-cut model: - (g_{k}/\rho_{k})
            return - (self.untransported_subgradients[-1]/self.proximal_parameter)
        else:
            # use stored two-cut model information to compute proximal direction
            numerator = self.proximal_parameter * (self.prev_true_obj - self.prev_error_shift - self.prev_model_obj)
            denominator = (self.manifold.norm(self.current_proximal_center, self.prev_transport_subg - self.prev_model_subg))**2
            
            convex_comb_arg = numerator / denominator
            convex_comb = min(1, convex_comb_arg)

            # computes the proxmial direction - convex combination of two subg
            return - (1 / self.proximal_parameter) * (convex_comb * self.prev_transport_subg + (1 - convex_comb) * self.prev_model_subg)

    def model_versus_true(self, cand_obj, cand_model):
        ''' computes the model's predicted objective gap versus the true objective gap'''

        numerator = self.compute_objective(self.current_proximal_center) - cand_obj # computes true gap on the manifold
        denominator = self.compute_objective(self.current_proximal_center) - cand_model # computes model gap on the tangent space
        ratio = numerator / denominator # computes the ratio of the two gaps

        return ratio
    
    def proximal_parameter_check(self, candidate_direction, new_subgradient, candidate_point):
        ''' compare proximal gap with shift'''

        # compute the proximal gap and shift adjustment
        shift_adjustment = self.compute_shift_adjustment(new_subgradient, candidate_point)
        model_proximal_gap = self.compute_model_proximal_gap(candidate_direction)

        check_value = 0.5*model_proximal_gap - (shift_adjustment / (1-self.trust_parameter))

        if check_value >= 0:
            return True
        else:
            return False

    def compute_shift_adjustment(self, new_subgradient, candidate_point):
        # compute relevant subgradient norms
        norm_subgradient_center = self.manifold.norm(self.current_proximal_center, self.subgradient_at_center)
        norm_new_subgradient = self.manifold.norm(candidate_point, new_subgradient)

        radius_of_cand = ((2 * norm_subgradient_center) / self.proximal_parameter) + self.retraction_error * (((2 * norm_subgradient_center) / self.proximal_parameter)**2)
        shift_adjustment = self.transport_error * norm_new_subgradient * radius_of_cand**2
        return shift_adjustment
    
    def compute_model_proximal_gap(self, candidate_direction):
        current_location_objective = self.compute_objective(self.proximal_center_history[-1])
        prox_obj_on_model = self.model_evaluation(candidate_direction) + (self.proximal_parameter/2)*((self.manifold.norm(self.current_proximal_center, candidate_direction))**2)
        
        prox_gap = current_location_objective - prox_obj_on_model
        return prox_gap



## helper functions for visualizations
    def plot_objective_versus_iter(self):
        ''' plot the objective gap versus iteration number'''
        # Create a plot of objective gap versus iteration number

        plt.figure(figsize=(10, 6))
        plt.plot(self.objective_history - self.true_min_obj, label='Objective Gap')
        plt.scatter(self.indices_of_descent_steps, np.array(self.objective_history)[self.indices_of_descent_steps] - self.true_min_obj, color='red', label='Descent Steps')
        plt.title('Objective Gap vs Iteration Number')
        plt.xlabel('Iteration Number')
        plt.ylabel('Objective Gap')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.show()
        print('Objective Gap vs Iteration Number')
        print('----------------------------------')
        print('Objective Gap:', self.objective_history[-1] - self.true_min_obj)
        print('----------------------------------')
        print('Descent Steps:', len(self.indices_of_descent_steps))
        print('----------------------------------')

    def plot_proximal_parameter_versus_iter(self):
        ''' plot the proximal parameter versus iteration number'''
        
        raise NotImplementedError("The method 'plot_proximal_parameter_versus_iter' is not yet implemented.")