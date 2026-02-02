import matplotlib.pyplot as plt
import numpy as np

class RProximalBundle(object):
    '''Riemannian proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
    This class implements the proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
    The algorithm is based on the proximal bundle method for convex optimization, which is a generalization of the bundle method for non-convex optimization.
    This implementation centers on a two-cut surrogate model, however, one may easily replace with other convex surrogates, satisfying certain properties.'''

    def __init__(self, manifold, retraction_map, transport_map, objective_function, 
                 subgradient, initial_point, initial_objective, initial_subgradient, true_min_obj = 0, retraction_error = 0,
                 transport_error = 0, sectional_curvature = -0.5, proximal_parameter = 0.02,
                 trust_parameter = 0.05,
                 max_iter = 200, tolerance = 1e-12, adaptive_proximal = False, know_minimizer = True, relative_error = True):


        # Riemannian Optimization Tools
        self.manifold = manifold
        self.retraction_map = retraction_map
        self.transport_map = transport_map

        # Parameters for first-order retraction/transport
        self.retraction_error = retraction_error
        self.transport_error = transport_error
        self.sectional_curvature = sectional_curvature

        # Explicit Optimization Oracles
        self.compute_objective = objective_function
        self.compute_subgradient = subgradient

        # State Variables
        self.current_proximal_center = initial_point # where model lives during constructions
        self.subgradient_at_center = initial_subgradient # g_{k}, subgradient at proximal center
        self.proximal_parameter = proximal_parameter # current \rho_{k}
        self.single_cut = True # flag for if the model is single-cut or two-cut

        self.initial_objective = initial_objective # f(x_{0})
        self.max_iter = max_iter # maximum number of iterations
        self.tolerance = tolerance # tolerance for convergence
        self.trust_parameter = trust_parameter # pre-specified \beta
        
        # Toggles for variants
        self.adaptive_proximal = adaptive_proximal # flag for if the proximal parameter is adaptively updated
        self.know_minimizer = know_minimizer # flag for if the true minimizer is known
        self.relative_error = relative_error # flag for if the error to store should be taken with the initial objective as a denominator

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
        self.relative_objective_history = [(initial_objective - true_min_obj) / (initial_objective - true_min_obj)]
        self.objective_history = [initial_objective - true_min_obj]  # Store gaps for visualization
        self.raw_objective_history = [initial_objective]  # Store raw objectives for algorithm logic
        self.true_min_obj = true_min_obj
        self.indices_of_descent_steps = []
        self.indices_of_null_steps = []
        self.indices_of_proximal_doubling_steps = []
        # Store intermediate points for animation (every 5 iterations)
        self.intermediate_points = [initial_point.copy()]
        self.intermediate_iterations = [0]
        

    def run(self):
        ''' run the proximal bundle algorithm'''
        for i in range(1, self.max_iter + 1):  # Use 1-based indexing like Julia
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
            
            candidate_objective = self.compute_objective(candidate_point)
            self.candidate_obj_history.append(candidate_objective)

            # cache current objective for consistent recording
            current_objective = self.compute_objective(self.current_proximal_center)

            # query new subgradient
            new_subgradient = self.compute_subgradient(candidate_point)

            # compute the model's predicted objective gap versus the true objective gap
            ratio = self.model_versus_true(candidate_objective, model_objective, current_objective)

            # Accept Rule
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

                self.indices_of_descent_steps.append(i)
                self.proximal_parameter_history.append(self.proximal_parameter)
            else:
                if self.proximal_parameter_check(candidate_direction, new_subgradient, candidate_point, model_objective) or (self.adaptive_proximal == False): # NULL STEP
                    # don't move - just update model
                    # transports subg to proximal center tangent space
                    transported_subg = self.transport_map(candidate_point, self.current_proximal_center, new_subgradient)

                    self.untransported_subgradients.append(new_subgradient) # g_{k+1}
                    self.transported_subgradients.append(transported_subg) # \hat{g}_{k+1}

                    error_shift = self.compute_shift_adjustment(new_subgradient, candidate_point)
                    self.error_shifts.append(error_shift) # conservative shift adjustment

                    self.single_cut = False # no longer single-cut model after taking steps unless we take a descent step

                    self.proximal_parameter_history.append(self.proximal_parameter)
                    self.indices_of_null_steps.append(i)
                    # print('Null Step Taken.')

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
                    # print('Proximal Parameter Doubling Step Taken.')

                    self.proximal_parameter_history.append(self.proximal_parameter)
                    self.indices_of_proximal_doubling_steps.append(i)

            # Store objective at current proximal center after each iteration (regardless of step type)
            current_proximal_objective = self.compute_objective(self.current_proximal_center)
            self.relative_objective_history.append((current_proximal_objective - self.true_min_obj) / (self.initial_objective - self.true_min_obj))
            self.objective_history.append(current_proximal_objective - self.true_min_obj)
            self.raw_objective_history.append(current_proximal_objective)

            # Store intermediate points for animation every 5 iterations
            if i % 5 == 0:
                self.intermediate_points.append(self.current_proximal_center.copy())
                self.intermediate_iterations.append(i)
        
            # check for convergence
            if self.know_minimizer:
                if abs(self.compute_objective(self.current_proximal_center) - self.true_min_obj) < self.tolerance:
                    print('Converged to true minimum.')
                    break
            else:
                # look at objective at previous descent step and descent step before that
                if len(self.indices_of_descent_steps) > 1:
                    prev_descent_step = self.indices_of_descent_steps[-1]
                    prev_prev_descent_step = self.indices_of_descent_steps[-2]

                    if abs(self.objective_history[prev_descent_step] - self.objective_history[prev_prev_descent_step]) < self.tolerance:
                        print('Converged to local minimum.')
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
            # computes the model objective using the single-cut model: f_k(d_{k}) = f(x_{k}) + \langle g_{k}, d_{k} \rangle
            return self.raw_objective_history[-1] + self.manifold.inner_product(self.current_proximal_center, self.untransported_subgradients[-1], direction)

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

    def model_versus_true(self, cand_obj, cand_model, current_obj=None):
        ''' computes the model's predicted objective gap versus the true objective gap'''

        # failsafe check
        if current_obj is None:
            current_obj = self.compute_objective(self.current_proximal_center)

        numerator = current_obj - cand_obj # computes true gap on the manifold
        denominator = current_obj - cand_model # computes model gap on the tangent space
        ratio = numerator / denominator # computes the ratio of the two gaps

        return ratio
    
    def proximal_parameter_check(self, candidate_direction, new_subgradient, candidate_point, model_objective):
        ''' compare proximal gap with shift'''

        # compute the proximal gap and shift adjustment
        shift_adjustment = self.compute_shift_adjustment(new_subgradient, candidate_point)
        model_proximal_gap = self.compute_model_proximal_gap(candidate_direction, model_objective)

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
        shift_adjustment = (np.sqrt(-self.sectional_curvature) + self.retraction_error + 2 * self.transport_error) * norm_new_subgradient * radius_of_cand**2
        return shift_adjustment
    
    def compute_model_proximal_gap(self, candidate_direction, model_objective):
        current_location_objective = self.compute_objective(self.current_proximal_center)

        # computes the proximal objective on the model
        prox_obj_on_model = model_objective + (self.proximal_parameter/2)*((self.manifold.norm(self.current_proximal_center, candidate_direction))**2)
        
        prox_gap = current_location_objective - prox_obj_on_model

        return prox_gap



## helper functions for visualizations
    def plot_objective_versus_iter(self, log_log=False):
        plt.figure(figsize=(12, 6))

        # Choose what to plot based on whether we know the minimizer
        if self.know_minimizer:
            y_data = self.objective_history  # Plot gaps when we know the minimizer
            y_label = 'Objective Gap'
            title = 'Objective Gap vs Iteration Number'
            print_label = 'Final Objective Gap'
        else:
            y_data = self.raw_objective_history  # Plot raw values when we don't know the minimizer
            y_label = 'Objective Value'
            title = 'Objective Value vs Iteration Number'
            print_label = 'Final Objective Value'

        # Plot the main line
        plt.plot(y_data, label=y_label, color='blue', linewidth=1)

        # Plot different types of steps with different colors and markers
        y_data_array = np.array(y_data)
        max_index = len(y_data_array) - 1

        if self.indices_of_descent_steps:
            valid_indices = [i for i in self.indices_of_descent_steps if i <= max_index]
            if valid_indices:
                plt.scatter(valid_indices,
                            y_data_array[valid_indices],
                            color='green', marker='o', s=8, label='Descent Steps', zorder=5)

        if self.indices_of_null_steps:
            valid_indices = [i for i in self.indices_of_null_steps if i <= max_index]
            if valid_indices:
                plt.scatter(valid_indices,
                            y_data_array[valid_indices],
                            color='orange', marker='s', s=6, label='Null Steps', zorder=5)

        if self.indices_of_proximal_doubling_steps:
            valid_indices = [i for i in self.indices_of_proximal_doubling_steps if i <= max_index]
            if valid_indices:
                plt.scatter(valid_indices,
                            y_data_array[valid_indices],
                            color='red', marker='^', s=6, label='Proximal Doubling Steps', zorder=5)

        plt.title(title + (' (Log-Log Scale)' if log_log else ''))
        plt.xlabel('Iteration Number')
        plt.ylabel(y_label)

        # Set scaling based on log_log parameter
        if log_log:
            plt.xscale('log')
            plt.yscale('log')
        else:
            plt.yscale('log')

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        print(title)
        print('----------------------------------')
        print(f'{print_label}: {y_data[-1]}')
        print('----------------------------------')
        print('Descent Steps:', len(self.indices_of_descent_steps))
        print('Null Steps:', len(self.indices_of_null_steps))
        print('Proximal Doubling Steps:', len(self.indices_of_proximal_doubling_steps))
        print('----------------------------------')

    def plot_proximal_parameter_versus_iter(self):
        ''' plot the proximal parameter versus iteration number'''
        
        raise NotImplementedError("The method 'plot_proximal_parameter_versus_iter' is not yet implemented.")