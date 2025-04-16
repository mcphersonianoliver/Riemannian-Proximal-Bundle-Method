class RProximalBundle(object):
    def __init__(self, manifold = manifold, retraction = retraction,
                 transport = transport, proximal_parameter = proximal_parameter,
                 trust_parameter = trust_parameter,
                 true_objective = true_objective, initial_point = initial_point,
                 initial_subgradient = initial_subgradient, minimizer = minimizer,
                 max_iter = max_iter, tolerance = tolerance):

        # parameters and tools
        self.manifold = manifold
        self.retraction_map = retraction 
        self.transport_map = transport

        self.current_proximal_center = initial_point
        self.proximal_parameter = proximal_parameter

        self.trust_parameter = trust_parameter
        self.objective_function = true_objective

        # storage for algorithm run
        self.objective_gap = []
        self.proximal_parameter_history = []
        self.candidate_point_history = []
        self.proximal_center_history = []
        self.indices_of_descent_steps = []
    
        # algorithm run parameters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.initial_point = initial_point
        self.initial_subgradent = initial_subgradient
        self.minimizer = minimizer
        
        # storage for aggregate cut surrogate models
        self.subgradients = [initial_subgradient]
        self.affine_shifts =[true_objective(initial_point)]

    def run(self, T):
        ''' run the proximal bundle algorithm'''
        
        for i in range(self.max_iter):
            # compute the candidate direction and convert to a point on the manifold using retraction map
            candidate_direction = self.cand_prox_direction() # IMPLEMENT
            candidate_point = self.direction_to_point(candidate_direction)
            new_subgradient = self.manifold.subgradient(candidate_point) # IMPLEMENT

            # compute the model's predicted objective gap versus the true objective gap
            ratio = self.model_versus_true(candidate_direction, candidate_point)
            if ratio > self.trust_parameter:
                self.descent_step(new_subgradient, candidate_point)
                self.indices_of_descent_steps.append(i)
                self.proximal_center_history.append(candidate_point)
            else:
                self.null_step(new_subgradient, candidate_point)
        raise NotImplementedError("The method 'run' is not yet implemented.")

## helper functions for proximal bundle algorithm run
    def cand_prox_direction(self):
        ''' compute the proximal direction in current tangent space'''

        raise NotImplementedError("The method 'proximal_direction' is not yet implemented.")

    def direction_to_point(self, candidate_direction):
        ''' conver the tangent vector to a point on the manifold'''
        candidate_point = self.retraction_map(candidate_direction)
        self.candidate_point_history.append(candidate_point)
        return candidate_point

    def model_versus_true(self, candidate_direction, candidate_point):
        ''' computes the model's predicted objective gap versus the true objective gap'''

        numerator = self.objective_function(self.proximal_center_history[-1]) - self.objective_function(candidate_point) # computes true gap on the manifold
        denominator = self.objective_function(self.proximal_center_history[-1]) - self.cut_surrogate_evaluation(candidate_direction) # computes model gap on the tangent space
        ratio = numerator / denominator # computes the ratio of the two gaps
        return ratio

    def descent_step(self, subgradient, candidate_point):
        ''' throw away previous model and start a new local surrogate at candidate iterate, move to candidate iterate'''
        # construct new surrogate at new tangent space
        self.subgradients = [subgradient]
        self.affine_shifts = [self.objective_function(candidate_point)]

        # updates new proximal center 
        self.current_proximal_center = candidate_point

    def null_step(self, subgradient, candidate_point):
        ''' incorporate subgradient information from candidate iterate to the current surrogate model, don't move'''
        
        # transport subgradient to the current proximal center
        transported_subgradient = self.transport_map(subgradient, self.current_proximal_center, candidate_point)

        # compute the affine shift of the surrogate model
        # TODO : this should be a function of the transported subgradient
        
        self.update_surrogate_model()

        
        raise NotImplementedError("The method 'null_step' is not yet implemented.")

## helper for cut surrogate model - this should be removed for generality later 
    def cut_surrogate_evaluation(self):
        ''' compute the cut surrogate model'''

        raise NotImplementedError("The method 'cut_surrogate_model' is not yet implemented.")
    
    def update_surrogate_model(self):
        ''' update the surrogate model'''

    def add_cut_two_cut_model(self):
        ''' add the cut to the surrogate model'''

        raise NotImplementedError("The method 'add_cut_two_cut_model' is not yet implemented.")

    def descent_step_cut_model(self):
        ''' throw away previous model and start a new local surrogate at candidate iterate, move to candidate iterate'''
        
        raise NotImplementedError("The method 'descent_step_cut_model' is not yet implemented.")

    def prox_step_cut_model(self):
        ''' throw away previous model and start a new local surrogate at candidate iterate, move to candidate iterate'''
        
        raise NotImplementedError("The method 'prox_step_cut_model' is not yet implemented.")


## helper functions for visualizations
    def plot_objective_gap_versus_iter(self):
        ''' plot the objective gap versus iteration number'''
        
        raise NotImplementedError("The method 'plot_objective_gap_versus_iter' is not yet implemented.")

    def plot_proximal_parameter_versus_iter(self):
        ''' plot the proximal parameter versus iteration number'''
        
        raise NotImplementedError("The method 'plot_proximal_parameter_versus_iter' is not yet implemented.")

    
        
        


        

