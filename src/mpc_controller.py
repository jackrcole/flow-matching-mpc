import cvxpy as cp
import numpy as np

class LinearMPCController:
    """
    Layer 2: The Safety Filter.
    Solves a Convex QP to track the Reference while enforcing limits and collision constraints.
    """
    def __init__(self, 
                 horizon=20, 
                 dt=0.1, 
                 wheelbase=2.5,
                 Q_diag=[10, 10, 5, 1],   
                 R_diag=[1, 10],
                 safe_distance=2.0):
        self.N = horizon
        self.dt = dt
        self.L = wheelbase
        self.safe_dist = safe_distance
        
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        
        self.acc_max = 3.0   
        self.steer_max = np.deg2rad(35)
        self.jerk_max = 2.0 

    def get_linearized_dynamics(self, v_ref, phi_ref):
        dt = self.dt
        L = self.L
        
        A = np.eye(4)
        A[0, 2] = np.cos(phi_ref) * dt  
        A[0, 3] = -v_ref * np.sin(phi_ref) * dt
        A[1, 2] = np.sin(phi_ref) * dt  
        A[1, 3] = v_ref * np.cos(phi_ref) * dt
        A[3, 2] = (np.tan(0.0) / L) * dt 
        
        B = np.zeros((4, 2))
        B[2, 0] = dt 
        B[3, 1] = (v_ref / L) * dt 
        
        return A, B

    def get_separating_hyperplane(self, x_ref, y_ref, obs_x, obs_y):
        """
        Computes the linear constraint A_obs * x <= b_obs.
        """
        dx = x_ref - obs_x
        dy = y_ref - obs_y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Avoid singularity if reference is EXACTLY inside/on obstacle
        if dist < 0.01:
            dx, dy = 1.0, 0.0 # Default to pushing back
            dist = 1.0

        nx = dx / dist
        ny = dy / dist
        
        # Constraint: -nx*x - ny*y <= -(safe + nx*obs_x + ny*obs_y)
        A_row = np.array([-nx, -ny, 0, 0]) 
        b_scalar = -(self.safe_dist + nx * obs_x + ny * obs_y)
        
        return A_row, b_scalar

    def solve(self, current_state, reference_traj, obstacles=[]):
        x = cp.Variable((4, self.N + 1))
        u = cp.Variable((2, self.N))
        
        cost = 0
        constraints = []
        constraints += [x[:, 0] == current_state]
        
        for k in range(self.N):
            # 1. Dynamics
            v_ref = reference_traj[k, 2]
            phi_ref = reference_traj[k, 3]
            A_lin, B_lin = self.get_linearized_dynamics(v_ref, phi_ref)
            constraints += [x[:, k+1] == A_lin @ x[:, k] + B_lin @ u[:, k]]
            
            # 2. Cost
            state_error = x[:, k] - reference_traj[k, :]
            cost += cp.quad_form(state_error, self.Q)
            cost += cp.quad_form(u[:, k], self.R)
            
            # 3. Actuator Limits
            constraints += [cp.abs(u[0, k]) <= self.acc_max]
            constraints += [cp.abs(u[1, k]) <= self.steer_max]
            
            # 4. Collision Constraints
            x_ref_k = reference_traj[k, 0]
            y_ref_k = reference_traj[k, 1]
            
            for obs in obstacles:
                obs_x, obs_y, obs_r = obs
                # Use geometry to define the "Safe Side" of the line
                A_obs, b_obs = self.get_separating_hyperplane(x_ref_k, y_ref_k, obs_x, obs_y)
                constraints += [A_obs @ x[:, k] <= b_obs]

            # 5. Jerk Limits
            if k > 0:
                jerk = (u[0, k] - u[0, k-1]) / self.dt
                constraints += [cp.abs(jerk) <= self.jerk_max]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except cp.SolverError:
            return np.array([0.0, 0.0])

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return np.array([-self.acc_max, 0.0]) 

        return u[:, 0].value