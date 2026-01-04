import cvxpy as cp
import numpy as np

class LinearMPCController:
    """
    Layer 2: The Safety Filter (Updated with Collision Avoidance).
    Solves a Convex QP to track the Generative Reference while enforcing:
    1. Kinematics (Linearized)
    2. Actuator Limits
    3. Collision Avoidance (Linear Separating Hyperplanes)
    """
    def __init__(self, 
                 horizon=20, 
                 dt=0.1, 
                 wheelbase=2.5,
                 Q_diag=[10, 10, 5, 1],   
                 R_diag=[1, 10],
                 safe_distance=2.0):      # Min distance (Ego Radius + Obs Radius)
        self.N = horizon
        self.dt = dt
        self.L = wheelbase
        self.safe_dist = safe_distance
        
        # Diagonal Cost Matrices
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        
        # Hard Constraints
        self.acc_max = 3.0   
        self.steer_max = np.deg2rad(35)
        self.jerk_max = 2.0 

    def get_linearized_dynamics(self, v_ref, phi_ref):
        """Linearizes Kinematic Bicycle Model around reference."""
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
        Computes the linear constraint A_obs * [x, y] <= b_obs.
        Includes a 'Lookahead Nudge' to break symmetry for obstacles directly ahead.
        """
        dx = x_ref - obs_x
        dy = y_ref - obs_y
        dist = np.sqrt(dx**2 + dy**2)

        # --- ROBUSTNESS FIX ---
        # If obstacle is roughly ahead (dist > 0) but we are perfectly aligned (dy ~ 0),
        # the solver sees a flat wall and will just brake.
        # We artificially shift 'dy' to simulate the reference being slightly to the side.
        # This angles the constraint hyperplane, encouraging a swerve.
        
        if dist > 0.1 and abs(dy) < 0.5: 
            # Force a "Right Swerve" bias
            # We pretend the reference path is actually 2.0m to the right
            dy = -2.0 
            # Re-calculate distance and normal with this bias
            dist = np.sqrt(dx**2 + dy**2)
        
        # Handle the "Inside Obstacle" singularity (dist ~ 0)
        elif dist <= 0.1:
            dx = 1.0
            dy = -1.0 # Push right
            dist = np.sqrt(dx**2 + dy**2)
        # ----------------------

        # Normalize
        nx = dx / dist
        ny = dy / dist
        
        # Constraint form: -n_x * x - n_y * y <= - (safe_dist + n_x*obs_x + n_y*obs_y)
        A_row = np.array([-nx, -ny, 0, 0]) 
        b_scalar = -(self.safe_dist + nx * obs_x + ny * obs_y)
        
        return A_row, b_scalar

    def solve(self, current_state, reference_traj, obstacles=[]):
        """
        Args:
            current_state: [4] (x, y, v, phi)
            reference_traj: [N+1, 4] Output from Flow Matching
            obstacles: List of tuples [(x, y, radius), ...]
        """
        # Optimization Variables
        x = cp.Variable((4, self.N + 1))
        u = cp.Variable((2, self.N))
        
        cost = 0
        constraints = []
        
        # Initial Condition
        constraints += [x[:, 0] == current_state]
        
        for k in range(self.N):
            # 1. Dynamics
            v_ref = reference_traj[k, 2]
            phi_ref = reference_traj[k, 3]
            A_lin, B_lin = self.get_linearized_dynamics(v_ref, phi_ref)
            constraints += [x[:, k+1] == A_lin @ x[:, k] + B_lin @ u[:, k]]
            
            # 2. Cost (Tracking + Effort)
            state_error = x[:, k] - reference_traj[k, :]
            cost += cp.quad_form(state_error, self.Q)
            cost += cp.quad_form(u[:, k], self.R)
            
            # 3. Actuator Limits
            constraints += [cp.abs(u[0, k]) <= self.acc_max]
            constraints += [cp.abs(u[1, k]) <= self.steer_max]
            
            # 4. Collision Avoidance (NEW Logic)
            x_ref_k = reference_traj[k, 0]
            y_ref_k = reference_traj[k, 1]
            
            for obs in obstacles:
                obs_x, obs_y, obs_r = obs
                
                # Compute hyperplane separating reference from obstacle
                A_obs, b_obs = self.get_separating_hyperplane(x_ref_k, y_ref_k, obs_x, obs_y)
                
                # Add Linear Inequality: A_obs * x[:, k] <= b_obs
                constraints += [A_obs @ x[:, k] <= b_obs]

            # 5. Jerk Limits
            if k > 0:
                jerk = (u[0, k] - u[0, k-1]) / self.dt
                constraints += [cp.abs(jerk) <= self.jerk_max]

        # Solve QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # Use OSQP solver
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except cp.SolverError:
            return np.array([0.0, 0.0])

        # Check Status
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # If problem is infeasible (collision unavoidable), brake hard
            return np.array([-self.acc_max, 0.0]) 

        return u[:, 0].value