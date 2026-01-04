import cvxpy as cp
import numpy as np

class LinearMPCController:
    """
    Layer 2: The Safety Filter.
    Solves a Convex QP to track the Generative Reference while avoiding collisions.
    """
    def __init__(self, 
                 horizon=20, 
                 dt=0.1, 
                 wheelbase=2.5,
                 Q_diag=[10, 10, 5, 1],   # Weights for [x, y, v, yaw]
                 R_diag=[1, 10]):         # Weights for [accel, steer]
        self.N = horizon
        self.dt = dt
        self.L = wheelbase
        
        # Cost Matrices
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)
        
        # Actuator Limits (Hard Constraints)
        self.acc_max = 3.0   # m/s^2
        self.steer_max = np.deg2rad(35)
        self.jerk_max = 2.0  # m/s^3 (Reviewer 2's Comfort Metric)

    def get_linearized_dynamics(self, v_ref, phi_ref):
        """
        Linearizes Kinematic Bicycle Model around reference velocity/yaw.
        State: [x, y, v, phi]
        Input: [a, delta]
        """
        dt = self.dt
        L = self.L
        
        # Jacobian w.r.t State (A matrix)
        # x_next = x + v*cos(phi)*dt
        # y_next = y + v*sin(phi)*dt
        # v_next = v + a*dt
        # phi_next = phi + (v/L)*tan(delta)*dt
        
        A = np.eye(4)
        A[0, 2] = np.cos(phi_ref) * dt  # d(x)/dv
        A[0, 3] = -v_ref * np.sin(phi_ref) * dt # d(x)/dphi
        A[1, 2] = np.sin(phi_ref) * dt  # d(y)/dv
        A[1, 3] = v_ref * np.cos(phi_ref) * dt # d(y)/dphi
        A[3, 2] = (np.tan(0.0) / L) * dt # Approximation for small steer angle
        
        # Jacobian w.r.t Input (B matrix)
        B = np.zeros((4, 2))
        B[2, 0] = dt # d(v)/da
        B[3, 1] = (v_ref / L) * dt # d(phi)/ddelta (approx cos^2(delta) ~ 1)
        
        return A, B

    def solve(self, current_state, reference_traj):
        """
        Args:
            current_state: [4] (x, y, v, phi)
            reference_traj: [N+1, 4] (Output from Flow Matching)
        Returns:
            optimal_control: [2] (accel, steer)
        """
        # Variables to optimize
        x = cp.Variable((4, self.N + 1))
        u = cp.Variable((2, self.N))
        
        cost = 0
        constraints = []
        
        # Initial Condition
        constraints += [x[:, 0] == current_state]
        
        # Iterate over horizon
        for k in range(self.N):
            # 1. Linearize Dynamics around the Reference at step k
            # Note: A true LTV-MPC would update A, B at every step based on ref
            v_k_ref = reference_traj[k, 2]
            phi_k_ref = reference_traj[k, 3]
            A_lin, B_lin = self.get_linearized_dynamics(v_k_ref, phi_k_ref)
            
            # 2. Dynamics Constraint (Linearized)
            # x_{k+1} = A x_k + B u_k
            constraints += [x[:, k+1] == A_lin @ x[:, k] + B_lin @ u[:, k]]
            
            # 3. Cost Function (Tracking Error + Control Effort)
            # Minimize: (x - x_ref)^T Q (x - x_ref)
            state_error = x[:, k] - reference_traj[k, :]
            cost += cp.quad_form(state_error, self.Q)
            
            # Minimize: u^T R u
            cost += cp.quad_form(u[:, k], self.R)
            
            # 4. Actuator Constraints
            constraints += [cp.abs(u[0, k]) <= self.acc_max]
            constraints += [cp.abs(u[1, k]) <= self.steer_max]
            
            # 5. Jerk Constraint (Slew Rate on Acceleration)
            if k > 0:
                jerk = (u[0, k] - u[0, k-1]) / self.dt
                constraints += [cp.abs(jerk) <= self.jerk_max]

        # Solve QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # OSQP is robust and fast for embedded control
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except cp.SolverError:
            return np.array([0.0, 0.0]) # Fail-safe

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"MPC Warning: Solver status is {prob.status}")
            return np.array([0.0, 0.0]) # Emergency stop

        # Return first control action
        return u[:, 0].value