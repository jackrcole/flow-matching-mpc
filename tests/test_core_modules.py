import unittest
import torch
import numpy as np
import sys
import os

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from vectornet import VectorNetEncoder
from flow_matching import FlowMatchingActionExpert
from mpc_controller import LinearMPCController

class TestGenerativeStack(unittest.TestCase):
    """
    Tests for Layer 1: The 'Brain' (VectorNet + Flow Matching)
    """

    def setUp(self):
        """Runs before every test method. Initializes dimensions."""
        self.batch_size = 2
        self.state_dim = 6  # [x, y, z, dx, dy, type]
        self.hidden_dim = 128
        self.horizon = 20
        self.device = torch.device('cpu') # Keep on CPU for CI/Unit Testing

    def test_vectornet_dimensions(self):
        """
        Verifies VectorNet accepts ragged inputs and outputs correct context embedding.
        """
        # 1. Create Dummy Input (Map + Agents)
        N_map, P_map = 10, 20
        N_agent, P_agent = 5, 20
        
        map_polylines = torch.randn(self.batch_size, N_map, P_map, self.state_dim)
        map_mask = torch.ones(self.batch_size, N_map, P_map)
        agent_polylines = torch.randn(self.batch_size, N_agent, P_agent, self.state_dim)
        agent_mask = torch.ones(self.batch_size, N_agent, P_agent)

        # 2. Initialize Model
        model = VectorNetEncoder(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        
        # 3. Forward Pass
        context_emb = model(map_polylines, map_mask, agent_polylines, agent_mask)

        # 4. Assertions
        expected_elements = N_map + N_agent
        self.assertEqual(context_emb.shape, (self.batch_size, expected_elements, self.hidden_dim),
                         "VectorNet output shape mismatch.")
        self.assertFalse(torch.isnan(context_emb).any(), "VectorNet output contains NaNs.")
        
        # 5. Gradient Check (Crucial for Training)
        loss = context_emb.sum()
        loss.backward()
        self.assertIsNotNone(model.proj.weight.grad, "Gradients are not flowing through VectorNet.")

    def test_flow_matching_integration(self):
        """
        Verifies the full 5-step Euler Integration loop (Alpamayo-R1 logic).
        """
        N_flow_steps = 5
        dt_flow = 1.0 / N_flow_steps
        
        # Models
        encoder = VectorNetEncoder(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        expert = FlowMatchingActionExpert(state_dim=4, hidden_dim=self.hidden_dim)
        
        # Dummy Context (Simulating encoder output)
        context_len = 15
        z_context = torch.randn(self.batch_size, context_len, self.hidden_dim)
        
        # Initial Noise (Gaussian Prior)
        traj = torch.randn(self.batch_size, self.horizon, 4) # [x, y, v, yaw]
        
        # Run Euler Loop
        for i in range(N_flow_steps):
            t_scalar = torch.tensor([i * dt_flow] * self.batch_size)
            velocity = expert(traj, t_scalar, z_context)
            traj = traj + velocity * dt_flow
            
        # Assertions
        self.assertEqual(traj.shape, (self.batch_size, self.horizon, 4))
        self.assertFalse(torch.isnan(traj).any(), "Flow Matching Integration produced NaNs.")


class TestControlStack(unittest.TestCase):
    """
    Tests for Layer 2: The 'Safety Filter' (Linear MPC)
    """

    def setUp(self):
        self.mpc = LinearMPCController(horizon=10, dt=0.1)

    def test_mpc_trivial_solution(self):
        """
        Test if MPC can solve a straight-line tracking problem without crashing.
        """
        # 1. Create a feasible reference (Straight line at 5 m/s)
        # N+1 states needed for reference
        ref_traj = np.zeros((11, 4)) 
        for k in range(11):
            ref_traj[k, 0] = k * 0.5  # x
            ref_traj[k, 2] = 5.0      # v

        # 2. Initial State (Perfectly aligned)
        x0 = np.array([0.0, 0.0, 5.0, 0.0])

        # 3. Solve
        u_opt = self.mpc.solve(x0, ref_traj)

        # 4. Assertions
        # Since we are perfectly aligned, inputs should be near zero
        self.assertLess(abs(u_opt[0]), 0.1, "Acceleration should be near 0 for perfect tracking.")
        self.assertLess(abs(u_opt[1]), 0.01, "Steering should be near 0 for straight line.")

    def test_mpc_correction_logic(self):
        """
        Test if MPC steers correctly when off-track.
        """
        ref_traj = np.zeros((11, 4))
        ref_traj[:, 2] = 5.0 # Target velocity
        
        # Ego is to the LEFT of the path (y = 0.5)
        # Should steer RIGHT (negative steering angle, assuming standard bicycle model)
        x0 = np.array([0.0, 0.5, 5.0, 0.0])
        
        u_opt = self.mpc.solve(x0, ref_traj)
        
        # Verify steering is negative (turning right)
        self.assertLess(u_opt[1], -0.01, 
                        f"MPC Failed to steer right when offset left. Steering: {u_opt[1]}")
        
    def test_mpc_obstacle_avoidance(self):
        """
        Test if MPC modifies trajectory to avoid an obstacle on the path.
        """
        # 1. Reference: Straight line collision course
        ref_traj = np.zeros((11, 4))
        for k in range(11):
            ref_traj[k, 0] = k * 1.0  # x = 0, 1, 2...
            ref_traj[k, 1] = 0.0      # y = 0
            ref_traj[k, 2] = 5.0      # v = 5
        
        # 2. Obstacle: Placed directly at x=5, y=0
        # The reference path goes STRAIGHT THROUGH it.
        # MPC must deviate y to avoid it.
        obstacles = [(5.0, 0.0, 1.0)] # x, y, radius
        
        # 3. Solve
        x0 = np.array([0.0, 0.0, 5.0, 0.0])
        u_opt = self.mpc.solve(x0, ref_traj, obstacles)
        
        # 4. Check Behavior
        # To avoid (5,0), the car must steer (u[1] != 0) immediately or soon.
        # It shouldn't just drive straight (steer ~ 0).
        print(f"Obstacle Avoidance Steering: {u_opt[1]}")
        self.assertNotAlmostEqual(u_opt[1], 0.0, places=2, 
                                  msg="MPC ignored obstacle and drove straight!")


if __name__ == '__main__':
    unittest.main()