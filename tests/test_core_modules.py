import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from vectornet import VectorNetEncoder
from flow_matching import FlowMatchingActionExpert
from mpc_controller import LinearMPCController

class TestGenerativeStack(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.state_dim = 6 
        self.hidden_dim = 128
        self.horizon = 20
        self.device = torch.device('cpu') 

    def test_vectornet_dimensions(self):
        N_map, P_map = 10, 20
        N_agent, P_agent = 5, 20
        map_polylines = torch.randn(self.batch_size, N_map, P_map, self.state_dim)
        map_mask = torch.ones(self.batch_size, N_map, P_map)
        agent_polylines = torch.randn(self.batch_size, N_agent, P_agent, self.state_dim)
        agent_mask = torch.ones(self.batch_size, N_agent, P_agent)

        model = VectorNetEncoder(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        context_emb = model(map_polylines, map_mask, agent_polylines, agent_mask)

        expected_elements = N_map + N_agent
        self.assertEqual(context_emb.shape, (self.batch_size, expected_elements, self.hidden_dim))
        
        loss = context_emb.sum()
        loss.backward()
        self.assertIsNotNone(model.proj.weight.grad)

    def test_flow_matching_integration(self):
        N_flow_steps = 5
        dt_flow = 1.0 / N_flow_steps
        encoder = VectorNetEncoder(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        expert = FlowMatchingActionExpert(state_dim=4, hidden_dim=self.hidden_dim)
        
        context_len = 15
        z_context = torch.randn(self.batch_size, context_len, self.hidden_dim)
        traj = torch.randn(self.batch_size, self.horizon, 4) 
        
        for i in range(N_flow_steps):
            t_scalar = torch.tensor([i * dt_flow] * self.batch_size)
            velocity = expert(traj, t_scalar, z_context)
            traj = traj + velocity * dt_flow
            
        self.assertEqual(traj.shape, (self.batch_size, self.horizon, 4))
        self.assertFalse(torch.isnan(traj).any())


class TestControlStack(unittest.TestCase):
    def setUp(self):
        self.mpc = LinearMPCController(horizon=10, dt=0.1)

    def test_mpc_trivial_solution(self):
        ref_traj = np.zeros((11, 4)) 
        for k in range(11):
            ref_traj[k, 0] = k * 0.5  
            ref_traj[k, 2] = 5.0      
        x0 = np.array([0.0, 0.0, 5.0, 0.0])
        u_opt = self.mpc.solve(x0, ref_traj)
        self.assertLess(abs(u_opt[0]), 0.1)
        self.assertLess(abs(u_opt[1]), 0.01)

    def test_mpc_correction_logic(self):
        ref_traj = np.zeros((11, 4))
        ref_traj[:, 2] = 5.0 
        x0 = np.array([0.0, 0.5, 5.0, 0.0]) # Left of path
        u_opt = self.mpc.solve(x0, ref_traj)
        self.assertLess(u_opt[1], -0.01) # Should steer Right

    def test_mpc_obstacle_avoidance(self):
        """
        Test if MPC avoids an obstacle with sufficient reaction time.
        """
        # 1. Setup Controller with Longer Horizon (2.0 seconds)
        # This gives the car time to kinematically achieve the swerve.
        long_mpc = LinearMPCController(horizon=20, dt=0.1) 

        # 2. Reference: Straight line for 20 steps (10m)
        ref_traj = np.zeros((21, 4))
        for k in range(21):
            ref_traj[k, 0] = k * 0.5  # v=5m/s * 0.1s = 0.5m/step
            ref_traj[k, 2] = 5.0      
        
        # 3. Obstacle: Placed at x=8.0 (1.6s away), Offset y=0.1
        # The car has 1.6s to move laterally by ~2m.
        # Req Accel ~ 1.5 m/s^2 (Very Feasible)
        obstacles = [(8.0, 0.1, 1.0)] 
        
        # 4. Solve
        x0 = np.array([0.0, 0.0, 5.0, 0.0])
        u_opt = long_mpc.solve(x0, ref_traj, obstacles)
        
        print(f"Avoidance Steering: {u_opt[1]:.4f}")
        
        # 5. Check Behavior: Steer Right (Negative)
        # We expect significant steering to initiate the swerve
        self.assertLess(u_opt[1], -0.001, 
                        "MPC failed to steer away (Problem likely Infeasible).")

if __name__ == '__main__':
    unittest.main()