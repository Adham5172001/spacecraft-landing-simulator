"""
Neural Network Controller for Spacecraft Landing
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List


@dataclass
class ControllerConfig:
    hidden_layers: tuple = (128, 64, 32)
    activation: str = "relu"
    learning_rate: float = 0.001
    max_iter: int = 500
    random_state: int = 42


class NeuralLandingController:
    """
    Neural network controller for spacecraft landing.
    Input:  [altitude, velocity_x, velocity_y, fuel, wind_speed, wind_direction]
    Output: [thrust_x, thrust_y, magnitude]
    """

    def __init__(self, config: ControllerConfig = None):
        self.config = config or ControllerConfig()
        self.model = MLPRegressor(
            hidden_layer_sizes=self.config.hidden_layers,
            activation=self.config.activation,
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._fitted = False

    def generate_optimal_trajectories(self, n_trajectories: int = 1000,
                                       seed: int = 42) -> tuple:
        """
        Generate synthetic optimal landing trajectories for supervised training.
        """
        np.random.seed(seed)
        X, y = [], []

        for _ in range(n_trajectories):
            altitude = np.random.uniform(10, 2000)
            velocity_x = np.random.uniform(-50, 50)
            velocity_y = np.random.uniform(-100, -1)
            fuel = np.random.uniform(10, 100)
            wind_speed = np.random.uniform(0, 15)
            wind_direction = np.random.uniform(0, 360)

            # Optimal action: counteract velocity, slow descent
            thrust_x = -np.sign(velocity_x) * min(abs(velocity_x) / 50, 1.0)
            thrust_y = min(abs(velocity_y) / 20, 1.0)
            magnitude = 0.8 if altitude < 100 else 0.5 if altitude < 500 else 0.3

            X.append([altitude, velocity_x, velocity_y, fuel, wind_speed, wind_direction])
            y.append([thrust_x, thrust_y, magnitude])

        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray = None, y: np.ndarray = None) -> "NeuralLandingController":
        """Train the controller on optimal trajectories."""
        if X is None or y is None:
            X, y = self.generate_optimal_trajectories()

        X_s = self.scaler_X.fit_transform(X)
        y_s = self.scaler_y.fit_transform(y)
        self.model.fit(X_s, y_s)
        self._fitted = True
        return self

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Predict optimal action for a given state."""
        if not self._fitted:
            self.fit()
        obs = observation.reshape(1, -1)
        obs_s = self.scaler_X.transform(obs)
        action_s = self.model.predict(obs_s)
        action = self.scaler_y.inverse_transform(action_s)[0]
        return np.clip(action, -1, 1)

    def evaluate(self, env, n_episodes: int = 100) -> dict:
        """Evaluate controller performance over multiple episodes."""
        outcomes = {"soft_landing": 0, "hard_landing": 0, "fuel_exhausted": 0, "timeout": 0}
        touchdown_velocities = []
        fuel_used_list = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = self.act(obs)
                obs, _, done, info = env.step(action)

            outcomes[info.get("outcome", "timeout")] = outcomes.get(info.get("outcome", "timeout"), 0) + 1
            if info.get("touchdown_velocity") is not None:
                touchdown_velocities.append(info["touchdown_velocity"])
            fuel_used_list.append(info.get("fuel_used", 0))

        success_rate = outcomes["soft_landing"] / n_episodes
        return {
            "success_rate": success_rate,
            "outcomes": outcomes,
            "mean_touchdown_velocity": float(np.mean(touchdown_velocities)) if touchdown_velocities else 0.0,
            "mean_fuel_used": float(np.mean(fuel_used_list)),
        }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from simulator.environment import SpacecraftEnv

    print("Neural Landing Controller Demo")
    print("=" * 40)

    config = ControllerConfig(hidden_layers=(128, 64, 32))
    controller = NeuralLandingController(config)

    print("Training on optimal trajectories...")
    controller.fit()
    print(f"  Training complete: {config.hidden_layers} architecture")

    print("\nEvaluating on 50 episodes...")
    env = SpacecraftEnv(wind_enabled=True)
    results = controller.evaluate(env, n_episodes=50)

    print(f"  Success rate: {results['success_rate']:.0%}")
    print(f"  Outcomes: {results['outcomes']}")
    print(f"  Mean touchdown velocity: {results['mean_touchdown_velocity']:.2f} m/s")
    print(f"  Mean fuel used: {results['mean_fuel_used']:.1f} kg")
