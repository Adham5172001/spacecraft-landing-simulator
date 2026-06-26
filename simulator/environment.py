"""Spacecraft Landing Physics Simulator — Author: Adham Aboulkheir | University of Essex"""
import numpy as np
from dataclasses import dataclass

@dataclass
class SpacecraftState:
    altitude: float; velocity_x: float; velocity_y: float
    fuel: float; wind_speed: float; wind_direction: float

class SpacecraftEnv:
    def __init__(self, gravity=1.62, dt=0.1, wind_enabled=True, seed=42):
        self.gravity = gravity; self.dt = dt; self.wind_enabled = wind_enabled; self.seed = seed; self.steps = 0

    def reset(self):
        np.random.seed(self.seed + self.steps)
        self.state = SpacecraftState(
            altitude=np.random.uniform(500, 2000), velocity_x=np.random.uniform(-50, 50),
            velocity_y=np.random.uniform(-100, -10), fuel=100.0,
            wind_speed=np.random.uniform(0, 15) if self.wind_enabled else 0.0,
            wind_direction=np.random.uniform(0, 360))
        self.steps = 0; return self._obs()

    def _obs(self):
        s = self.state
        return np.array([s.altitude, s.velocity_x, s.velocity_y, s.fuel, s.wind_speed, s.wind_direction])

    def step(self, action):
        tx, ty, mag = np.clip(action, -1, 1); mag = abs(mag); s = self.state
        s.velocity_x += tx * mag * 5 * self.dt
        s.velocity_y += (ty * mag * 10 - self.gravity) * self.dt
        if self.wind_enabled:
            s.velocity_x += np.cos(np.radians(s.wind_direction)) * s.wind_speed * 0.01
        s.altitude += s.velocity_y * self.dt
        s.fuel = max(0, s.fuel - mag * 0.5 * self.dt)
        self.steps += 1
        done = s.altitude <= 0 or s.fuel <= 0 or self.steps >= 2000
        outcome = "soft_landing" if done and s.altitude <= 0 and abs(s.velocity_y) < 5 else "hard_landing" if done and s.altitude <= 0 else "timeout"
        return self._obs(), 0.0, done, {"outcome": outcome, "touchdown_velocity": abs(s.velocity_y) if s.altitude <= 0 else None, "fuel_used": 100 - s.fuel}

if __name__ == "__main__":
    env = SpacecraftEnv()
    successes = 0
    for ep in range(50):
        obs = env.reset(); done = False
        while not done:
            alt, vx, vy, fuel, _, _ = obs
            action = np.array([0, 0.8 if vy < -20 else 0.4, 0.6])
            obs, _, done, info = env.step(action)
        if info["outcome"] == "soft_landing": successes += 1
    print(f"Success rate: {successes}/50 ({successes/50:.0%})")
