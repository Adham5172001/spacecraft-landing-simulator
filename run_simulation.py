"""Spacecraft Landing Simulation Demo — Author: Adham Aboulkheir | University of Essex"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from simulator.environment import SpacecraftEnv

def main():
    print("Spacecraft Landing Simulator Demo")
    os.makedirs("outputs", exist_ok=True)
    env = SpacecraftEnv(wind_enabled=True)
    outcomes = {}
    for ep in range(100):
        obs = env.reset(); done = False
        while not done:
            alt, vx, vy, fuel, _, _ = obs
            action = np.array([-np.sign(vx)*0.2 if abs(vx)>10 else 0, 0.8 if vy<-20 else 0.4 if vy<-5 else 0.1, 0.6])
            obs, _, done, info = env.step(action)
        outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
    print(f"  Soft landings: {outcomes.get('soft_landing',0)}/100")
    print(f"  Hard landings: {outcomes.get('hard_landing',0)}/100")
    env2 = SpacecraftEnv(seed=7)
    obs = env2.reset(); alts, vels = [obs[0]], [obs[2]]; done = False
    while not done:
        alt, vx, vy, fuel, _, _ = obs
        action = np.array([0, 0.8 if vy<-20 else 0.4 if vy<-5 else 0.1, 0.6])
        obs, _, done, info = env2.step(action)
        alts.append(obs[0]); vels.append(obs[2])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    t = np.arange(len(alts))
    axes[0].plot(t, alts, color="#00c9b1", linewidth=2); axes[0].axhline(y=0, color="#ff7b72", linestyle="--", linewidth=1.5, label="Ground")
    axes[0].set_title("Altitude Profile", color="white"); axes[0].set_xlabel("Time Step", color="white"); axes[0].legend(facecolor="#161b22", labelcolor="white", fontsize=8); axes[0].tick_params(colors="white"); axes[0].grid(alpha=0.3, color="#21262d")
    axes[1].plot(t, vels, color="#f4a261", linewidth=2); axes[1].axhline(y=-5, color="#3fb950", linestyle="--", linewidth=1.5, label="Safe threshold")
    axes[1].set_title("Vertical Velocity", color="white"); axes[1].set_xlabel("Time Step", color="white"); axes[1].legend(facecolor="#161b22", labelcolor="white", fontsize=8); axes[1].tick_params(colors="white"); axes[1].grid(alpha=0.3, color="#21262d")
    plt.tight_layout()
    plt.savefig("outputs/spacecraft_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/spacecraft_results.png")

if __name__ == "__main__":
    main()
