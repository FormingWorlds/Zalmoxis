from __future__ import annotations

import os

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Run from the root directory with: python -m src.zalmoxis.plots.plot_animated_pressure_density_profiles

def create_video(pressure_filename, density_filename):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    def read_pressure_file(pressure_filename):
        all_profiles = []
        with open(pressure_filename, "r") as f:
            lines = f.readlines()

        radius = []
        pressure = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):  # new iteration marker
                if radius and pressure:
                    all_profiles.append((np.array(radius), np.array(pressure)))
                    radius, pressure = [], []
            elif line.startswith("radius"):
                continue
            else:
                vals = line.split()
                radius.append(float(vals[0]))
                pressure.append(float(vals[1]))

        if radius and pressure:
            all_profiles.append((np.array(radius), np.array(pressure)))

        return all_profiles

    def read_density_file(density_filename):
        all_profiles = []
        with open(density_filename, "r") as f:
            lines = f.readlines()

        radius = []
        density = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):  # new iteration marker
                if radius and density:
                    all_profiles.append((np.array(radius), np.array(density)))
                    radius, density = [], []
            elif line.startswith("radius"):
                continue
            else:
                vals = line.split()
                radius.append(float(vals[0]))
                density.append(float(vals[1]))

        if radius and density:
            all_profiles.append((np.array(radius), np.array(density)))

        return all_profiles

    def make_pressure_movie(r_all, p_all):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)

        ax.set_xlim(min(r_all[0]), max(r_all[0]))
        ax.set_ylim(min(min(p) for p in p_all), max(max(p) for p in p_all))
        ax.set_xlabel("Radius (m)")
        ax.set_ylabel("Pressure (Pa)")
        ax.set_title("Pressure vs Radius Evolution")

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(r_all[frame], p_all[frame])
            ax.legend([f"Iteration {frame+1}"], loc="upper right")
            return line,

        ani = FuncAnimation(fig, update, frames=len(p_all), init_func=init, blit=True)

        ani.save(os.path.join(ZALMOXIS_ROOT, "output_files", "pressure_evolution.mp4"), writer="ffmpeg", fps=2)
        #plt.show()

    def make_density_movie(r_all, d_all):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)

        ax.set_xlim(min(r_all[0]), max(r_all[0]))
        ax.set_ylim(min(min(d) for d in d_all), max(max(d) for d in d_all))
        ax.set_xlabel("Radius (m)")
        ax.set_ylabel("Density (kg/m^3)")
        ax.set_title("Density vs Radius Evolution")

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(r_all[frame], d_all[frame])
            ax.legend([f"Iteration {frame+1}"], loc="upper right")
            return line,

        ani = FuncAnimation(fig, update, frames=len(d_all), init_func=init, blit=True)

        ani.save(os.path.join(ZALMOXIS_ROOT, "output_files", "density_evolution.mp4"), writer="ffmpeg", fps=2)
        #plt.show()

    pressure_profiles = read_pressure_file(pressure_filename)
    density_profiles = read_density_file(density_filename)
    r_all = [rp[0] for rp in pressure_profiles]
    p_all = [rp[1] for rp in pressure_profiles]
    d_all = [dp[1] for dp in density_profiles]

    make_pressure_movie(r_all, p_all)
    make_density_movie(r_all, d_all)

if __name__ == "__main__":
    pressure_filename = os.path.join(ZALMOXIS_ROOT, "output_files", "pressure_profiles.txt")
    density_filename = os.path.join(ZALMOXIS_ROOT, "output_files", "density_profiles.txt")

    create_video(pressure_filename=pressure_filename, density_filename=density_filename)
