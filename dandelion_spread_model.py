"""
HiMCM Dandelion Spread Model

This is a cleaned reconstruction of the modeling approach I used for the
2023 HiMCM dandelion problem. The original contest code is not included.

The model keeps the main idea:
- simulate spread on a one-hectare grid,
- adjust growth under different climates,
- include local diffusion and wind-based seed dispersal,
- estimate an invasive-species impact factor.

The goal of this file is to document the modeling framework clearly enough
for a GitHub project, not to reproduce the original submission exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Climate:
    """Climate profile used to adjust biological growth."""

    name: str
    temperature_c: float
    precipitation_mm: float


@dataclass(frozen=True)
class Species:
    """Parameters used for one plant species in the simulation."""

    name: str
    growth_rate: float
    carrying_capacity: float
    diffusion_rate: float
    dispersal_strength: float
    wind_distance: float
    optimal_temperature_c: float
    optimal_precipitation_mm: float
    temperature_tolerance: float
    precipitation_tolerance: float
    ecological_harm: float
    human_benefit: float


CLIMATES: Dict[str, Climate] = {
    "temperate": Climate("temperate", temperature_c=16.0, precipitation_mm=75.0),
    "arid": Climate("arid", temperature_c=27.0, precipitation_mm=20.0),
    "tropical": Climate("tropical", temperature_c=28.0, precipitation_mm=180.0),
}


SPECIES: Dict[str, Species] = {
    "dandelion": Species(
        name="dandelion",
        growth_rate=0.85,
        carrying_capacity=35.0,
        diffusion_rate=0.08,
        dispersal_strength=0.12,
        wind_distance=6.0,
        optimal_temperature_c=18.0,
        optimal_precipitation_mm=80.0,
        temperature_tolerance=11.0,
        precipitation_tolerance=55.0,
        ecological_harm=0.45,
        human_benefit=0.35,
    ),
    "kudzu": Species(
        name="kudzu",
        growth_rate=1.10,
        carrying_capacity=50.0,
        diffusion_rate=0.12,
        dispersal_strength=0.10,
        wind_distance=4.0,
        optimal_temperature_c=26.0,
        optimal_precipitation_mm=130.0,
        temperature_tolerance=8.0,
        precipitation_tolerance=65.0,
        ecological_harm=0.90,
        human_benefit=0.10,
    ),
    "purple_loosestrife": Species(
        name="purple_loosestrife",
        growth_rate=0.95,
        carrying_capacity=45.0,
        diffusion_rate=0.10,
        dispersal_strength=0.16,
        wind_distance=7.0,
        optimal_temperature_c=20.0,
        optimal_precipitation_mm=120.0,
        temperature_tolerance=10.0,
        precipitation_tolerance=70.0,
        ecological_harm=0.80,
        human_benefit=0.05,
    ),
}


class DandelionSpreadModel:
    """Plant-spread simulator on a one-hectare grid."""

    def __init__(
        self,
        species: Species,
        climate: Climate,
        grid_size: int = 100,
        initial_density: float = 1.0,
        random_seed: int = 7,
    ) -> None:
        self.species = species
        self.climate = climate
        self.grid_size = grid_size
        self.rng = np.random.default_rng(random_seed)

        self.density = np.zeros((grid_size, grid_size), dtype=float)

        center = grid_size // 2
        self.density[center, center] = initial_density

    def climate_suitability(self) -> float:
        """Estimate how suitable the climate is for this species."""

        temp_score = np.exp(
            -((self.climate.temperature_c - self.species.optimal_temperature_c) ** 2)
            / (2 * self.species.temperature_tolerance**2)
        )
        precip_score = np.exp(
            -((self.climate.precipitation_mm - self.species.optimal_precipitation_mm) ** 2)
            / (2 * self.species.precipitation_tolerance**2)
        )
        return float(temp_score * precip_score)

    def _laplacian(self, grid: np.ndarray) -> np.ndarray:
        """Local spread term using a discrete Laplacian."""

        return (
            np.roll(grid, 1, axis=0)
            + np.roll(grid, -1, axis=0)
            + np.roll(grid, 1, axis=1)
            + np.roll(grid, -1, axis=1)
            - 4 * grid
        )

    def _seed_kernel(self) -> np.ndarray:
        """Create a simple directional kernel for wind dispersal."""

        radius = max(3, int(self.species.wind_distance * 3))
        y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]

        wind_shift = self.species.wind_distance
        sigma_x = max(1.0, self.species.wind_distance)
        sigma_y = max(1.0, self.species.wind_distance / 2)

        kernel = np.exp(-((x - wind_shift) ** 2) / (2 * sigma_x**2) - (y**2) / (2 * sigma_y**2))
        kernel /= kernel.sum()
        return kernel

    def _convolve_same(self, grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Convolve grid with kernel using FFT and return same-sized output."""

        g_h, g_w = grid.shape
        k_h, k_w = kernel.shape
        out_shape = (g_h + k_h - 1, g_w + k_w - 1)

        grid_fft = np.fft.rfftn(grid, out_shape)
        kernel_fft = np.fft.rfftn(kernel, out_shape)
        full = np.fft.irfftn(grid_fft * kernel_fft, out_shape)

        start_h = k_h // 2
        start_w = k_w // 2
        return full[start_h : start_h + g_h, start_w : start_w + g_w]

    def step(self) -> None:
        """Move the simulation forward by one month."""

        suitability = self.climate_suitability()
        n = self.density
        s = self.species

        growth = suitability * s.growth_rate * n * (1 - n / s.carrying_capacity)
        diffusion = s.diffusion_rate * self._laplacian(n)

        kernel = self._seed_kernel()
        seed_dispersal = s.dispersal_strength * self._convolve_same(n, kernel)

        next_density = n + growth + diffusion + seed_dispersal
        self.density = np.clip(next_density, 0, s.carrying_capacity)

    def simulate(self, months: int) -> Dict[int, np.ndarray]:
        """Run the model and save selected monthly snapshots."""

        snapshots = {}
        for month in range(1, months + 1):
            self.step()
            if month in {1, 2, 3, 6, 12}:
                snapshots[month] = self.density.copy()
        return snapshots

    def metrics(self) -> Dict[str, float]:
        """Summarize the current spread pattern."""

        occupied = self.density > 0.05
        coverage_ratio = occupied.mean()
        total_density = float(self.density.sum())
        mean_density = float(self.density[occupied].mean()) if occupied.any() else 0.0
        max_density = float(self.density.max())

        return {
            "coverage_ratio": float(coverage_ratio),
            "total_density": total_density,
            "mean_occupied_density": mean_density,
            "max_density": max_density,
            "climate_suitability": self.climate_suitability(),
        }


def calculate_impact_factor(
    species: Species,
    spread_metrics: Dict[str, float],
) -> float:
    """
    Calculate a 0-100 impact factor.

    Higher values indicate stronger invasive impact.
    The benefit adjustment is included because the prompt treats dandelions
    as both potentially useful and potentially harmful.
    """

    spread_score = np.clip(spread_metrics["coverage_ratio"] / 0.75, 0, 1)
    density_score = np.clip(spread_metrics["mean_occupied_density"] / species.carrying_capacity, 0, 1)
    climate_score = np.clip(spread_metrics["climate_suitability"], 0, 1)
    dispersal_score = np.clip(species.dispersal_strength / 0.20, 0, 1)
    harm_score = np.clip(species.ecological_harm, 0, 1)
    benefit_adjustment = np.clip(species.human_benefit, 0, 1)

    raw_score = (
        0.25 * spread_score
        + 0.20 * density_score
        + 0.15 * climate_score
        + 0.15 * dispersal_score
        + 0.25 * harm_score
    )

    adjusted_score = raw_score * (1 - 0.25 * benefit_adjustment)
    return float(round(100 * adjusted_score, 2))


def plot_snapshot(grid: np.ndarray, title: str, output_path: Path) -> None:
    """Save a heatmap for the simulated plant density."""

    plt.figure(figsize=(6, 5))
    plt.imshow(grid, origin="lower")
    plt.colorbar(label="Plant density")
    plt.title(title)
    plt.xlabel("x position, meters")
    plt.ylabel("y position, meters")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_all_simulations(output_dir: str = "results") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all species-climate simulations and save summary tables."""

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    spread_rows: List[Dict[str, float | str | int]] = []
    impact_rows: List[Dict[str, float | str]] = []

    for species_name, species in SPECIES.items():
        for climate_name, climate in CLIMATES.items():
            model = DandelionSpreadModel(species=species, climate=climate)
            snapshots = model.simulate(months=12)

            for month, grid in snapshots.items():
                occupied = grid > 0.05
                spread_rows.append(
                    {
                        "species": species_name,
                        "climate": climate_name,
                        "month": month,
                        "coverage_ratio": float(occupied.mean()),
                        "total_density": float(grid.sum()),
                        "max_density": float(grid.max()),
                    }
                )

                if species_name == "dandelion":
                    plot_snapshot(
                        grid,
                        title=f"Dandelion Spread - {climate_name.title()} Climate - Month {month}",
                        output_path=out / f"dandelion_{climate_name}_month_{month}.png",
                    )

            final_metrics = model.metrics()
            impact = calculate_impact_factor(species, final_metrics)
            impact_rows.append(
                {
                    "species": species_name,
                    "climate": climate_name,
                    "impact_factor": impact,
                    "coverage_ratio": final_metrics["coverage_ratio"],
                    "climate_suitability": final_metrics["climate_suitability"],
                }
            )

    spread_df = pd.DataFrame(spread_rows)
    impact_df = pd.DataFrame(impact_rows)

    spread_df.to_csv(out / "spread_summary.csv", index=False)
    impact_df.to_csv(out / "impact_factor_summary.csv", index=False)

    return spread_df, impact_df


def main() -> None:
    spread_df, impact_df = run_all_simulations()

    print("\nSpread summary:")
    print(spread_df.to_string(index=False))

    print("\nImpact factor summary:")
    print(impact_df.to_string(index=False))

    print("\nResults saved to the 'results' folder.")


if __name__ == "__main__":
    main()
