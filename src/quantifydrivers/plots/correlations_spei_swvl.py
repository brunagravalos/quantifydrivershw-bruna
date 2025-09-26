# ============================================================================
# IMPORT NEEDED LIBRARIES
# ============================================================================

import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import pearsonr, norm, rankdata
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ===========================================================================

# Function to normalize to Gaussian via empirical CDF
def gaussianize(series):
    mask = np.isfinite(series)
    x = series[mask]
    if len(x) == 0:
        return np.full_like(series, np.nan, dtype=float)

    ranks = rankdata(x, method='average')
    uniform = (ranks - 0.5) / len(x)
    gaussian = norm.ppf(uniform)

    out = np.full_like(series, np.nan, dtype=float)
    out[mask] = gaussian
    return out

# =========================================================================
# General configuration ---------------------------------------------------

sites = ['cordoba', 'hannover', 'stockholm', 'lyon', 'belgrado']
sties_titles = ['Córdoba', 'Hannover', 'Stockholm', 'Lyon', 'Belgrado']
spei_scales = [30, 60, 90]
soil_moisture_levels = ['swvl1', 'swvl2', 'swvl3']
spei_spi = 'spi'
distribution = "gamma"
dataset = 'eobs'  # or 'era5'
# =========================================================================

# Collect all outputs
output_lines = []
# Create a dictionary to store correlation results for plotting
plot_data = {site: {swvl: [] for swvl in soil_moisture_levels} for site in sites}


# Start loop ----------------------------------------------------------------------------------------------------

for swvl in soil_moisture_levels:
    results = []

    for spei_scale in spei_scales:
        print(f"Doing spei scale : {spei_scale}")
        for site in sites:
            # Load soil water
            swvl_da = xr.open_dataset(
                f"/path/to/your/data/era5_land/variables_era5land_data_{site}_1950_2024.nc"
            ).sel(time=slice('1950','2023'))[swvl]

            # Load SPEI/SPI
            if spei_spi == 'spei':
                spei_eobs = xr.open_dataset(
                    f"/path/to/your/data/observations/gamma_daily_{spei_spi}{spei_scale}_tasmax_{site}_hg.nc"
                )[f'spei_hg_{spei_scale}']
            else:
                spei_eobs = xr.open_dataset(
                    f"/path/to/your/data/observations/gamma_{spei_spi}_tasmax_{site}_{spei_scale}_daily.nc"
                )[f'spi_{spei_scale}']

            if spei_spi == 'spei':
                spei_da = xr.open_dataset(
                    f"/path/to/your/data/era5_land/spei_computations/era5land_gamma_daily_{spei_spi}{spei_scale}_tasmax_{site}_hg.nc"
                )[f'spei_hg_{spei_scale}']
            else:
                spei_da = xr.open_dataset(
                    f"/path/to/your/data/era5_land/spei_computations/era5land_{distribution}_spi_tasmax_{site}_{spei_scale}_daily.nc"
                )[f'spi_{spei_scale}'].sel(time=slice('1951-01-04','2024'))

                # Drop a problematic date if present
                spei_da = spei_da.where(
                    ~((spei_da.time.dt.day == 31) & 
                      (spei_da.time.dt.month == 7) & 
                      (spei_da.time.dt.year == 2024)),
                    drop=True
                )

            # Extract aligned values
            swvl_values = swvl_da.values
            spei_values = spei_eobs.values

            mask = np.isfinite(swvl_values) & np.isfinite(spei_values)
            swvl_clean = swvl_values[mask].ravel()
            spei_clean = spei_values[mask].ravel()

            # Gaussianized correlation
            swvl_norm = gaussianize(swvl_clean)
            spei_norm = gaussianize(spei_clean)
            mask2 = np.isfinite(swvl_norm) & np.isfinite(spei_norm)

            # Spearman correlation
            if len(swvl_clean) > 1:
                corr_spear, pval_spear = spearmanr(swvl_clean, spei_clean)
            else:
                corr_spear, pval_spear = np.nan, np.nan

            # Store results for table
            results.append({
                "Site": site,
                "SPEI_scale": spei_scale,
                "Corr_spear": corr_spear,
                "pval_spear": pval_spear,
            })
            
            # Store results for plotting
            plot_data[site][swvl].append(corr_spear)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    table = df_results.pivot(index="Site", columns="SPEI_scale", values=["Corr_spear"])

    # Add to output text
    output_lines.append(f"\n--- Results for soil moisture level: {swvl} ---\n")
    output_lines.append(table.round(3).to_string())
    output_lines.append("\n")  # extra newline for readability

# Save all tables into one text file
with open(f"/path/to/your/output/{spei_spi}_spearman_{dataset}_soil_moisture_correlation_results.txt", "w") as f:
    f.write("\n".join(output_lines))

print("Results saved to text file.")

# Create the plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()

# Define colors and markers for each soil moisture level
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
markers = ['o', 's', '^']
swvl_labels = ['swvl1', 'swvl2', 'swvl3']

for i, site in enumerate(sites):
    ax = axes[i]
    
    for j, swvl in enumerate(soil_moisture_levels):
        correlations = plot_data[site][swvl]
        ax.plot(spei_scales, correlations, 
                color=colors[j], marker=markers[j], 
                linewidth=2, markersize=8, 
                label=swvl_labels[j])
    
    ax.set_title(sties_titles[i], fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(spei_scales)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    if i == 0:
        ax.legend(loc='upper left', fontsize=14)

    if i < 3:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("SPI Scale (days)", fontsize=16)

    if i % 3 != 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Spearman Correlation", fontsize=16)

    if i == 2:
        ax.set_xlabel("SPI Scale (days)", fontsize=16) 

if len(sites) < 6:
    for i in range(len(sites), 6):
        fig.delaxes(axes[i])

plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save plots
plt.savefig(f"/path/to/your/output/{spei_spi}_{dataset}_soil_moisture_spei_correlation_plot.pdf", 
           dpi=300, bbox_inches='tight')

plt.show()

print(f"Plot saved to /path/to/your/output/{spei_spi}_{dataset}_soil_moisture_spei_correlation_plot.pdf")
