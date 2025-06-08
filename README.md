# Threshold Dialectics - Chapter 16: Echoes of Motion
### Experiment 16C: Fractal Characteristics of Dynamic Lever Trajectories


This repository contains the Python code for **Experiment 16C**, as described in Chapter 16 ("*Echoes of Motion: Fractals as Frozen Signatures of Threshold Dynamics*") of the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness*.

This experiment investigates the relationship between the dynamic pathways leading to systemic collapse and their resulting geometric complexity, measured using fractal analysis.

**Note:** This repository only covers Experiment 16C. Experiments 16A (Sandpile Avalanche Scars) and 16B (Cellular Automaton Cascades) are detailed in separate repositories.

## Introduction: The Geometry of Collapse

A central thesis of Threshold Dialectics (TD) is that the dynamic processes governing a system's struggle for viability leave tangible geometric imprints. This repository explores this idea by analyzing the fractal characteristics of the system's trajectory in its "lever-velocity space" as it approaches collapse.

We hypothesize that the manner in which a system manages its core adaptive capacities leaves a geometric "echo of motion" in its state space. By analyzing the fractal dimension and persistence (Hurst exponent) of these trajectories, we can gain deeper insights into the nature of the collapse itself. This experiment tests the hypothesis that faster, more direct collapses follow geometrically simpler paths, while slower, more contested descents into fragility trace more complex, higher-dimensional trajectories.

## Key Concepts of Threshold Dialectics (TD)

For those unfamiliar with the framework, here are the core concepts used in this experiment:

*   **Adaptive Levers:** The three core capacities a system uses to manage viability:
    *   **Perception Gain ($\gLever$):** The system's sensitivity to new information or prediction errors.
    *   **Policy Precision ($\betaLever$):** The rigidity or confidence with which the system adheres to its current operational rules or models.
    *   **Energetic Slack ($\FEcrit$):** The system's buffer of readily available resources (e.g., energy, capital, time) to absorb shocks and fuel adaptation.
*   **Tolerance Sheet ($\Theta_T$):** A dynamic, multi-dimensional boundary representing the maximum systemic strain a system can withstand given its current configuration of the three levers. Collapse occurs when strain exceeds this tolerance.
*   **TD Diagnostics:**
    *   **Speed Index ($\SpeedIndex$):** Measures the joint velocity of the structural levers ($\betaLever, \FEcrit$), indicating the rate of change in the system's adaptive posture.
    *   **Couple Index ($\CoupleIndex$):** Measures the correlation between the lever velocities, indicating how their drifts are coordinated.
*   **Fractal Analysis:**
    *   **Box-Counting Dimension:** A measure of a pattern's geometric complexity and how it fills space. A higher dimension indicates a more intricate, space-filling pattern.
    *   **Hurst Exponent (H):** A measure of long-term memory in a time series. H > 0.5 indicates a persistent or trend-following behavior.

## Experiment 16C: Methodology

The simulation script ("chap16_experiment_16C.py") executes the following methodology:

1.  **Simulate Collapse Scenarios:** It runs a large set of Monte Carlo simulations of a Threshold Dialectics system under three distinct collapse scenarios:
    *   "slow_creep": A gradual erosion of systemic viability.
    *   "sudden_shock": An abrupt, shock-induced failure.
    *   "tightening_loop": A feedback-driven descent into fragility.
2.  **Record Lever Velocities:** For each simulation run, the script records the time series of the lever velocities ($\dot{\gLever}, \dot{\betaLever}, \dot{\FEcrit}$).
3.  **Calculate Fractal Metrics:** It then analyzes the geometry of these dynamic pathways by computing:
    *   The **2D Box-Counting Dimension** of the trajectory in the $(\dot{\betaLever}, \dot{\FEcrit})$ velocity plane.
    *   The **Hurst Exponent** for each of the individual lever velocity time series.
4.  **Correlate Dynamics with Geometry:** Finally, it performs a statistical analysis to find correlations between the pre-collapse TD diagnostics (like the average Speed Index) and the calculated fractal metrics of the trajectory.

## Setup and Installation

To run this experiment, you will need Python 3.x and the packages listed in "requirements.txt".

1.  Clone the repository:
    """bash
    git clone <repository-url>
    cd <repository-directory>
    """

2.  Install the required packages. It is recommended to use a virtual environment.
    """bash
    pip install -r requirements.txt
    """

    **requirements.txt:**
    """
    numpy
    pandas
    matplotlib
    scipy
    seaborn
    tqdm
    """

3.  **Optional Dependencies:** The script can leverage more advanced fractal analysis libraries if they are installed. The experiment will still run without them, falling back to simplified internal implementations.
    *   **MFDFA:** "pip install MFDFA"
    *   **nolds:** "pip install nolds"

## Usage

You can run the full experiment from the command line.

#### Standard Run

This will execute the entire suite of simulations and analyses. The number of Monte Carlo runs per parameter set is defined by "N_MC_RUNS" in the script.

"""bash
python chap16_experiment_16C.py
"""

#### Reproducible Run

To ensure reproducibility, you can provide a seed for the random number generator.

"""bash
python chap16_experiment_16C.py --seed 123
"""

#### Post-Processing

The main script generates a detailed CSV of all results. A separate script, "summarize_results.py", is provided to load the latest results and generate a JSON summary of key statistics and correlations.

"""bash
python summarize_results.py
"""

## What the Script Does

When executed, "chap16_experiment_16C.py" performs the following steps:

1.  **Creates Output Directory:** A unique, time-stamped directory (e.g., "results_20231027_143000") is created to store all outputs.
2.  **Generates Parameter Space:** It defines a grid of parameters to be tested across the "slow_creep", "sudden_shock", and "tightening_loop" scenarios.
3.  **Runs Simulations in Parallel:** It uses "concurrent.futures.ProcessPoolExecutor" to run the simulations in parallel, significantly speeding up the experiment. Each run simulates the "TDSystem" until collapse or "max_time".
4.  **Calculates Metrics:** During each run, it calculates lever states, TD diagnostics, and performs fractal analysis on the resulting lever velocity trajectories.
5.  **Generates Plots:** For a subset of runs, it generates and saves several diagnostic plots to the results directory, including:
    *   2D and 3D lever velocity trajectories.
    *   Evolution of Speed and Couple Indices over time.
    *   Evolution of rolling fractal metrics.
    *   Evolution of the Tolerance Sheet and systemic strain.
6.  **Saves Aggregate Results:** All metrics and parameters from every simulation run are aggregated into a single "pandas" DataFrame and saved as "td_fractal_experiment_1_results.csv" in the results directory.
7.  **Prints Summary to Console:** Finally, it prints a summary of the results, collapse rates by scenario, and a correlation report to the console.

## Key Findings: The Simplicity of a Fast Collapse

The primary and most compelling finding from this experiment is a **strong and statistically significant negative correlation between the pre-collapse Speed Index and the fractal dimension of the velocity trajectory** (Pearson's r ≈ -0.92).

> This counter-intuitive result suggests that the faster a system is drifting towards collapse (high $\SpeedIndex$), the *simpler* and less geometrically complex its path in the lever-velocity space.

**Interpretation:**

*   **Fast Collapse (High Speed, Low Dimension):** A direct, efficient, and "unsurprising" path to failure. The system has found a steep gradient in its viability landscape and follows it directly without much hesitation or exploration of alternative paths. This is characteristic of the "sudden_shock" scenario.
*   **Slow Collapse (Low Speed, High Dimension):** A more tortured, complex, and contested path. The system is likely making many small, conflicting adjustments as it struggles to maintain viability against a more gradual pressure. This meandering, high-dimension path is characteristic of the "slow_creep" and "tightening_loop" scenarios, and may be an imprint of a system operating in a state of Self-Organized Tolerance Creep (SOTC).

This finding powerfully suggests that the geometry of the "motion" towards collapse is a rich source of diagnostic information, reflecting the fundamental nature of the system's adaptive struggle.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.