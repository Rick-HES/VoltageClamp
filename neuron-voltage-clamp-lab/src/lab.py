"""
NEURON Voltage Clamp Lab (Jupyter/ipywidgets)

Designed to run in:
- JupyterLab / Jupyter Notebook locally
- Binder (via GitHub repo)

Notes:
- This module provides a `build_ui()` function that renders ipywidgets controls and plots.
- GitHub Pages cannot execute this Python code; use Binder for a shareable "website-like" link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ipywidgets import FloatSlider, RadioButtons, Button, Output, Layout, VBox
from IPython.display import display

try:
    from neuron import h  # type: ignore
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The 'neuron' package is not installed. Install dependencies first:\n"
        "  pip install -r requirements.txt\n"
        "or run via Binder using the badge link in README.md."
    ) from e


PlotChoice = Literal["iNa", "iK", "Both"]
HistoryChoice = Literal["Clear History", "Keep History"]


@dataclass
class Model:
    soma: any
    vc: any


def setup_model(
    soma_L_um: float = 100.0,
    soma_diam_um: float = 100.0,
    v_init_mV: float = -90.0,
    tstop_ms: float = 20.0,
    dt_ms: float = 0.025,
    hold_ms: float = 5.0,
    step_ms: float = 10.0,
    hold_mV: float = -90.0,
    default_step_mV: float = -20.0,
) -> Model:
    """Create a simple single-compartment HH soma + voltage clamp."""
    # Load standard NEURON run library which defines v_init and other globals.
    h.load_file("stdrun.hoc")

    soma = h.Section(name="soma")
    soma.L = soma_L_um
    soma.diam = soma_diam_um
    soma.insert("hh")

    h.v_init = float(v_init_mV)
    h.tstop = float(tstop_ms)
    h.dt = float(dt_ms)

    vc = h.VClamp(soma(0.5))
    vc.dur[0] = float(hold_ms)
    vc.amp[0] = float(hold_mV)

    vc.dur[1] = float(step_ms)
    vc.amp[1] = float(default_step_mV)

    # Remaining time returns to holding potential
    remaining = float(tstop_ms) - float(hold_ms) - float(step_ms)
    if remaining <= 0:
        raise ValueError("tstop_ms must be greater than hold_ms + step_ms.")
    vc.dur[2] = remaining
    vc.amp[2] = float(hold_mV)

    return Model(soma=soma, vc=vc)


def _run_and_record(model: Model):
    """
    Run a NEURON simulation and return time, voltage, iNa, iK vectors as Python lists.

    Uses fresh Vectors each run to avoid subtle recording/linking issues across multiple runs.
    """
    # Initialize
    h.finitialize(h.v_init)

    t_vec = h.Vector()
    v_vec = h.Vector()
    i_na_vec = h.Vector()
    i_k_vec = h.Vector()

    t_vec.record(h._ref_t)
    v_vec.record(model.soma(0.5)._ref_v)
    i_na_vec.record(model.soma(0.5)._ref_ina)
    i_k_vec.record(model.soma(0.5)._ref_ik)

    h.run()

    return list(t_vec), list(v_vec), list(i_na_vec), list(i_k_vec)


def build_ui():
    """
    Render the interactive lab UI in a Jupyter environment.
    Returns the top-level widget container.
    """
    model = setup_model()

    # --- Widgets ---
    vstep_slider = FloatSlider(
        min=-100.0, max=100.0, step=5.0, value=-20.0,
        description="VStep (mV):",
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        layout=Layout(width="520px")
    )

    # Reasonable HH defaults: gnabar=0.12, gkbar=0.036 in NEURON's hh mechanism
    # Your collaborator used higher values; we keep their defaults but ensure reset matches.
    gnabar_slider = FloatSlider(
        min=0.0, max=0.35, step=0.01, value=0.20,
        description="G_Na:",
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".2f",
        layout=Layout(width="520px")
    )

    gkbar_slider = FloatSlider(
        min=0.0, max=0.20, step=0.005, value=0.08,
        description="G_K:",
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".3f",
        layout=Layout(width="520px")
    )

    current_radio = RadioButtons(
        options=["iNa", "iK", "Both"],
        value="Both",
        description="Plot Current(s):",
        disabled=False,
        layout=Layout(width="220px")
    )

    history_radio = RadioButtons(
        options=["Clear History", "Keep History"],
        value="Clear History",
        description="Plot History:",
        disabled=False,
        layout=Layout(width="220px")
    )

    run_button = Button(description="Run Simulation", icon="play")
    reset_button = Button(description="Reset Parameters", button_style="warning", icon="undo")
    run_full_iv_button = Button(description="Run Full IV", button_style="info", icon="chart-line")

    output_area = Output()

    # --- History ---
    t_history: List[List[float]] = []
    v_history: List[List[float]] = []
    i_na_history: List[List[float]] = []
    i_k_history: List[List[float]] = []

    def _plot_single_run(plot_choice: PlotChoice, history_choice: HistoryChoice):
        # Update model parameters
        model.vc.amp[1] = float(vstep_slider.value)
        model.soma.gnabar_hh = float(gnabar_slider.value)
        model.soma.gkbar_hh = float(gkbar_slider.value)

        if history_choice == "Clear History":
            t_history.clear(); v_history.clear(); i_na_history.clear(); i_k_history.clear()

        t, v, ina, ik = _run_and_record(model)

        t_history.append(t); v_history.append(v); i_na_history.append(ina); i_k_history.append(ik)

        # Plot
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

        # Historical traces (dotted grey)
        for i in range(max(0, len(t_history) - 1)):
            axes[0].plot(t_history[i], v_history[i], color="darkgray", linestyle=":", alpha=0.7)

            if plot_choice == "iNa":
                axes[1].plot(t_history[i], i_na_history[i], color="darkgray", linestyle=":", alpha=0.7)
            elif plot_choice == "iK":
                axes[1].plot(t_history[i], i_k_history[i], color="darkgray", linestyle=":", alpha=0.7)
            else:
                total_hist = [a + b for a, b in zip(i_na_history[i], i_k_history[i])]
                axes[1].plot(t_history[i], total_hist, color="darkgray", linestyle=":", alpha=0.7)

        # Current run (prominent)
        axes[0].plot(t_history[-1], v_history[-1], label="Membrane Voltage")
        axes[0].set_title("Membrane Voltage vs. Time")
        axes[0].set_ylabel("Voltage (mV)")
        axes[0].set_ylim(-100, 120)
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_title("Selected Ionic Current(s) vs. Time")
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Current (nA)")

        if plot_choice == "iNa":
            axes[1].plot(t_history[-1], i_na_history[-1], label="Sodium Current (iNa)")
            # Auto y-range based on current magnitude (keeps plots readable)
            span = max(1e-6, np.max(np.abs(i_na_history[-1])))
            axes[1].set_ylim(-1.15 * span, 1.15 * span)
        elif plot_choice == "iK":
            axes[1].plot(t_history[-1], i_k_history[-1], label="Potassium Current (iK)")
            span = max(1e-6, np.max(np.abs(i_k_history[-1])))
            axes[1].set_ylim(-1.15 * span, 1.15 * span)
        else:
            total = [a + b for a, b in zip(i_na_history[-1], i_k_history[-1])]
            axes[1].plot(t_history[-1], total, label="Total Ionic Current (iNa + iK)")
            span = max(1e-6, np.max(np.abs(total)))
            axes[1].set_ylim(-1.15 * span, 1.15 * span)

        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def _plot_full_iv(plot_choice: PlotChoice):
        # Apply conductances for all steps
        model.soma.gnabar_hh = float(gnabar_slider.value)
        model.soma.gkbar_hh = float(gkbar_slider.value)

        v_steps = list(range(-100, 101, 20))

        all_v = []
        all_ina = []
        all_ik = []
        peaks = []
        t_trace: Optional[List[float]] = None

        for v_step in v_steps:
            model.vc.amp[1] = float(v_step)

            t, v, ina, ik = _run_and_record(model)
            if t_trace is None:
                t_trace = t

            all_v.append(v)
            all_ina.append(ina)
            all_ik.append(ik)

            if plot_choice == "iNa":
                # Keep collaborator's convention: for depolarized steps above ~50mV, iNa may reverse sign
                peaks.append(min(ina) if v_step < 50 else max(ina))
            elif plot_choice == "iK":
                peaks.append(max(ik))
            else:
                total = [a + b for a, b in zip(ina, ik)]
                peaks.append(min(total) if v_step < 50 else max(total))

        # Plot layout: voltage family, current family, IV curve
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 2, figure=fig)

        ax_v = fig.add_subplot(gs[0, 0])
        ax_i = fig.add_subplot(gs[1, 0])
        ax_iv = fig.add_subplot(gs[:, 1])

        cmap = plt.cm.jet
        colors = [cmap(i) for i in np.linspace(0, 1, len(v_steps))]

        assert t_trace is not None

        for i, v_step in enumerate(v_steps):
            ax_v.plot(t_trace, all_v[i], color=colors[i], label=f"Vstep={v_step}mV")

            if plot_choice == "iNa":
                ax_i.plot(t_trace, all_ina[i], color=colors[i])
            elif plot_choice == "iK":
                ax_i.plot(t_trace, all_ik[i], color=colors[i])
            else:
                total = [a + b for a, b in zip(all_ina[i], all_ik[i])]
                ax_i.plot(t_trace, total, color=colors[i])

        ax_v.set_title("Family of Voltage Traces (Full IV)")
        ax_v.set_ylabel("Voltage (mV)")
        ax_v.set_ylim(-100, 120)
        ax_v.grid(True)

        ax_i.set_title(f"Family of {plot_choice} Current Traces (Full IV)")
        ax_i.set_xlabel("Time (ms)")
        ax_i.set_ylabel("Current (nA)")
        ax_i.grid(True)

        # IV curve
        ax_iv.plot(v_steps, peaks, marker="o", linestyle="-", color="black")
        ax_iv.set_title(f"Peak {plot_choice} Current I-V Curve")
        ax_iv.set_xlabel("Voltage (mV)")
        ax_iv.set_ylabel("Peak Current (nA)")
        ax_iv.grid(True)
        ax_iv.axhline(0, color="black", lw=1.5)
        ax_iv.axvline(0, color="black", lw=1.5)

        # Ensure 0,0 visible
        ax_iv.set_xlim(min(v_steps), max(v_steps))
        max_abs = max(np.max(np.abs(peaks)), 1e-6)
        ax_iv.set_ylim(-1.15 * max_abs, 1.15 * max_abs)

        plt.tight_layout()
        plt.show()

    # --- Button handlers ---
    def on_run(_):
        with output_area:
            output_area.clear_output(wait=True)
            print(
                f"Running: Vstep={vstep_slider.value:.1f} mV | "
                f"G_Na={gnabar_slider.value:.2f} | G_K={gkbar_slider.value:.3f} | "
                f"Plot={current_radio.value} | History={history_radio.value}"
            )
            _plot_single_run(current_radio.value, history_radio.value)  # type: ignore
            print("Done.")

    def on_reset(_):
        # Keep reset values consistent with widget defaults
        vstep_slider.value = -20.0
        gnabar_slider.value = 0.20
        gkbar_slider.value = 0.08
        history_radio.value = "Clear History"
        t_history.clear(); v_history.clear(); i_na_history.clear(); i_k_history.clear()

        with output_area:
            output_area.clear_output(wait=True)
            print("Parameters reset to defaults. Click 'Run Simulation' or 'Run Full IV'.")

    def on_full_iv(_):
        with output_area:
            output_area.clear_output(wait=True)
            print(
                f"Running Full IV: Vstep -100→100 mV (20 mV steps) | "
                f"G_Na={gnabar_slider.value:.2f} | G_K={gkbar_slider.value:.3f} | "
                f"Plot={current_radio.value}"
            )
            _plot_full_iv(current_radio.value)  # type: ignore
            print("Done.")

    run_button.on_click(on_run)
    reset_button.on_click(on_reset)
    run_full_iv_button.on_click(on_full_iv)

    controls = VBox(
        [vstep_slider, gnabar_slider, gkbar_slider, current_radio, history_radio,
         run_button, reset_button, run_full_iv_button]
    )

    ui = VBox([controls, output_area])
    display(ui)

    # Helpful first message
    with output_area:
        print("UI ready. Click 'Run Simulation' or 'Run Full IV'.")

    return ui
