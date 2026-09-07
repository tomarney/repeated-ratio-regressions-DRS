# Metadata
#/ Type: DRS
#/ Name: Repeated Ratio Regressions (R3)
#/ Authors: Thomas Arney
#/ Description: Calibration by time-varying regressions of measured and reference molar ratios.
#/ References: Tang et al. (2025, JAAS) DOI: 10.1039/D5JA00333D
#/ Version: 0.4
#/ Contact: t.arney@soton.ac.uk


# # Uncomment for autocomplete in your IDE (outside iolite)
# from iolitepythoninterface import (
#     data,
#     TimeSeriesDataType,
#     SelectionGroupType,
#     Result,
#     drs,
#     IoLog,
# )

from iolite import QtGui, QtCore
from iolite.Qt import Qt, QColor
from iolite.ui import IolitePlotPyInterface as Plot
from iolite.ui import IolitePlotSettingsDialog as PlotSettings
from iolite.ui import QCPErrorBars, QCPRange
from iolite.QtGui import QAction, QPen
from iolite.ui import CommonUIPyInterface as CUI
from functools import partial
import numpy as np
import re
from scipy import odr

RM_COLOR_MAP = {}


def get_color(rm_name):
    if rm_name not in RM_COLOR_MAP:
        RM_COLOR_MAP[rm_name] = PLOT_COLOURS[len(RM_COLOR_MAP) % len(PLOT_COLOURS)]
    return RM_COLOR_MAP[rm_name]


tableau20 = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
]

PLOT_COLOURS = [QColor(r, g, b) for r, g, b in tableau20]


class RMBlock:
    """
    Represents a block of standards in a standard-sample-standard protocol.
    Contains RM selections (measurements) to be used for the regression.
    
    Each RM can appear multiple times within a block if measurements are contiguous.
    """

    def __init__(self, time, rm_sels):
        """
        Args:
            time: float. Midpoint time of the first and last RM in the block.
            rm_sels: dict {RM_name: [Selection, ...]}. List of selections for each RM.
        """
        self.time = time
        self.rm_sels = rm_sels

    def __repr__(self):
        if not self.rm_sels:
            return f"RMBlock(t={self.time:.1f}s, empty)"
        rm_counts = ", ".join(
            [f"{name}({len(sels)})" for name, sels in self.rm_sels.items()]
        )
        return f"RMBlock(t={self.time:.1f}s, {rm_counts})"


# --- Initialize Plot object ---
try:
    PLOT
except:
    PLOT = Plot()
    PLOT.setAttribute(Qt.WA_DeleteOnClose)
    PLOT.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    PLOT.bottom().label = "Measured ratio"
    PLOT.left().label = "Reference ratio"

    ann = PLOT.annotate("", 0.015, 0.01, "ptAxisRectRatio", Qt.AlignLeft | Qt.AlignTop)
    ann.visible = False

    def showSettings():
        d = PlotSettings(PLOT)
        d.exec_()

    settingsAction = QAction(PLOT.contextMenu())
    settingsAction.setText("Settings")
    settingsAction.triggered.connect(showSettings)
    PLOT.contextMenu().addAction(settingsAction)


# --- Core fitting functions ---


def lm(B, x):
    """Linear function y = m*x + c for ODR fitting."""
    return B[0] * x + B[1]


def R_squared(observed, predicted):
    """Calculate R-squared goodness of fit."""
    return 1.0 - (np.var((observed - predicted)) / np.var(observed))


def fit_odr(x, y, x_err, y_err):
    """Fit ODR linear model"""
    linear_model = odr.Model(lm)
    # Need an initial rough guess for ODR
    slope_init = (
        (np.max(y) - np.min(y)) / (np.max(x) - np.min(x))
        if (np.max(x) - np.min(x)) != 0
        else 1
    )
    if np.isnan(slope_init):
        slope_init = 1.0

    odr_data = odr.RealData(x, y, sx=x_err, sy=y_err)
    odr_instance = odr.ODR(odr_data, linear_model, beta0=[slope_init, 0.0])
    odr_out = odr_instance.run()
    r_sq = R_squared(y, lm(odr_out.beta, x))

    return odr_out, r_sq


# --- Data gathering helpers ---


def _check_and_fix_uncertainty(value, unc):
    """
    Check the given uncertainty. If invalid, and option is enabled in UI,
    calculate an uncertainty value based on a fixed (user-defined) relative error.
    Returns None if the given uncertainty is invalid and the fix is not enabled.
    """
    if unc <= 0 or np.isnan(unc):
        if drs.setting("FixMissingUnc"):
            fix_percent = drs.setting("MissingUnc2RSD")
            if fix_percent is not None:
                return value * (fix_percent / 100.0)
        return None
    return unc


def _extract_element_symbol(el_name):
    """Get element symbol from isotope name (e.g. Sr from Sr88)."""
    match = re.match(r"([a-zA-Z]+)([0-9]+)", el_name)
    return match.group(1) if match else None


def gather_ratio_stats(target_el, ca_channel, selection=None, rm_group=None):
    """
    Gather measured and reference values and uncertainties for a given ratio.

    Args:
        target_el: str, element with mass (e.g. "Sr88")
        ca_channel: str, calcium channel name (e.g. "Ca43")
        selection: Selection object (for individual block fits)
        rm_group: SelectionGroup object (for overview fit)

    Returns: tuple (meas_mean, ref_val, meas_unc, ref_unc) or None
    """
    el_sym = _extract_element_symbol(target_el)
    if not el_sym:
        return None

    ratio_channel_name = f"{target_el}_{ca_channel}_Raw"
    ref_lookup = f"{el_sym}/Ca"

    if selection:
        # single selection in a block
        rm_name = selection.group().name
        sg = selection.group()
    elif rm_group:
        # all measurements of a particular RM (in overview)
        rm_name = rm_group.name
        sg = rm_group
    else:
        return None

    try:
        # Get reference data
        rm_data = data.referenceMaterialData(rm_name)
        if ref_lookup not in rm_data:
            return None

        ref_val = rm_data[ref_lookup].value()
        ref_unc = _check_and_fix_uncertainty(ref_val, rm_data[ref_lookup].uncertainty())
        if ref_unc is None:
            return None

        # Get measured data
        if ratio_channel_name not in data.timeSeriesNames(data.Intermediate):
            IoLog.error(f"Ratio channel {ratio_channel_name} not found.")
            return None

        ts = data.timeSeries(ratio_channel_name)

        # iolite returns results differently for selections and groups
        if selection:
            # For individual selection, use data.result()
            result = data.result(selection, ts)
            meas_mean = result.value()
            meas_unc = result.uncertainty()
        else:
            # For group, use stats()
            stats = sg.stats(ts)
            meas_mean = stats["mean"]
            meas_unc = stats["uncertainty"]

        return (meas_mean, ref_val, meas_unc, ref_unc)

    except Exception as e:
        IoLog.error(f"Could not get data for {target_el} in {rm_name}: {e}")
        return None


def filter_elements_to_available(selected_elements, allInputChannels):
    """
    Filter selected elements to those available in input channels
    Returns a the filtered list and a list of (name, mass) tuples.
    """
    available_names = [ch.name for ch in allInputChannels]
    filtered = [el for el in selected_elements if el in available_names]

    if not filtered:
        return [], []

    elements_tuples = []
    for el_name in filtered:
        # extract element symbol and mass from the isotope name
        match = re.match(r"([a-zA-Z]+)([0-9]+)", el_name)
        if match:
            elements_tuples.append((match.group(1), match.group(2)))
        else:
            IoLog.warning(f"Could not parse element and mass from {el_name}. Skipping.")

    return filtered, elements_tuples


def get_baseline_group():
    """Get the baseline selection group."""
    baseline_grps = data.selectionGroupList(1)  # 1 = Baseline
    if baseline_grps is None or len(baseline_grps) == 0:
        return None
    if len(baseline_grps) > 1:
        IoLog.warning("Multiple baseline groups found. Using the first one.")
    return baseline_grps[0]


def create_mask(isMaskDefined, maskChannel, cutoff, trim, indexChannel):
    """Create mask using iolite's cutoff function"""
    if isMaskDefined:
        mask = drs.createMaskFromCutoff(maskChannel, cutoff, trim)
    else:
        # if user has not defined a mask, return an array of ones (all True)
        mask = np.ones_like(indexChannel.data())

    data.createTimeSeries("mask", data.Intermediate, indexChannel.time(), mask)
    return mask


def calc_raw_ratios(ca_channel_name, selected_elements, indexChannel):
    """
    Calculate raw element/Ca ratios from baseline-subtracted CPS channels.
    Checks if CPS channels exist and triggers baseline subtraction if not.

    Returns true if successful and false otherwise
    """

    ca_cps_name = f"{ca_channel_name}_CPS"

    # Check if CPS channels exist. if not, run baseline subtraction
    try:
        data.timeSeries(ca_cps_name)
    except Exception:
        # CPS channels don't exist. Go baseline subtract
        try:
            bl_grp = get_baseline_group()
            if not bl_grp:
                IoLog.error("Cannot compute raw ratios: no baseline group found.")
                return False

            # mask needed for baseline subtraction. just use a temprary all-true mask here
            mask = np.ones_like(indexChannel.data())
            allInputChannels = data.timeSeriesList(data.Input)
            drs.baselineSubtract(bl_grp, allInputChannels, mask, 5, 30)
        except Exception as e:
            # some other failure not related to a missing baseline group
            IoLog.error(f"Baseline subtraction failed: {e}")
            return False

    try:
        ca_data = data.timeSeries(ca_cps_name).data()
    except Exception as e:
        IoLog.error(
            f"No baseline-subtracted {ca_cps_name} found. DRS cannot proceed: {e}"
        )
        return False

    # Baseline-subtracted channels should exist now, so calc El/Ca ratios:

    ca_mass_match = re.search(r"(\d+)", ca_channel_name)
    ca_mass = (
        ca_mass_match.group(1) if ca_mass_match else "43"
    )  # default. Maybe better to raise exception?

    for ch in data.timeSeriesList(data.Intermediate):
        ch_name = ch.name.replace("_CPS", "")
        if ch_name not in selected_elements:
            continue

        el = ch.property("Element")
        mass = ch.property("Mass")

        # suppress warnings about NaNs or dividing by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ch.data() / ca_data
            ratio[~np.isfinite(ratio)] = np.nan

        ratio_name = f"{el}{mass}_Ca{ca_mass}_Raw"
        data.createTimeSeries(ratio_name, data.Intermediate, indexChannel.time(), ratio)

    return True


def drs_complete(error=False, error_message="Finished!"):
    """
    Tell the user that the DRS has stopped, hopefully because it was successful
    but gives a notification of errors if not.
    """
    if error:
        IoLog.error(error_message)
        drs.message("Error. See Messages")
    else:
        drs.message("Finished!")
    drs.progress(100)
    drs.finished()


def get_block_regression_data(
    block: RMBlock, el_name, ca_channel_name, selected_rms=None, min_rms_per_block=2
):
    """
    Gathers RM stats then calculates ODR fit for a single RM block.
    Returns dict: fit params and data, used in both DRS and UI flow.
    """
    rm_data = []
    valid_rm_names = []

    for rm_group_name, sel_list in block.rm_sels.items():
        if selected_rms is not None and rm_group_name not in selected_rms:
            continue

        for sel in sel_list:
            stats = gather_ratio_stats(el_name, ca_channel_name, selection=sel)
            if stats:
                rm_data.append(stats)
                valid_rm_names.append(rm_group_name)

    if len(set(valid_rm_names)) < min_rms_per_block:
        return None

    x, y, x_err, y_err = (np.array(vals) for vals in zip(*rm_data))
    odr_out, r_sq = fit_odr(x, y, x_err, y_err)

    slope_unc = odr_out.sd_beta[0] if hasattr(odr_out, "sd_beta") else 0.0
    intercept_unc = odr_out.sd_beta[1] if hasattr(odr_out, "sd_beta") else 0.0

    return {
        "fit": {
            "slope": float(odr_out.beta[0]),
            "intercept": float(odr_out.beta[1]),
            "r_squared": float(r_sq),
            "slope_unc": float(slope_unc),
            "intercept_unc": float(intercept_unc),
        },
        "data": {
            "x": x,
            "y": y,
            "x_err": x_err,
            "y_err": y_err,
            "rm_names": valid_rm_names,
            "raw_stats": rm_data,
        },
    }


def fit_regressions_for_all_blocks(
    blocks, elements, ca_channel_name, min_rms_per_block=2, rm_selections=None
):
    """
    Fit regressions for all blocks and elements.

    Applies `get_block_regression_data()` to each block in the session

    Args:
        blocks: list of RMBlock objects
        elements: list of (element, mass) tuples
        ca_channel_name: str
        min_rms_per_block: int. Minimum standards per block
        rm_selections: dict. Standards to use in the fits, per element

    Returns: dict {element: {times, slopes, intercepts, slopes_unc, intercepts_unc, r_squared}}
    """
    results_dict = {}

    for el, mass in elements:
        el_name = f"{el}{mass}"
        times = []
        slopes = []
        intercepts = []
        slopes_unc = []
        intercepts_unc = []
        r_squared = []

        selected_rms = rm_selections.get(el_name, []) if rm_selections else None

        for block_num, block in enumerate(blocks):
            # Count unique RM types in this block (after filtering by selected_rms if needed)
            if selected_rms is not None:
                viable_rms = sum(1 for rm in block.rm_sels.keys() if rm in selected_rms)
            else:
                viable_rms = len(block.rm_sels)

            if viable_rms < min_rms_per_block:
                IoLog.warning(
                    f"Block has {viable_rms} valid RM types for {el_name}. "
                    f"Need at least {min_rms_per_block}. Skipping."
                )
                continue

            result = get_block_regression_data(
                block,
                el_name,
                ca_channel_name,
                selected_rms,
                min_rms_per_block=min_rms_per_block,
            )
            if result is None:
                IoLog.warning(f"Could not fit {el_name}/Ca for block {block_num}")
                continue

            fit_result = result["fit"]
            times.append(block.time)
            slopes.append(fit_result["slope"])
            intercepts.append(fit_result["intercept"])
            slopes_unc.append(fit_result["slope_unc"])
            intercepts_unc.append(fit_result["intercept_unc"])
            r_squared.append(fit_result["r_squared"])

        if slopes:
            results_dict[el] = {
                "times": np.array(times),
                "slopes": np.array(slopes),
                "intercepts": np.array(intercepts),
                "slopes_unc": np.array(slopes_unc),
                "intercepts_unc": np.array(intercepts_unc),
                "r_squared": np.array(r_squared),
            }

    return results_dict


def fit_splines_for_element(
    block_reg_results, element, ca_channel_name, indexChannel, spline_type="StepLinear"
):
    """
    Fit the given spline to block regression results for a single element/Ca ratio.
    Creates intermediate channels for time-varying slope and intercept.

    Args:
        block_reg_results: dict from fit_regressions_for_all_blocks for one element
        element: str (element symbol e.g. "Sr")
        ca_channel_name: str (e.g. "Ca43")
        indexChannel: ChannelData object for full time array
        spline_type: str, type of spline

    Returns: tuple (slope_ts, intercept_ts) or (None, None)
    """
    if element not in block_reg_results:
        return None, None

    reg_data = block_reg_results[element]
    times = reg_data["times"]
    slopes = reg_data["slopes"]
    intercepts = reg_data["intercepts"]
    slopes_unc = reg_data["slopes_unc"]
    intercepts_unc = reg_data["intercepts_unc"]

    if len(times) < 2:
        IoLog.warning(
            f"Not enough valid blocks for {element}/Ca spline. Need at least 2."
        )
        return None, None

    slopes_unc = np.where(slopes_unc <= 0, np.abs(slopes) * 0.05, slopes_unc)
    intercepts_unc = np.where(intercepts_unc <= 0, 1e-6, intercepts_unc)

    ca_mass_match = re.search(r"(\d+)", ca_channel_name)
    ca_mass = ca_mass_match.group(1) if ca_mass_match else "43"

    try:
        slope_spl = data.spline(
            times, slopes, slopes_unc, spline_type, indexChannel.time()
        )
        intercept_spl = data.spline(
            times, intercepts, intercepts_unc, spline_type, indexChannel.time()
        )
    except Exception as e:
        IoLog.error(f"Failed to create splines for {element}/Ca: {e}")
        return None, None

    slope_channel_name = f"{element}_Ca{ca_mass}_slope"
    intercept_channel_name = f"{element}_Ca{ca_mass}_intercept"

    data.createTimeSeries(
        slope_channel_name, data.Intermediate, indexChannel.time(), slope_spl
    )
    data.createTimeSeries(
        intercept_channel_name, data.Intermediate, indexChannel.time(), intercept_spl
    )

    return slope_spl, intercept_spl


def calibrate_ratios(
    block_reg_results, elements, ca_channel_name, indexChannel, spline_type="StepLinear"
):
    """Calibrate raw ratios using time-varying (splined) slope/intercept values."""

    for el, mass in elements:
        el_name = f"{el}{mass}"

        slope_spl, intercept_spl = fit_splines_for_element(
            block_reg_results, el, ca_channel_name, indexChannel, spline_type
        )

        if slope_spl is None or intercept_spl is None:
            IoLog.warning(f"Could not create splines for {el}/Ca. Skipping.")
            continue

        try:
            raw_ts = data.timeSeries(f"{el_name}_{ca_channel_name}_Raw")
        except Exception as e:
            IoLog.error(
                f"Raw ratio channel {el_name}_{ca_channel_name}_Raw not found: {e}"
            )
            continue

        raw_data = raw_ts.data()
        corrected_data = slope_spl * raw_data + intercept_spl

        data.createTimeSeries(
            f"{el}/Ca", data.Output, indexChannel.time(), corrected_data
        )


def apply_secondary_normalisation(
    sec_norm_enabled, sec_norm_rm, sec_norm_ref_material, elements, indexChannel
):
    """
    Apply secondary normalisation using reference material database.
    Factor = (Ref_Val_from_ref_material / Meas_Val_from_sec_norm_rm)

    Args:
        sec_norm_enabled: bool
        sec_norm_rm: str. Name of measured RM group
        sec_norm_ref_material: str. Name of RM in database for reference values
        elements: list of (el, mass) tuples to normalise
        indexChannel: ChannelData object for output time series
    """
    if not sec_norm_enabled or not sec_norm_rm or not sec_norm_ref_material:
        return

    try:
        sg = data.selectionGroup(sec_norm_rm)
        ref_data = data.referenceMaterialData(sec_norm_ref_material)
    except Exception as e:
        IoLog.error(f"Failed to load secondary normalisation data: {e}")
        return

    for el, _ in elements:
        ratio_name = f"{el}/Ca"
        try:
            ts = data.timeSeries(ratio_name)
            meas_stats = sg.stats(ts)
            meas_val = meas_stats["mean"]

            if ratio_name not in ref_data:
                IoLog.warning(
                    f"No reference for {ratio_name} in {sec_norm_ref_material}. Skipping."
                )
                continue

            ref_val = ref_data[ratio_name].value()
            if meas_val == 0:
                IoLog.warning(
                    f"Measured {ratio_name} in {sec_norm_rm} is zero. Skipping."
                )
                continue

            factor = ref_val / meas_val
            IoLog.information(
                f"  {ratio_name}: Factor = {factor:.4f} ({ref_val:.6f} / {meas_val:.6f})"
            )

            corrected_data = ts.data() * factor
            data.createTimeSeries(
                ratio_name, data.Output, indexChannel.time(), corrected_data
            )

        except Exception as e:
            IoLog.error(f"Error applying secondary normalisation for {ratio_name}: {e}")


def find_rm_blocks():
    """
    Build standards blocks based on time gaps between neighbouring RM selections.

    A new block starts whenever the gap to the previous RM selection is longer than
    the mean gap across all RM selections, multiplied by a tuning factor.
    Inspired by the "simple" detection method in the "3D Trace Elements" DRS,
    but with a configurable cutoff.
    
    Returns: list of RMBlock objects sorted by time, or empty list if no blocks found
    """
    selections = []

    for rm_name in data.selectionGroupNames(data.ReferenceMaterial):
        sg = data.selectionGroup(rm_name)
        if sg:
            for sel in sg.selections():
                t = sel.midTimestamp if not sel.isLinked() else sel.linkedMidTimestamp()
                selections.append((t, sel, rm_name))

    if not selections:
        IoLog.error("No RM selections found in this session.")
        return []

    selections.sort(key=lambda x: x[0])
    sel_mid_times = [t for t, _, _ in selections]

    if len(sel_mid_times) == 1:
        diffs = np.array([0.0])
    else:
        diffs = np.insert(np.diff(sel_mid_times), 0, 0.0)

    sensitivity = drs.setting("BlockDetectionSensitivity")
    if sensitivity is None:
        sensitivity = 1.0

    cutoff = (float(np.mean(diffs)) if len(diffs) else 0.0) * float(sensitivity)

    labels = []
    current_label = 1
    for gap in diffs:
        if gap > cutoff:
            current_label += 1
        labels.append(current_label)

    blocks = []
    block_selections = {}

    for label, (t, sel, group_name) in zip(labels, selections):
        if label not in block_selections:
            block_selections[label] = []
        block_selections[label].append((t, sel, group_name))

    for label in sorted(block_selections):
        labeled_sels = block_selections[label]
        first_time = labeled_sels[0][0]
        last_time = labeled_sels[-1][0]
        block_time = (first_time + last_time) / 2.0

        current_block_sels = {}
        for _, sel, group_name in labeled_sels:
            if group_name not in current_block_sels:
                current_block_sels[group_name] = []
            current_block_sels[group_name].append(sel)

        blocks.append(RMBlock(block_time, current_block_sels))
    
    if not blocks:
        IoLog.error("No RM blocks detected in session.")
    
    return blocks


# ************************************
# Main coordinating function
# ************************************


def runDRS():
    """Main DRS entry point and coordinating function."""

    drs.message("Starting Element/Calcium ratios DRS")
    drs.progress(0)

    #
    # Step 1: Setup
    # ===========================
    settings = drs.settings()
    try:
        indexChannel = data.timeSeries(settings["IndexChannel"])
        maskOption = settings["Mask"]
        maskChannel = data.timeSeries(settings["MaskChannel"])
        cutoff = settings["MaskCutoff"]
        trim = settings["MaskTrim"]
        selected_elements = settings["Elements"]
        ca_channel = settings["CaChannel"]

        sec_norm_enabled = settings.get("SecondaryNorm", False)
        sec_norm_rm = settings.get("SecondaryNormRM", "")
        sec_norm_ref_material = settings.get("SecondaryNormRefMaterial", "")
    except Exception as e:
        IoLog.error(f"Could not read settings: {e}")
        drs_complete(error=True, error_message="Settings error. See messages.")
        return

    drs.setIndexChannel(indexChannel)
    drs.message("Making mask")
    mask = create_mask(maskOption, maskChannel, cutoff, trim, indexChannel)
    drs.progress(5)

    #
    # Step 2: Baseline subtraction
    # ===========================
    drs.message("Baseline subtracting")
    allInputChannels = data.timeSeriesList(data.Input)
    selected_elements, elements = filter_elements_to_available(
        selected_elements, allInputChannels
    )
    if not selected_elements:
        drs_complete(error=True, error_message="No valid elements selected.")
        return

    baseline_grp = get_baseline_group()
    if not baseline_grp:
        drs_complete(error=True, error_message="No baseline group found.")
        return

    drs.baselineSubtract(baseline_grp, allInputChannels, mask, 5, 30)
    drs.progress(30)

    #
    # Step 3: Calculate raw ratios
    # ===========================
    drs.message("Calculating raw ratios")
    raw_ratios_ok = calc_raw_ratios(ca_channel, selected_elements, indexChannel)
    if not raw_ratios_ok:
        return
    drs.progress(50)

    #
    # Step 4: Detect RM blocks
    # ===========================
    drs.message("Detecting standard-sample brackets")

    blocks = find_rm_blocks()
    if not blocks:
        drs_complete(
            error=True, error_message="Could not build RM blocks. Check selections."
        )
        return

    IoLog.information(f"Built {len(blocks)} contiguous RM blocks.")

    #
    # Step 5: Fit regressions
    # ===========================
    drs.progress(60)
    drs.message("Fitting regressions to each standards block")

    regr_results = fit_regressions_for_all_blocks(
        blocks,
        elements,
        ca_channel,
        min_rms_per_block=drs.setting("MinRMsPerBlock"),
        rm_selections=drs.setting("RMSelections"),
    )
    if not regr_results:
        drs_complete(
            error=True, error_message="No successful regressions. Check RM data."
        )
        return

    drs.progress(70)

    #
    # Step 6: Calibrate raw ratios
    # ===========================
    drs.message("Calibrating raw ratios")
    spline_type = drs.setting("SplineType")
    calibrate_ratios(
        regr_results, elements, ca_channel, indexChannel, spline_type=spline_type
    )
    drs.progress(80)

    #
    # Step 7: Apply secondary normalisation (optional)
    # ===========================
    if sec_norm_enabled:
        drs.message(
            f"Applying secondary normalisation: {sec_norm_rm} → {sec_norm_ref_material}"
        )
        apply_secondary_normalisation(
            sec_norm_enabled, sec_norm_rm, sec_norm_ref_material, elements, indexChannel
        )

    #
    # Step 8: Calculate derived channels (e.g. Li/Mg)
    # ===========================
    drs.message("Calculating derived channels")
    try:
        li_ca = data.timeSeries("Li/Ca").data()
        mg_ca = data.timeSeries("Mg/Ca").data()
        with np.errstate(divide="ignore", invalid="ignore"):
            li_mg = li_ca / mg_ca
            li_mg[~np.isfinite(li_mg)] = np.nan
        data.createTimeSeries("Li/Mg", data.Output, indexChannel.time(), li_mg)
    except (KeyError, Exception):
        IoLog.information("Li/Mg calculation skipped (missing Li/Ca or Mg/Ca).")

    drs_complete()


# ************************************
# UI stuff
# ************************************


class InputChannelsMenu(QtGui.QMenu):
    selectionChanged = QtCore.Signal(list)

    def __init__(self, parent, all_channels, current_selection):
        super().__init__(parent)
        self.all_channels = all_channels
        self.current_selection = list(current_selection)

        for ch in self.all_channels:
            a = QtGui.QWidgetAction(self)
            cb = QtGui.QCheckBox(ch, self)
            cb.setStyleSheet("QCheckBox { padding-left: 5px; margin: 3px; }")
            if ch in self.current_selection:
                cb.setChecked(True)

            a.setDefaultWidget(cb)
            self.addAction(a)
            cb.clicked.connect(partial(self.updateSelection, ch))

    def updateSelection(self, channel, checked):
        if checked:
            if channel not in self.current_selection:
                self.current_selection.append(channel)
        else:
            if channel in self.current_selection:
                self.current_selection.remove(channel)

        # Maintain order based on all_channels
        self.current_selection = [
            ch for ch in self.all_channels if ch in self.current_selection
        ]

        self.selectionChanged.emit(self.current_selection)


def settingsWidget():
    """
    Construct the user interface.
    """
    # Create an outer widget to handle centering
    outerWidget = QtGui.QWidget()
    outerLayout = QtGui.QHBoxLayout()
    outerWidget.setLayout(outerLayout)
    # Align the content of the layout to the center
    outerLayout.setAlignment(Qt.AlignHCenter)
    outerWidget.setStyleSheet("font-size: 10pt;")

    # Create the main content widget with fixed max width
    widget = QtGui.QWidget()
    widget.setMaximumWidth(1000)
    # Ensure the widget attempts to fill the available space up to max width
    widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)

    mainLayout = QtGui.QVBoxLayout()
    widget.setLayout(mainLayout)

    # Add main content widget to outer layout
    outerLayout.addWidget(widget)

    # --- Setup Global Colours ---
    rmNames = data.selectionGroupNames(data.ReferenceMaterial)
    for rm in rmNames:
        get_color(rm)

    # Identify default secondary RM (JCp...)
    default_sec_rm = ""
    for r in rmNames:
        if "jcp" in r.lower():
            default_sec_rm = r
            break
    if not default_sec_rm and rmNames:
        default_sec_rm = rmNames[0]

    # --- Get list of channels in this session ---
    timeSeriesNames = data.timeSeriesNames(data.Input)
    defaultChannelName = ""
    if timeSeriesNames:
        defaultChannelName = timeSeriesNames[0]

    caChannels = [c for c in timeSeriesNames if c.startswith("Ca4")]
    defaultCa = caChannels[0] if caChannels else defaultChannelName

    # --- Default Settings ---
    allElementNames = [
        ch.name
        for ch in data.timeSeriesList(data.Input)
        if not (ch.name.startswith("Ca") or ch.name.startswith("TotalBeam"))
    ]

    # Default settings
    drs.setSetting("IndexChannel", defaultChannelName)
    drs.setSetting("Mask", False)
    drs.setSetting("MaskChannel", defaultCa)
    drs.setSetting("MaskCutoff", 500000.0)
    drs.setSetting("MaskTrim", 0.0)
    drs.setSetting("CaChannel", defaultCa)
    drs.setSetting("FixMissingUnc", True)
    drs.setSetting("MissingUnc2RSD", 10.0)
    drs.setSetting("MinRMsPerBlock", 2)
    drs.setSetting("BlockDetectionSensitivity", 1.0)
    drs.setSetting("SplineType", "StepLinear")

    # Initialise Elements
    # In the case that the user has already worked in a previous session, we want to preserve their selection
    # but filter to available channels to be safe
    saved_elements = drs.setting("Elements")
    if saved_elements:
        # Filter saved elements to only include currently available ones
        elements_to_use = [el for el in saved_elements if el in allElementNames]
    else:
        # First run: use all available elements
        elements_to_use = allElementNames
    drs.setSetting("Elements", elements_to_use)

    # do the same for RM selections: preserve previous but default to all RMs for all elements if not set
    current_rm_sel = drs.setting("RMSelections")
    if not current_rm_sel:
        # Default to all RMs for all elements
        default_dict = {}
        for el in allElementNames:
            default_dict[el] = rmNames
        drs.setSetting("RMSelections", default_dict)
    else:
        # Filter existing selections to only include current RMs
        filtered_rm_dict = {}
        for el, sel_list in current_rm_sel.items():
            filtered_sel_list = [sel for sel in sel_list if sel in rmNames]
            filtered_rm_dict[el] = filtered_sel_list
        drs.setSetting("RMSelections", filtered_rm_dict)

    settings = drs.settings()

    # --- Initial Setup: Baseline Subtraction and Raw Ratios ---
    # Ensure both CPS channels and raw ratio channels exist for the preview
    try:
        # calc_raw_ratios will handle baseline subtraction if CPS channels don't exist
        ca_channel = settings["CaChannel"]
        idx_name = settings["IndexChannel"]
        if idx_name:
            idx_ch = data.timeSeries(idx_name)
            drs.setIndexChannel(idx_ch)

            selected_elements, elements = filter_elements_to_available(
                settings["Elements"], data.timeSeriesList(data.Input)
            )
            QtGui.QApplication.processEvents()

            calc_raw_ratios(ca_channel, selected_elements, idx_ch)

    except Exception as e:
        IoLog.error(
            f"Failed to run initial setup for DRS. See the messages tab for details."
        )

    # --- Notes ---
    note = QtGui.QLabel(
        "Note: This DRS calculates E/Ca ratios in the units stored in the reference material database. "
        + "Make sure all values are in the appropriate units and consistent between RMs before running the DRS. "
        + "Iolite interprets uncertainties in the reference material values as 2 standard deviations."
    )
    note.setWordWrap(True)
    mainLayout.addWidget(note)

    mainLayout.addSpacing(20)  # Vertical space

    # --- Split layout for calcium and elements ---
    ecGroup = QtGui.QGroupBox("E/Ca ratios")
    ecGroupLayout = QtGui.QHBoxLayout()
    ecGroup.setLayout(ecGroupLayout)
    mainLayout.addWidget(ecGroup)

    ecLabel = QtGui.QLabel("Calculate these ratios:")
    ecGroupLayout.addWidget(ecLabel)

    ecGroupLayout.addSpacing(10)

    # ToolButton for element selection
    elButton = QtGui.QToolButton(widget)
    elButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    elButton.setIcon(CUI().icon("checklist"))
    elButton.setPopupMode(QtGui.QToolButton.InstantPopup)

    # Menu
    elMenu = InputChannelsMenu(elButton, allElementNames, settings["Elements"])
    elButton.setMenu(elMenu)

    ecGroupLayout.addWidget(elButton)

    # Solidus
    solidusLabel = QtGui.QLabel("/")
    solidusLabel.setStyleSheet("font-size: 20px; font-weight: bold;")
    ecGroupLayout.addWidget(solidusLabel)

    # Calcium channel (denominator)
    caComboBox = QtGui.QComboBox(widget)
    caComboBox.addItems(caChannels)
    caComboBox.setCurrentText(settings["CaChannel"])
    ecGroupLayout.addWidget(caComboBox)

    # Push everything to the left
    ecGroupLayout.addStretch()

    mainLayout.addSpacing(20)  # Vertical space

    # --- Plotting & Preview Section ---
    previewGroup = QtGui.QGroupBox("Regression Preview")
    previewLayout = QtGui.QVBoxLayout()
    previewGroup.setLayout(previewLayout)

    # HBox for list and plot
    plotAreaLayout = QtGui.QHBoxLayout()

    # Left: RM list and controls
    rmLayout = QtGui.QVBoxLayout()

    # Ratio Selection (Moved here)
    ratioRow = QtGui.QHBoxLayout()
    ratioPlotCombo = QtGui.QComboBox(widget)
    ratioViewCombo = QtGui.QComboBox(widget)
    ratioViewCombo.addItems(["Overview", "Block: 1"])
    ratioViewCombo.setCurrentIndex(0)
    ratioRow.addWidget(ratioPlotCombo, 1)
    ratioRow.addWidget(ratioViewCombo)
    rmLayout.addWidget(QtGui.QLabel("Preview:"))
    rmLayout.addLayout(ratioRow)
    rmLayout.addSpacing(10)

    rmLabel = QtGui.QLabel("Reference materials:")
    rmLayout.addWidget(rmLabel)

    RMsListWidget = QtGui.QListWidget(widget)
    RMsListWidget.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
    RMsListWidget.setMinimumHeight(200)
    rmLayout.addWidget(RMsListWidget)

    missingUncLabel = QtGui.QLabel("* missing uncertainty in ref. value")
    missingUncLabel.setStyleSheet("font-style: italic; font-size: 9pt;")
    # Hidden by default, shown if needed later
    missingUncLabel.setVisible(False)
    rmLayout.addWidget(missingUncLabel)
    rmLayout.setSpacing(10)

    # List buttons
    listBtnsLayout = QtGui.QHBoxLayout()
    selAllBtn = QtGui.QPushButton("Select All")
    deSelAllBtn = QtGui.QPushButton("Deselect All")
    listBtnsLayout.addWidget(selAllBtn)
    listBtnsLayout.addWidget(deSelAllBtn)
    rmLayout.addLayout(listBtnsLayout)

    # Reset button
    resetBtn = QtGui.QPushButton("Reset all ratios")
    resetBtn.setToolTip(
        "Select all RMs for all ratios (except Secondary Norm RM if enabled)"
    )
    rmLayout.addWidget(resetBtn)

    # 'Missing uncertainty' controls
    uncCtrlLayout = QtGui.QVBoxLayout()
    uncCtrlLayout.addSpacing(10)
    uncCtrlLayout.setSpacing(2)

    fixUncCheck = QtGui.QCheckBox("Assume relative error if missing")
    fixUncCheck.setToolTip(
        "If a reference value has <= 0 uncertainty, assume a relative % error."
    )
    fixUncCheck.setChecked(drs.setting("FixMissingUnc"))

    uncInputLayout = QtGui.QHBoxLayout()
    uncInputLayout.addWidget(QtGui.QLabel("2RSD (%):"))
    uncSpinBox = QtGui.QDoubleSpinBox()
    uncSpinBox.setRange(0.1, 1000.0)
    uncSpinBox.setValue(drs.setting("MissingUnc2RSD"))
    uncSpinBox.setEnabled(fixUncCheck.isChecked())
    uncInputLayout.addWidget(uncSpinBox)

    uncCtrlLayout.addWidget(fixUncCheck)
    uncCtrlLayout.addLayout(uncInputLayout)
    rmLayout.addLayout(uncCtrlLayout)

    # Signals for unc controls
    def update_unc_fix_state(b):
        drs.setSetting("FixMissingUnc", bool(b))
        uncSpinBox.setEnabled(bool(b))
        updateRMListForRatio()

    fixUncCheck.toggled.connect(update_unc_fix_state)
    uncSpinBox.valueChanged.connect(
        lambda v: (drs.setSetting("MissingUnc2RSD", float(v)), refreshPlot())
    )

    plotAreaLayout.addLayout(rmLayout)
    plotAreaLayout.addSpacing(30)

    # Right: Plot
    PLOT.setMinimumHeight(500)
    PLOT.setMaximumHeight(700)
    PLOT.setMinimumWidth(500)
    PLOT.setMaximumWidth(700)
    plotAreaLayout.addWidget(PLOT, 1)

    previewLayout.addLayout(plotAreaLayout)
    mainLayout.addWidget(previewGroup)

    # --- Functions ---

    def set_all_rms_checked(state):
        try:
            RMsListWidget.blockSignals(True)
            count_prop = RMsListWidget.count
            n = count_prop() if callable(count_prop) else count_prop

            for i in range(n):
                item = RMsListWidget.item(i)
                item.setCheckState(Qt.Checked if state else Qt.Unchecked)

            RMsListWidget.blockSignals(False)
            save_current_rm_selection()
            refreshPlot()
        except Exception as e:
            IoLog.error(f"Error selecting all: {e}")

    def reset_all_ratios():
        try:
            # Iterate all elements in settings and reset their list to full
            all_els = drs.setting("Elements")
            full_rm_list = data.selectionGroupNames(data.ReferenceMaterial)

            new_dict = {}
            for el in all_els:
                new_dict[el] = full_rm_list
            drs.setSetting("RMSelections", new_dict)
            updateRMListForRatio()
        except Exception as e:
            IoLog.error(f"Error resetting: {e}")

    # Connect buttons
    selAllBtn.clicked.connect(lambda: set_all_rms_checked(True))
    deSelAllBtn.clicked.connect(lambda: set_all_rms_checked(False))
    resetBtn.clicked.connect(reset_all_ratios)

    def save_current_rm_selection():
        """Save check states of current list to settings dict"""
        try:
            curr_el = ratioPlotCombo.currentData
            if not curr_el:
                return

            if callable(curr_el):
                curr_el = curr_el()

            selected = []
            count_prop = RMsListWidget.count
            n = count_prop() if callable(count_prop) else count_prop
            for i in range(n):
                item = RMsListWidget.item(i)
                if item.checkState() == Qt.Checked:
                    # Use data to store clean RM name
                    t = item.data(Qt.UserRole)
                    selected.append(t)

            # Update Dict
            current_dict = drs.setting("RMSelections")
            if current_dict is None:
                current_dict = {}
            current_dict[curr_el] = selected
            drs.setSetting("RMSelections", current_dict)

        except Exception as e:
            IoLog.error(f"Error saving RM selection: {e}")

    def refreshPlot():
        PLOT.clearGraphs()
        ann.visible = False

        ca_chan = drs.setting("CaChannel")

        try:
            target_el = ratioPlotCombo.currentData
            if callable(target_el):
                target_el = target_el()
        except TypeError:
            target_el = ratioPlotCombo.currentData

        if not target_el:
            PLOT.replot()
            return

        # Get RM selections for current ratio
        rm_dict = drs.setting("RMSelections")
        if not rm_dict:
            rm_dict = {}

        selected_rms = rm_dict.get(target_el, [])

        if not selected_rms or len(selected_rms) < 2:
            IoLog.error("Fewer than 2 valid RMs found for regression.")
            PLOT.replot()
            return

        blocks = find_rm_blocks()
        if not blocks:
            IoLog.error("Block detection failed. Check selection groups and RM ordering.")
            return

        # Update view options based on block count
        prev_view = ratioViewCombo.currentText
        ratioViewCombo.blockSignals(True)
        ratioViewCombo.clear()
        ratioViewCombo.addItem("Overview")
        for i in range(len(blocks)):
            ratioViewCombo.addItem(f"Block: {i+1}")
        if prev_view in [
            ratioViewCombo.itemText(i) for i in range(ratioViewCombo.count)
        ]:
            ratioViewCombo.setCurrentText(prev_view)
        else:
            ratioViewCombo.setCurrentIndex(0)
        ratioViewCombo.blockSignals(False)

        # Plot for each block
        block_colors = [PLOT_COLOURS[i % len(PLOT_COLOURS)] for i in range(len(blocks))]
        all_x_global = []
        all_y_global = []
        max_x_global = 0.0
        max_y_global = 0.0
        block_plot_data = []
        block_fit_lines = []

        for block_idx, block in enumerate(blocks):
            result = get_block_regression_data(
                block, target_el, ca_chan, selected_rms, drs.setting("MinRMsPerBlock")
            )

            if not result:
                continue

            fit = result["fit"]
            data = result["data"]

            all_x_global.extend(data["x"])
            all_y_global.extend(data["y"])
            max_x_global = max(all_x_global) if all_x_global else 0.0
            max_y_global = max(all_y_global) if all_y_global else 0.0

            block_plot_data.append((block_idx, data["raw_stats"], data["rm_names"]))

            max_x = np.max(data["x"]) * 1.1
            x_range = np.linspace(0, max_x, 50)
            y_fit = lm([fit["slope"], fit["intercept"]], x_range)

            block_fit_lines.append(
                {
                    "idx": block_idx,
                    "x": x_range,
                    "y": y_fit,
                    "color": block_colors[block_idx],
                    **fit,  # slope, intercept, r_squared
                }
            )

        view_text = ratioViewCombo.currentText
        show_overview = view_text == "Overview"
        fit_index = None
        if not show_overview:
            try:
                fit_index = int(view_text.split(":")[-1].strip()) - 1
            except Exception:
                fit_index = None

        def plot_hollow_point(x_val, y_val, color, size):
            g = PLOT.addGraph()
            g.setLineStyle("lsNone")
            g.setScatterStyle("ssCircle", size, color, Qt.transparent)
            g.setData(np.array([x_val]), np.array([y_val]))
            return g

        def plot_mean_with_errors(x_val, y_val, x_err, y_err, color, size):
            g = PLOT.addGraph()
            g.setLineStyle("lsNone")
            g.setScatterStyle("ssDisc", size, color)
            g.setData(np.array([x_val]), np.array([y_val]))

            # x data
            try:
                eb_x = QCPErrorBars(PLOT.bottom(), PLOT.left())
                eb_x.setDataPlottable(g)
                eb_x.errorType = QCPErrorBars.etKeyError
                eb_x.setData(np.array([x_err]))
                eb_x.pen = QPen(color)
                eb_x.removeFromLegend()
            except:
                pass

            # y data
            try:
                eb_y = QCPErrorBars(PLOT.bottom(), PLOT.left())
                eb_y.setDataPlottable(g)
                eb_y.errorType = QCPErrorBars.etValueError
                eb_y.setData(np.array([y_err]))
                eb_y.pen = QPen(color)
                eb_y.removeFromLegend()
            except:
                pass

            return g

        if show_overview:
            # Plot fit lines first so points sit on top
            for fit_line in block_fit_lines:
                fg = PLOT.addGraph()
                fg.setName(f"Block {fit_line['idx']+1}")
                pen = QPen(fit_line["color"])
                pen.setWidth(1.5)
                pen.setStyle(Qt.SolidLine)
                fg.pen = pen
                fg.setData(fit_line["x"], fit_line["y"])

            # Plot individual selections as hollow circles (no error bars)
            for block_idx, block_data, block_rm_names in block_plot_data:
                for i, (meas_mean, ref_val, meas_err, ref_err) in enumerate(block_data):
                    rm_name = block_rm_names[i]
                    color = get_color(rm_name)
                    g = plot_hollow_point(meas_mean, ref_val, color, 6.0)
                    g.setName(f"{rm_name} (Selection {block_idx+1})")

            # Plot group means as solid circles with error bars
            plotted_means = set()
            for rm_name in selected_rms:
                if rm_name in plotted_means:
                    continue
                plotted_means.add(rm_name)

                stats = gather_ratio_stats(
                    target_el, ca_chan, rm_group=data.selectionGroup(rm_name)
                )
                if not stats:
                    continue

                meas_mean, ref_val, meas_err, ref_err = stats
                color = get_color(rm_name)

                g = plot_mean_with_errors(
                    meas_mean, ref_val, meas_err, ref_err, color, 9.0
                )
                g.setName(f"{rm_name} (Mean)")

            # For overview, find fit with median slope to display
            if block_fit_lines:
                median_idx = len(block_fit_lines) // 2
                median_fit = sorted(block_fit_lines, key=lambda x: x["slope"])[
                    median_idx
                ]
                ann_fit_data = median_fit
            else:
                ann_fit_data = None
        else:
            # Individual fit view for a single block, includes error bars
            if fit_index is None or fit_index < 0 or fit_index >= len(block_plot_data):
                print("Selected fit view is out of range.")
                ann_fit_data = None
            else:
                for fit_line in block_fit_lines:
                    if fit_line["idx"] == fit_index:
                        fg = PLOT.addGraph()
                        fg.setName(f"Block {fit_line['idx']+1}")
                        pen = QPen(fit_line["color"])
                        pen.setWidth(1.5)
                        pen.setStyle(Qt.SolidLine)
                        fg.pen = pen
                        fg.setData(fit_line["x"], fit_line["y"])
                        ann_fit_data = fit_line
                        break
                else:
                    ann_fit_data = None

                block_idx, block_data, block_rm_names = block_plot_data[fit_index]
                for i, (meas_mean, ref_val, meas_err, ref_err) in enumerate(block_data):
                    rm_name = block_rm_names[i]
                    color = get_color(rm_name)
                    g = plot_hollow_point(meas_mean, ref_val, color, 6.0)
                    g.setName(f"{rm_name} (Selection {block_idx+1})")

                    # x data
                    try:
                        eb_x = QCPErrorBars(PLOT.bottom(), PLOT.left())
                        eb_x.setDataPlottable(g)
                        eb_x.errorType = QCPErrorBars.etKeyError
                        eb_x.setData(np.array([meas_err]))
                        eb_x.pen = QPen(color)
                        eb_x.removeFromLegend()
                    except:
                        pass

                    # y data
                    try:
                        eb_y = QCPErrorBars(PLOT.bottom(), PLOT.left())
                        eb_y.setDataPlottable(g)
                        eb_y.errorType = QCPErrorBars.etValueError
                        eb_y.setData(np.array([ref_err]))
                        eb_y.pen = QPen(color)
                        eb_y.removeFromLegend()
                    except:
                        pass

        ann.visible = True

        # Build annotation text with ODR parameters
        view_label = "overview" if show_overview else f"block {fit_index+1}"
        ann_text = f'<p style="color:black;font-size:9pt;line-height:1.15;"><b>{target_el}/{ca_chan}</b> ({view_label})'

        if ann_fit_data is not None:
            ann_text += (
                f"<br />Typical fit (median slope):"
                if show_overview
                else f"<br />ODR fit parameters:"
            )
            slope = ann_fit_data["slope"]
            slope_unc = (ann_fit_data["slope_unc"] * 2) / slope * 100
            intercept = ann_fit_data["intercept"]
            intercept_unc = ann_fit_data["intercept_unc"] * 2
            r_sq = ann_fit_data["r_squared"]

            ann_text += f"<br />slope = {slope:.3g} ± {slope_unc:.1f}% (2RSD)"
            ann_text += f"<br />intercept = {intercept:.3g} ± {intercept_unc:.3g} (2SD)"
            ann_text += f"<br />R² = {r_sq:.3f}"

        ann_text += "</p>"
        ann.text = ann_text

        PLOT.rescaleAxes()

        # Start axes at 0, keep global ranges across views when available
        if max_x_global > 0 or max_y_global > 0:
            upper_x = max_x_global * 1.2
            upper_y = max_y_global * 1.2
        else:
            y_range = PLOT.left().range
            upper_y = y_range.upper() * 1.2
            x_range = PLOT.bottom().range
            upper_x = x_range.upper() * 1.2

        PLOT.left().setRange(QCPRange(0, upper_y))
        PLOT.bottom().setRange(QCPRange(0, upper_x))

        PLOT.replot()
        return

    def updateRMListForRatio():
        """Populate the RM list based on the selected ratio/element"""
        RMsListWidget.blockSignals(True)
        RMsListWidget.clear()

        try:
            target_el = ratioPlotCombo.currentData
            if callable(target_el):
                target_el = target_el()
        except:
            target_el = None

        if not target_el:
            RMsListWidget.blockSignals(False)
            return

        rm_dict = drs.setting("RMSelections")
        if not rm_dict:
            rm_dict = {}

        # If this element isn't in dict yet, add it with all RMs
        if target_el not in rm_dict:
            rm_dict[target_el] = rmNames
            drs.setSetting("RMSelections", rm_dict)

        selected_for_ratio = rm_dict.get(target_el, [])

        apply_fix_unc = drs.setting("FixMissingUnc")

        dict_modified = False
        has_missing_unc_any = False  # Track if any RM has missing uncertainty

        # Prepare regex for Element/Mass parsing to look up reference value
        match = re.match(r"([a-zA-Z]+)([0-9]+)", target_el)
        ref_lookup = None
        if match:
            el_sym = match.group(1)
            ref_lookup = f"{el_sym}/Ca"

        for name in rmNames:
            item = QtGui.QListWidgetItem()  # Text set later
            item.setData(Qt.UserRole, name)  # Store clean name

            flags = item.flags() | Qt.ItemIsUserCheckable

            # Simple square with fill colour
            base_color = get_color(name)
            pix = QtGui.QPixmap(12, 12)
            pix.fill(base_color)

            is_checked = name in selected_for_ratio

            # Check if RM has reference value
            has_ref_val = False
            ref_unc = 0.0

            if ref_lookup:
                rm_data = data.referenceMaterialData(name)
                if ref_lookup in rm_data:
                    has_ref_val = True
                    ref_unc = rm_data[ref_lookup].uncertainty()

            display_text = name

            # Determine specific conditions
            is_missing_unc = False
            if has_ref_val and (ref_unc <= 0 or np.isnan(ref_unc)):
                is_missing_unc = True
                has_missing_unc_any = True
                display_text += "*"

            if not has_ref_val:
                # RM missing any ref value
                item.setFlags(flags & ~Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                item.setToolTip(
                    f"{name} has no reference value for {ref_lookup} and cannot be used."
                )

                # Set icon to light grey and add question mark
                pix.fill(QtGui.QColor(220, 220, 220))
                painter = QtGui.QPainter(pix)
                painter.setPen(QtGui.QColor(100, 100, 100))
                painter.drawRect(0, 0, 11, 11)
                painter.setPen(QtGui.QColor(Qt.red))
                f = painter.font()
                f.setPixelSize(10)
                f.setBold(True)
                painter.setFont(f)
                painter.drawText(0, 0, 12, 12, Qt.AlignCenter, "?")
                painter.end()
                item.setIcon(QtGui.QIcon(pix))

                if is_checked:
                    selected_for_ratio = [r for r in selected_for_ratio if r != name]
                    dict_modified = True

            elif is_missing_unc and not apply_fix_unc:
                # missing uncertainty and user has fix disabled
                # Disable and uncheck
                item.setFlags(flags & ~Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                item.setToolTip(
                    f"{name} has missing uncertainty and 'Assume relative error' is disabled."
                )

                # set as empty (transparent) icon with light grey border
                pix = QtGui.QPixmap(12, 12)
                pix.fill(QtGui.QColor(200, 200, 200, 50))
                # border
                painter = QtGui.QPainter(pix)
                pen = QtGui.QPen(QtGui.QColor(150, 150, 150))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawRect(0, 0, 11, 11)
                painter.end()
                item.setIcon(QtGui.QIcon(pix))

                if is_checked:
                    selected_for_ratio = [r for r in selected_for_ratio if r != name]
                    dict_modified = True
            else:
                # Valid RM
                item.setFlags(flags | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
                item.setIcon(QtGui.QIcon(pix))

            item.setText(display_text)
            RMsListWidget.addItem(item)

        if dict_modified:
            rm_dict[target_el] = selected_for_ratio
            drs.setSetting("RMSelections", rm_dict)

        # Update missing uncertainty controls
        missingUncLabel.setVisible(has_missing_unc_any)

        # If no RMs have missing uncertainty, disable the controls entirely
        if has_missing_unc_any:
            fixUncCheck.setEnabled(True)
            uncSpinBox.setEnabled(fixUncCheck.isChecked())
            fixUncCheck.setToolTip(
                "If a reference value has <= 0 uncertainty, assume a relative % error."
            )
        else:
            fixUncCheck.setEnabled(False)
            uncSpinBox.setEnabled(False)
            fixUncCheck.setToolTip(
                "No RMs with missing uncertainty found for this element."
            )

        RMsListWidget.blockSignals(False)
        refreshPlot()

    def updateRatioCombo():
        """Update the El/Ca selection combobox using the current settings"""
        selected_els = drs.setting("Elements")
        ca_channel = drs.setting("CaChannel")

        # Re-populate Ratio combo, keeping selection if possible
        old_sel = ratioPlotCombo.currentData
        if callable(old_sel):
            old_sel = old_sel()

        ratioPlotCombo.blockSignals(True)
        ratioPlotCombo.clear()
        for el in selected_els:
            # Display: Li7/Ca43, Data: Li7
            ratioPlotCombo.addItem(f"{el}/{ca_channel}", el)

        # Restore old selection
        idx = ratioPlotCombo.findData(old_sel)
        if idx >= 0:
            ratioPlotCombo.setCurrentIndex(idx)
        elif ratioPlotCombo.count > 0:
            ratioPlotCombo.setCurrentIndex(0)

        ratioPlotCombo.blockSignals(False)

    def on_elements_changed(selected):
        drs.setSetting("Elements", selected)
        elButton.setText(f"Elements ({len(selected)} selected)")
        updateRatioCombo()        
        updateRMListForRatio()
    
    def on_Ca_channel_changed(new_Ca_channel):
        drs.setSetting("CaChannel", new_Ca_channel)
        selected_els = drs.setting("Elements")
        calc_raw_ratios(new_Ca_channel, selected_els, data.timeSeries(drs.setting("IndexChannel")))
        updateRatioCombo()
        refreshPlot()        

    def on_rm_checked_changed(item):
        save_current_rm_selection()
        refreshPlot()

    # --- Connections ---
    # Use itemChanged for check state changes
    elMenu.selectionChanged.connect(on_elements_changed)
    ratioPlotCombo.currentIndexChanged.connect(updateRMListForRatio)
    ratioViewCombo.currentIndexChanged.connect(refreshPlot)
    RMsListWidget.itemChanged.connect(on_rm_checked_changed)
    caComboBox.currentTextChanged.connect(on_Ca_channel_changed)

    # Init state
    try:
        on_elements_changed(settings["Elements"])
    except RuntimeError:
        # Iolite may have saved elements that aren't valid. If so, use all elements.
        drs.setSetting("Elements", allElementNames)

    mainLayout.addSpacing(20)  # Vertical space

    # --- Time-varying regression group ---
    tsrGroup = QtGui.QGroupBox("Time-varying regression")
    tsrLayout = QtGui.QVBoxLayout()
    tsrGroup.setLayout(tsrLayout)

    tsrLabel = QtGui.QLabel(
        "Fit regressions to blocks of reference materials, then interpolate with splines "
        "to account for instrumental drift over time."
    )
    tsrLabel.setWordWrap(True)
    tsrLayout.addWidget(tsrLabel)

    tsrControlsLayout = QtGui.QHBoxLayout()
    tsrLayout.addLayout(tsrControlsLayout)

    # Min RMs per block
    tsrControlsLayout.addWidget(QtGui.QLabel("Min RMs per block:"))
    minRMsSpinBox = QtGui.QSpinBox()
    minRMsSpinBox.setRange(1, 10)
    minRMsSpinBox.setValue(drs.setting("MinRMsPerBlock"))
    minRMsSpinBox.valueChanged.connect(
        lambda v: (drs.setSetting("MinRMsPerBlock", int(v)), refreshPlot())
    )
    tsrControlsLayout.addWidget(minRMsSpinBox)

    tsrControlsLayout.addSpacing(20)

    # Block detection sensitivity
    tsrControlsLayout.addWidget(QtGui.QLabel("Block detection sensitivity:"))
    blockDetectionSensitivitySpinBox = QtGui.QDoubleSpinBox()
    blockDetectionSensitivitySpinBox.setRange(0.1, 10.0)
    blockDetectionSensitivitySpinBox.setSingleStep(0.1)
    blockDetectionSensitivitySpinBox.setDecimals(2)
    blockDetectionSensitivitySpinBox.setValue(
        drs.setting("BlockDetectionSensitivity")
    )
    blockDetectionSensitivitySpinBox.setToolTip(
        "1.0 should be fine for most sessions. Increase it to require a longer gap between blocks, or decrease it to allow shorter gaps."
    )
    blockDetectionSensitivitySpinBox.valueChanged.connect(
        lambda v: (
            drs.setSetting("BlockDetectionSensitivity", float(v)),
            refreshPlot(),
        )
    )
    tsrControlsLayout.addWidget(blockDetectionSensitivitySpinBox)

    tsrControlsLayout.addSpacing(20)

    # Spline type
    tsrControlsLayout.addWidget(QtGui.QLabel("Spline type:"))
    splineCombo = QtGui.QComboBox()
    spline_types = [
        "MeanMean",
        "MeanMedian",
        "LinearFit",
        "WeightedLinearFit",
        "StepLinear",
        "StepForward",
        "StepBackward",
        "StepAverage",
        "Nearest",
        "Akima",
        "Spline_NoSmoothing",
        "Spline_Smooth1",
        "Spline_Smooth2",
        "Spline_Smooth3",
        "Spline_Smooth4",
        "Spline_Smooth5",
        "Spline_Smooth6",
        "Spline_Smooth7",
        "Spline_Smooth8",
        "Spline_Smooth9",
        "Spline_Smooth10",
        "Spline_AutoSmooth",
    ]
    splineCombo.addItems(spline_types)
    splineCombo.setCurrentText(drs.setting("SplineType"))
    splineCombo.currentTextChanged.connect(lambda t: drs.setSetting("SplineType", t))
    tsrControlsLayout.addWidget(splineCombo)

    tsrControlsLayout.addStretch()

    mainLayout.addWidget(tsrGroup)

    mainLayout.addSpacing(30)  # Vertical space

    # --- Secondary Normalisation Group ---
    secNormGroup = QtGui.QGroupBox("Secondary normalisation")
    secNormLayout = QtGui.QVBoxLayout()
    secNormGroup.setLayout(secNormLayout)

    secNormLabel = QtGui.QLabel(
        "After the regression calibration, apply a second normalisation factor derived from a reference material.\n"
        "Select the measured RM to use, and which reference values to correct to."
    )
    secNormLabel.setWordWrap(True)
    secNormLayout.addWidget(secNormLabel)

    snHLayout = QtGui.QHBoxLayout()
    secNormLayout.addLayout(snHLayout)

    secNormCheckBox = QtGui.QCheckBox("Enabled")
    secNormCheckBox.setChecked(drs.setting("SecondaryNorm"))
    snHLayout.addWidget(secNormCheckBox)

    snHLayout.addSpacing(20)

    snHLayout.addWidget(QtGui.QLabel("Measured RM:"))
    secNormCombo = QtGui.QComboBox()
    secNormCombo.addItems(rmNames)
    secNormCombo.setCurrentText(drs.setting("SecondaryNormRM"))
    snHLayout.addWidget(secNormCombo)

    snHLayout.addSpacing(30)

    snHLayout.addWidget(QtGui.QLabel("Reference values:"))
    allRefMaterials = data.referenceMaterialNames()
    secNormRefCombo = QtGui.QComboBox()
    secNormRefCombo.addItems(allRefMaterials)
    secNormRefCombo.setCurrentText(drs.setting("SecondaryNormRefMaterial"))
    snHLayout.addWidget(secNormRefCombo)

    snHLayout.addStretch()

    # Logic for Secondary Norm
    def update_sec_norm_ui():
        enabled = secNormCheckBox.isChecked()
        secNormCombo.setEnabled(enabled)
        secNormRefCombo.setEnabled(enabled)
        drs.setSetting("SecondaryNorm", enabled)
        drs.setSetting("SecondaryNormRM", secNormCombo.currentText)
        drs.setSetting("SecondaryNormRefMaterial", secNormRefCombo.currentText)

    secNormCheckBox.toggled.connect(update_sec_norm_ui)
    secNormCombo.currentTextChanged.connect(update_sec_norm_ui)
    secNormRefCombo.currentTextChanged.connect(update_sec_norm_ui)

    # Init UI state
    secNormCombo.setEnabled(secNormCheckBox.isChecked())
    secNormRefCombo.setEnabled(secNormCheckBox.isChecked())

    mainLayout.addWidget(secNormGroup)

    mainLayout.addSpacing(30)  # Vertical space

    # --- Bottom Row: Index and Mask ---
    bottomRowLayout = QtGui.QHBoxLayout()
    mainLayout.addLayout(bottomRowLayout)

    # --- Index Channel Group ---
    indexGroup = QtGui.QGroupBox("Index")
    indexLayout = QtGui.QHBoxLayout()
    indexGroup.setLayout(indexLayout)

    indexLayout.addWidget(QtGui.QLabel("Channel:"))

    # --- Index channel combo box ---
    indexComboBox = QtGui.QComboBox(widget)
    indexComboBox.addItems(timeSeriesNames)
    indexComboBox.setCurrentText(settings["IndexChannel"])
    indexComboBox.currentTextChanged.connect(
        lambda t: drs.setSetting("IndexChannel", t)
    )
    indexLayout.addWidget(indexComboBox)

    bottomRowLayout.addWidget(indexGroup)

    # --- Mask Section --
    maskGroup = QtGui.QGroupBox("Mask")
    maskGroup.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
    maskLayout = QtGui.QHBoxLayout()
    maskGroup.setLayout(maskLayout)

    maskCheckBox = QtGui.QCheckBox("Enabled")
    maskCheckBox.setChecked(drs.setting("Mask"))
    maskLayout.addWidget(maskCheckBox)

    maskLayout.addStretch()
    maskLayout.addWidget(QtGui.QLabel("Channel:"))
    maskComboBox = QtGui.QComboBox()
    maskComboBox.addItems(data.timeSeriesNames(data.Input))
    maskComboBox.setCurrentText(drs.setting("MaskChannel"))
    maskLayout.addWidget(maskComboBox)

    maskLayout.addStretch()
    maskLayout.addWidget(QtGui.QLabel("Cutoff:"))
    maskLineEdit = QtGui.QLineEdit(str(drs.setting("MaskCutoff")))
    maskLayout.addWidget(maskLineEdit)

    maskLayout.addStretch()
    maskLayout.addWidget(QtGui.QLabel("Trim:"))
    maskTrimLineEdit = QtGui.QLineEdit(str(drs.setting("MaskTrim")))
    maskLayout.addWidget(maskTrimLineEdit)

    # Enable/Disable logic for Mask
    def update_mask_ui(b):
        drs.setSetting("Mask", bool(b))
        maskComboBox.setEnabled(bool(b))
        maskLineEdit.setEnabled(bool(b))
        maskTrimLineEdit.setEnabled(bool(b))

    maskCheckBox.toggled.connect(update_mask_ui)
    maskComboBox.currentTextChanged.connect(lambda t: drs.setSetting("MaskChannel", t))
    maskLineEdit.textChanged.connect(lambda t: drs.setSetting("MaskCutoff", float(t)))
    maskTrimLineEdit.textChanged.connect(lambda t: drs.setSetting("MaskTrim", float(t)))

    # Init Mask UI state
    update_mask_ui(maskCheckBox.isChecked())

    bottomRowLayout.addWidget(maskGroup, 1)

    # --- Register settings widget with DRS ---
    drs.setSettingsWidget(outerWidget)
