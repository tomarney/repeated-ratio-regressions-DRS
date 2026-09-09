# Metadata
#/ Type: DRS
#/ Name: Repeated Ratio Regressions (R3)
#/ Authors: Thomas Arney
#/ Description: Calibration by time-varying regressions of measured and reference molar ratios.
#/ References: Tang et al. (2025, JAAS) DOI: 10.1039/D5JA00333D
#/ Version: 2.0
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

import re
from functools import partial

import numpy as np
from iolite import QtCore, QtGui
from iolite.Qt import QColor, Qt
from iolite.QtGui import QAction, QPen
from iolite.ui import IolitePlotPyInterface as Plot
from iolite.ui import IolitePlotSettingsDialog as PlotSettings
from iolite.ui import QCPErrorBars, QCPRange
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


def _parse_isotope_name(el_name):
    """Get element symbol and mass from isotope name (e.g. (Sr, 88) from Sr88)."""
    match = re.match(r"([a-zA-Z]+)([0-9]+)", el_name)
    return (match.group(1) if match else None, match.group(2) if match else None)


def _get_output_name(analyte, normaliser, mode):
    """Generate the channel name and RM DB lookup key based on output mode."""
    an_parsed = _parse_isotope_name(analyte)
    norm_parsed = _parse_isotope_name(normaliser)

    an_el = an_parsed[0] if an_parsed and an_parsed[0] else analyte
    an_mass = an_parsed[1] if an_parsed and an_parsed[1] else ""
    norm_el = norm_parsed[0] if norm_parsed and norm_parsed[0] else normaliser
    norm_mass = norm_parsed[1] if norm_parsed and norm_parsed[1] else ""

    if mode == "Delta notation":
        return f"d{an_mass}{an_el}"
    elif mode == "Isotopic ratios":
        return f"{an_mass}{an_el}/{norm_mass}{norm_el}"
    else:  # "Elemental ratios"
        return f"{an_el}/{norm_el}"


def gather_ratio_stats(
    target_isotope, norm_channel, mode, selection=None, rm_group=None
):
    """
    Gather measured and reference values and uncertainties for a given ratio.

    Args:
        target_isotope: str, isotope name (e.g. "Sr88")
        norm_channel: str, normalisation channel name (e.g. "Ca43")
        mode: str, ratio format to search for
        selection: Selection object (for individual block fits)
        rm_group: SelectionGroup object (for overview fit)

    Returns: tuple (meas_mean, ref_val, meas_unc, ref_unc) or None
    """
    ratio_channel_name = f"{target_isotope}_{norm_channel}_Raw"
    ref_lookup = _get_output_name(target_isotope, norm_channel, mode)

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
        IoLog.error(f"Could not get data for {target_isotope} in {rm_name}: {e}")
        return None


def filter_available_channels(selected_channels, allInputChannels):
    """
    Get channels which are both in the user-selected list and available in the input channels.
    """
    available_channels = [ch.name for ch in allInputChannels]
    return [el for el in selected_channels if el in available_channels]


def create_mask(isMaskDefined, maskChannel, cutoff, trim, indexChannel):
    """Create mask using iolite's cutoff function"""
    if isMaskDefined:
        mask = drs.createMaskFromCutoff(maskChannel, cutoff, trim)
    else:
        # if user has not defined a mask, return an array of ones (all True)
        mask = np.ones_like(indexChannel.data())

    data.createTimeSeries("mask", data.Intermediate, indexChannel.time(), mask)
    return mask


def subtract_baselines(indexChannel, maskOption, maskChannel, cutoff, trim):
    """Wrapper around iolite baselineSubtract() to handle missing baseline group"""

    baseline_grps = data.selectionGroupList(1)  # 1 = Baseline
    if baseline_grps is None or len(baseline_grps) == 0:
        drs_complete(error=True, error_message="No baseline group found.")
        return False
    if len(baseline_grps) > 1:
        IoLog.warning("Multiple baseline groups found. Using the first one.")
    bl_grp = baseline_grps[0]

    mask = np.ones_like(indexChannel.data())

    if maskOption:
        mask = create_mask(maskOption, maskChannel, cutoff, trim, indexChannel)

    drs.baselineSubtract(bl_grp, data.timeSeriesList(data.Input), mask, 5, 6)
    return True


def calc_raw_ratios(unique_pairs, indexChannel):
    """
    Calculate raw ratios from baseline-subtracted CPS channels for all required analyte/norm pairs.

    Returns true if successful and false otherwise
    """
    norms = {n for a, n in unique_pairs}
    norm_data_dict = {}

    for norm in norms:
        norm_cps_name = f"{norm}_CPS"
        try:
            norm_data_dict[norm] = data.timeSeries(norm_cps_name).data()
        except Exception as e:
            IoLog.error(
                f"No baseline-subtracted {norm_cps_name} found. DRS cannot proceed: {e}"
            )
            return False

    for ch in data.timeSeriesList(data.Intermediate):
        if not ch.name.endswith("_CPS"):
            continue
        ch_name = ch.name.replace("_CPS", "")

        for norm, norm_data in norm_data_dict.items():
            if (ch_name, norm) in unique_pairs:
                # suppress warnings about NaNs or dividing by zero
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = ch.data() / norm_data
                    ratio[~np.isfinite(ratio)] = np.nan

                ratio_name = f"{ch_name}_{norm}_Raw"
                data.createTimeSeries(
                    ratio_name, data.Intermediate, indexChannel.time(), ratio
                )

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
    block: RMBlock,
    target_isotope,
    norm_channel_name,
    mode,
    selected_rms=None,
    min_rms_per_block=2,
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
            stats = gather_ratio_stats(
                target_isotope, norm_channel_name, mode, selection=sel
            )
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
    blocks,
    analytes_list,
    norm_channel_name,
    mode,
    min_rms_per_block=2,
    rm_selections=None,
):
    """
    Fit regressions for all blocks and elements in an output set.
    """
    results_dict = {}

    for istp_name in analytes_list:
        output_name = _get_output_name(istp_name, norm_channel_name, mode)
        times = []
        slopes = []
        intercepts = []
        slopes_unc = []
        intercepts_unc = []
        r_squared = []

        selected_rms = rm_selections.get(output_name, []) if rm_selections else None

        for block_num, block in enumerate(blocks):
            if selected_rms is not None:
                viable_rms = sum(1 for rm in block.rm_sels if rm in selected_rms)
            else:
                viable_rms = len(block.rm_sels)

            if viable_rms < min_rms_per_block:
                IoLog.warning(
                    f"Block has {viable_rms} valid RM types for {output_name}. "
                    f"Need at least {min_rms_per_block}. Skipping."
                )
                continue

            result = get_block_regression_data(
                block,
                istp_name,
                norm_channel_name,
                mode,
                selected_rms,
                min_rms_per_block=min_rms_per_block,
            )
            if result is None:
                IoLog.warning(f"Could not fit {output_name} for block {block_num}")
                continue

            fit_result = result["fit"]
            times.append(block.time)
            slopes.append(fit_result["slope"])
            intercepts.append(fit_result["intercept"])
            slopes_unc.append(fit_result["slope_unc"])
            intercepts_unc.append(fit_result["intercept_unc"])
            r_squared.append(fit_result["r_squared"])

        if slopes:
            results_dict[istp_name] = {
                "times": np.array(times),
                "slopes": np.array(slopes),
                "intercepts": np.array(intercepts),
                "slopes_unc": np.array(slopes_unc),
                "intercepts_unc": np.array(intercepts_unc),
                "r_squared": np.array(r_squared),
            }

    return results_dict


def fit_splines_for_one_ratio(
    block_reg_results,
    isotope_name,
    norm_channel_name,
    mode,
    indexChannel,
    spline_type="StepLinear",
):
    """
    Fit the given spline to block regression results for a single normalised ratio.
    Creates intermediate channels for time-varying slope and intercept.
    """
    if isotope_name not in block_reg_results:
        return None, None

    reg_data = block_reg_results[isotope_name]
    times = reg_data["times"]
    slopes = reg_data["slopes"]
    intercepts = reg_data["intercepts"]
    slopes_unc = reg_data["slopes_unc"]
    intercepts_unc = reg_data["intercepts_unc"]

    out_name = _get_output_name(isotope_name, norm_channel_name, mode)
    safe_out_name = out_name.replace("/", "_")

    if len(times) < 2:
        IoLog.warning(
            f"Not enough valid blocks for {out_name} spline. Need at least 2."
        )
        return None, None

    slopes_unc = np.where(slopes_unc <= 0, np.abs(slopes) * 0.05, slopes_unc)
    intercepts_unc = np.where(intercepts_unc <= 0, 1e-6, intercepts_unc)

    try:
        slope_spl = data.spline(
            times, slopes, slopes_unc, spline_type, indexChannel.time()
        )
        intercept_spl = data.spline(
            times, intercepts, intercepts_unc, spline_type, indexChannel.time()
        )
    except Exception as e:
        IoLog.error(f"Failed to create splines for {out_name}: {e}")
        return None, None

    slope_channel_name = f"{safe_out_name}_slope"
    intercept_channel_name = f"{safe_out_name}_intercept"

    data.createTimeSeries(
        slope_channel_name, data.Intermediate, indexChannel.time(), slope_spl
    )
    data.createTimeSeries(
        intercept_channel_name, data.Intermediate, indexChannel.time(), intercept_spl
    )

    return slope_spl, intercept_spl


def calibrate_ratios(
    block_reg_results,
    analytes_list,
    norm_channel_name,
    mode,
    indexChannel,
    spline_type="StepLinear",
):
    """Calibrate raw ratios using time-varying (splined) slope/intercept values."""

    for istp_name in analytes_list:
        slope_spl, intercept_spl = fit_splines_for_one_ratio(
            block_reg_results,
            istp_name,
            norm_channel_name,
            mode,
            indexChannel,
            spline_type,
        )

        out_name = _get_output_name(istp_name, norm_channel_name, mode)

        if slope_spl is None or intercept_spl is None:
            IoLog.warning(f"Could not create splines for {out_name}. Skipping.")
            continue

        try:
            raw_ts = data.timeSeries(f"{istp_name}_{norm_channel_name}_Raw")
        except Exception as e:
            IoLog.error(
                f"Raw ratio channel {istp_name}_{norm_channel_name}_Raw not found: {e}"
            )
            continue

        raw_data = raw_ts.data()
        corrected_data = slope_spl * raw_data + intercept_spl

        data.createTimeSeries(
            out_name,
            data.Output,
            indexChannel.time(),
            corrected_data,
        )


def apply_secondary_normalisation(
    sec_norm_rm,
    sec_norm_ref_material,
    analytes_list,
    norm_channel,
    mode,
    indexChannel,
):
    """
    Apply secondary normalisation using reference material database.
    Factor = (Ref_Val_from_ref_material / Meas_Val_from_sec_norm_rm)
    """
    if not sec_norm_rm or not sec_norm_ref_material:
        return

    try:
        sg = data.selectionGroup(sec_norm_rm)
        ref_data = data.referenceMaterialData(sec_norm_ref_material)
    except Exception as e:
        IoLog.error(f"Failed to load secondary normalisation data: {e}")
        return

    for el_name in analytes_list:
        ratio_name_meas = _get_output_name(el_name, norm_channel, mode)
        ratio_name_ref = ratio_name_meas

        try:
            ts = data.timeSeries(ratio_name_meas)
            meas_stats = sg.stats(ts)
            meas_val = meas_stats["mean"]

            if ratio_name_ref not in ref_data:
                IoLog.warning(
                    f"No reference for {ratio_name_ref} in {sec_norm_ref_material}. Skipping."
                )
                continue

            ref_val = ref_data[ratio_name_ref].value()
            if meas_val == 0:
                IoLog.warning(
                    f"Measured {ratio_name_meas} in {sec_norm_rm} is zero. Skipping."
                )
                continue

            factor = ref_val / meas_val
            IoLog.information(
                f"  {ratio_name_meas}: Factor = {factor:.4f} ({ref_val:.6f} / {meas_val:.6f})"
            )

            corrected_data = ts.data() * factor
            data.createTimeSeries(
                ratio_name_meas, data.Output, indexChannel.time(), corrected_data
            )

        except Exception as e:
            IoLog.error(
                f"Error applying secondary normalisation for {ratio_name_meas}: {e}"
            )


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

    drs.message("Starting R3 DRS")
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
        output_sets = settings.get("OutputSets", [])
    except Exception as e:
        IoLog.error(f"Could not read settings: {e}")
        drs_complete(error=True, error_message="Settings error. See messages.")
        return

    if not output_sets:
        drs_complete(error=True, error_message="No outputs defined.")
        return

    unique_pairs = set()
    for o in output_sets:
        n = o["normaliser"]
        for a in o["analytes"]:
            if a in data.timeSeriesNames(data.Input) and n in data.timeSeriesNames(
                data.Input
            ):
                unique_pairs.add((a, n))

    if not unique_pairs:
        drs_complete(
            error=True, error_message="No valid analytes selected across outputs."
        )
        return

    drs.setIndexChannel(indexChannel)
    drs.progress(5)

    #
    # Step 2: Baseline subtraction
    # ===========================
    drs.message("Baseline subtracting")
    subtract_baselines(indexChannel, maskOption, maskChannel, cutoff, trim)
    drs.progress(10)

    #
    # Step 3: Calculate raw ratios
    # ===========================
    drs.message("Calculating raw ratios")
    raw_ratios_ok = calc_raw_ratios(unique_pairs, indexChannel)
    if not raw_ratios_ok:
        return
    drs.progress(15)

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
    # Step 5: Process each output set
    # ===========================
    spline_type = drs.setting("SplineType")
    min_rms = drs.setting("MinRMsPerBlock")
    rm_selections = drs.setting("RMSelections")

    for i, o_set in enumerate(output_sets):
        drs.progress(30 + (i / len(output_sets)) * 60)
        drs.message(f"Processing output set {i + 1}...")

        analytes = o_set["analytes"]
        norm = o_set["normaliser"]
        mode = o_set["mode"]

        valid_analytes = [a for a in analytes if (a, norm) in unique_pairs]
        if not valid_analytes:
            continue

        #
        # Step 5.1: Fit regressions
        # ===========================

        regr_results = fit_regressions_for_all_blocks(
            blocks, valid_analytes, norm, mode, min_rms, rm_selections
        )

        if not regr_results:
            IoLog.warning(f"No successful regressions for Output {i + 1}.")
            continue

        #
        # Step 5.1: Calibrate raw ratios
        # ===========================

        calibrate_ratios(
            regr_results,
            valid_analytes,
            norm,
            mode,
            indexChannel,
            spline_type=spline_type,
        )

        #
        # Step 5.1: Apply secondary normalisation (optional)
        # ===========================

        if o_set.get("sec_norm_enabled"):
            IoLog.information(f"Applying secondary normalisation to Output {i + 1}")
            apply_secondary_normalisation(
                o_set["sec_norm_rm"],
                o_set["sec_norm_ref_material"],
                valid_analytes,
                norm,
                mode,
                indexChannel,
            )

    drs.progress(90)
    drs_complete()


# ************************************
# UI stuff
# ************************************

# Tip: much easier to see if you fold to level 2 (Ctrl+K, Ctrl+2 in VSCode)


class InputChannelsMenu(QtGui.QMenu):
    selectionChanged = QtCore.Signal(list)

    def __init__(self, parent, all_channels, current_selection, disabled_channel=None):
        super().__init__(parent)
        self.all_channels = all_channels
        self.current_selection = list(current_selection)
        self.disabled_channel = disabled_channel
        self.actions_map = {}

        for ch in self.all_channels:
            a = QtGui.QWidgetAction(self)
            cb = QtGui.QCheckBox(ch, self)
            cb.setStyleSheet("QCheckBox { padding-left: 5px; margin: 3px; }")

            if ch in self.current_selection:
                cb.setChecked(True)

            a.setDefaultWidget(cb)
            self.addAction(a)
            self.actions_map[ch] = (a, cb)

            cb.clicked.connect(partial(self.updateSelection, ch))

        if self.disabled_channel:
            self.setChannelDisabled(self.disabled_channel, True)

    def setChannelDisabled(self, channel, disabled):
        if channel in self.actions_map:
            action, cb = self.actions_map[channel]
            if disabled:
                cb.setChecked(False)
                cb.setEnabled(False)
                action.setEnabled(False)
                if channel in self.current_selection:
                    self.current_selection.remove(channel)
            else:
                cb.setEnabled(True)
                action.setEnabled(True)

    def setChannelChecked(self, channel, checked):
        if channel in self.actions_map:
            _, cb = self.actions_map[channel]
            cb.setChecked(checked)
            if checked:
                if channel not in self.current_selection:
                    self.current_selection.append(channel)
            else:
                if channel in self.current_selection:
                    self.current_selection.remove(channel)

            self.current_selection = [
                ch for ch in self.all_channels if ch in self.current_selection
            ]

    def updateSelection(self, channel, checked):
        if checked:
            if channel not in self.current_selection:
                self.current_selection.append(channel)
        else:
            if channel in self.current_selection:
                self.current_selection.remove(channel)

        self.current_selection = [
            ch for ch in self.all_channels if ch in self.current_selection
        ]

        self.selectionChanged.emit(self.current_selection)


class OutputSetRow(QtGui.QWidget):
    """Encapsulates the UI and state for a single group of analyte/normaliser output settings."""

    dataChanged = QtCore.Signal()
    addRequested = QtCore.Signal()
    removeRequested = QtCore.Signal(object)

    def __init__(self, parent_drs_widget, index, state=None):
        super().__init__()
        self.parent_drs = parent_drs_widget
        self.index = index
        self.all_channels = parent_drs_widget.allIsotopeChannels
        self.rm_names = parent_drs_widget.rmNames
        self.ref_mat_names = data.referenceMaterialNames()

        main_layout = QtGui.QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left Buttons Layout
        btn_group = QtGui.QGroupBox(
            " "
        )  # hack to align... I'm sure there's a better way
        btn_layout = QtGui.QHBoxLayout(btn_group)
        self.removeBtn = QtGui.QToolButton()
        self.removeBtn.setText("-")
        self.removeBtn.setToolTip("Remove this output")
        self.removeBtn.clicked.connect(lambda: self.removeRequested.emit(self))

        self.addBtn = QtGui.QToolButton()
        self.addBtn.setText("+")
        self.addBtn.setToolTip("Add an output")
        self.addBtn.clicked.connect(lambda checked=False: self.addRequested.emit())

        btn_layout.addWidget(self.removeBtn)
        btn_layout.addWidget(self.addBtn)
        main_layout.addWidget(btn_group)
        main_layout.addSpacing(10)

        # Output parameters group
        self.group = QtGui.QGroupBox(f"Set {index}")
        output_group_layout = QtGui.QHBoxLayout(self.group)

        self.analytesSelector = QtGui.QToolButton()
        self.analytesSelector.setText("Channels")
        self.analytesMenu = InputChannelsMenu(
            self.analytesSelector,
            self.all_channels,
            state.get("analytes", []) if state else [],
            disabled_channel=state.get("normaliser") if state else None,
        )
        self.analytesSelector.setMenu(self.analytesMenu)
        self.analytesSelector.setPopupMode(QtGui.QToolButton.InstantPopup)
        output_group_layout.addWidget(self.analytesSelector)

        lbl = QtGui.QLabel("/")
        lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
        output_group_layout.addWidget(lbl)

        self.normCombo = QtGui.QComboBox()
        self.normCombo.addItems(self.all_channels)
        if state and state.get("normaliser"):
            self.normCombo.setCurrentText(state.get("normaliser"))
        self.current_norm = self.normCombo.currentText
        output_group_layout.addWidget(self.normCombo)
        output_group_layout.addSpacing(10)

        self.modeCombo = QtGui.QComboBox()
        self.modeCombo.addItems(
            ["Elemental ratios", "Isotopic ratios", "Delta notation"]
        )
        self.modeCombo.setToolTip(
            'How to calibrate and present the output ratios:\n as elemental ratios (e.g. "Sr/Ca"), as isotopic ratios (e.g. "B11/C12"), or in delta notation (e.g. "d11B").\n This affects how the reference values are searched for in the RM database and the names of the output channels.'
        )

        if state and state.get("mode"):
            self.modeCombo.setCurrentText(state.get("mode"))
        output_group_layout.addWidget(self.modeCombo)

        output_group_layout.addSpacing(10)
        main_layout.addWidget(self.group)
        main_layout.addSpacing(10)

        # Secondary Normalisation Side
        sec_norm_group = QtGui.QGroupBox("Secondary normalisation")
        sec_norm_layout = QtGui.QHBoxLayout(sec_norm_group)

        self.secNormCheck = QtGui.QCheckBox("Enabled")
        if state:
            self.secNormCheck.setChecked(state.get("sec_norm_enabled", False))
        self.secNormCheck.setToolTip(
            "Apply a second normalisation factor derived from a reference material."
        )
        sec_norm_layout.addWidget(self.secNormCheck)
        sec_norm_layout.addSpacing(10)
        sec_norm_layout.addLayout(sec_norm_layout)

        self.snRmCombo = QtGui.QComboBox()
        self.snRmCombo.addItems(self.rm_names)
        if state and state.get("sec_norm_rm"):
            self.snRmCombo.setCurrentText(state["sec_norm_rm"])

        self.snRefCombo = QtGui.QComboBox()
        self.snRefCombo.addItems(self.ref_mat_names)
        if state and state.get("sec_norm_ref_material"):
            self.snRefCombo.setCurrentText(state["sec_norm_ref_material"])

        sec_norm_layout.addWidget(QtGui.QLabel("Measured RM:"))
        sec_norm_layout.addWidget(self.snRmCombo)
        sec_norm_layout.addWidget(QtGui.QLabel("Ref Values:"))
        sec_norm_layout.addWidget(self.snRefCombo)

        main_layout.addWidget(sec_norm_group)
        main_layout.addStretch()

        # Connect signals
        self.analytesMenu.selectionChanged.connect(self._emit_changed)
        self.normCombo.currentTextChanged.connect(self._norm_changed)
        self.modeCombo.currentTextChanged.connect(self._emit_changed)
        self.secNormCheck.toggled.connect(self._sn_toggled)
        self.snRmCombo.currentTextChanged.connect(self._emit_changed)
        self.snRefCombo.currentTextChanged.connect(self._emit_changed)

        self._sn_toggled(self.secNormCheck.isChecked())
        self._update_analytes_btn_text()

    def _sn_toggled(self, checked):
        self.snRmCombo.setEnabled(checked)
        self.snRefCombo.setEnabled(checked)
        self._emit_changed()

    def _norm_changed(self, new_norm):
        if self.current_norm:
            self.analytesMenu.setChannelDisabled(self.current_norm, False)
        self.analytesMenu.setChannelDisabled(new_norm, True)
        self.current_norm = new_norm
        self._emit_changed()

    def _update_analytes_btn_text(self):
        count = len(self.analytesMenu.current_selection)
        self.analytesSelector.setText(f"Channels ({count} selected)")

    def _emit_changed(self, *args):
        self._update_analytes_btn_text()
        self.dataChanged.emit()

    def setIndex(self, idx):
        self.index = idx
        self.group.setTitle(f"Set {idx}")
        self.removeBtn.setEnabled(idx > 1)

    def getState(self):
        return {
            "analytes": list(self.analytesMenu.current_selection),
            "normaliser": self.normCombo.currentText,
            "mode": self.modeCombo.currentText,
            "sec_norm_enabled": self.secNormCheck.isChecked,
            "sec_norm_rm": self.snRmCombo.currentText,
            "sec_norm_ref_material": self.snRefCombo.currentText,
        }


class R3SettingsWidget(QtGui.QWidget):
    """
    Main UI widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.rmNames = data.selectionGroupNames(data.ReferenceMaterial)
        self.timeSeriesNames = data.timeSeriesNames(data.Input)
        self.allIsotopeChannels = [
            ch for ch in self.timeSeriesNames if ch not in ["TotalBeam"]
        ]

        self.defaultChannelName = (
            self.timeSeriesNames[0] if self.timeSeriesNames else ""
        )

        self.init_default_settings()
        self.run_initial_setup()  # must run before setup_ui() so that all calculated channels are available for the plot
        self.setup_ui()
        self.connect_signals()
        QtGui.QApplication.processEvents()

    def init_default_settings(self):
        drs.setSetting("IndexChannel", self.defaultChannelName)
        drs.setSetting("Mask", False)
        drs.setSetting("MaskChannel", self.defaultChannelName)
        drs.setSetting("MaskCutoff", 500000.0)
        drs.setSetting("MaskTrim", 0.0)

        # set preferences for default normalising channel in this list
        # Only relevant for very first use: otherwise iolite remembers the last used channel.
        # use either the element symbol (matches the first isotope of that element) or the full isotope name (e.g. "Ca44")
        # The first match in the list is used.
        norm_preferences = ["Ca43", "Ca44", "Ca"]
        default_norm = self.defaultChannelName

        # Match against preference hierarchy
        for pref in norm_preferences:
            found = False
            for ch in self.allIsotopeChannels:
                elem, _ = _parse_isotope_name(ch)
                if ch == pref or elem == pref:
                    default_norm = ch
                    found = True
                    break
            if found:
                break

        drs.setSetting("FixMissingUnc", True)
        drs.setSetting("MissingUnc2RSD", 10.0)
        drs.setSetting("MinRMsPerBlock", 2)
        drs.setSetting("BlockDetectionSensitivity", 1.0)
        drs.setSetting("SplineType", "StepLinear")

        default_analytes = [ch for ch in self.allIsotopeChannels if ch != default_norm]

        saved_sets = drs.setting("OutputSets")
        if not saved_sets:
            drs.setSetting(
                "OutputSets",
                [
                    {
                        "analytes": default_analytes,
                        "normaliser": default_norm,
                        "mode": "Elemental ratios",
                        "sec_norm_enabled": False,
                        "sec_norm_rm": "",
                        "sec_norm_ref_material": "",
                    }
                ],
            )

        current_rm_sel = drs.setting("RMSelections")
        if not current_rm_sel:
            drs.setSetting("RMSelections", {})

    def run_initial_setup(self):
        try:
            settings = drs.settings()
            output_sets = settings.get("OutputSets", [])
            if not output_sets:
                return

            idx_name = settings["IndexChannel"]
            if idx_name:
                idx_ch = data.timeSeries(idx_name)
                drs.setIndexChannel(idx_ch)

                unique_pairs = set()
                unique_analytes = set()
                for o in output_sets:
                    n = o["normaliser"]
                    for a in o["analytes"]:
                        unique_pairs.add((a, n))
                        unique_analytes.add(a)

                for istp in unique_analytes:
                    if f"{istp}_CPS" not in data.timeSeriesNames(data.Intermediate):
                        # At least one selected input channel is missing CPS channel
                        # (re)run baseline subtraction
                        subtract_baselines(
                            idx_ch,
                            settings["Mask"],
                            data.timeSeries(settings["MaskChannel"]),
                            settings["MaskCutoff"],
                            settings["MaskTrim"],
                        )
                        break

                for an, norm in unique_pairs:
                    if f"{an}_{norm}_Raw" not in data.timeSeriesNames(
                        data.Intermediate
                    ):
                        # At least one output is missing raw ratio channel
                        calc_raw_ratios(unique_pairs, idx_ch)
                        break
        except Exception as e:
            IoLog.warning(f"Failed to run initial setup for DRS: {e}")

    def setup_ui(self):
        """Coordinates the layout and adds widgets to the main window."""
        # Outer layout (centering)
        outerLayout = QtGui.QHBoxLayout(self)
        outerLayout.setAlignment(Qt.AlignHCenter)
        self.setStyleSheet("font-size: 10pt;")

        # Main content widget
        self.contentWidget = QtGui.QWidget()
        self.contentWidget.setMaximumWidth(1000)
        self.contentWidget.setSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred
        )
        self.mainLayout = QtGui.QVBoxLayout(self.contentWidget)
        outerLayout.addWidget(self.contentWidget)

        for rm in self.rmNames:
            get_color(rm)

        note = QtGui.QLabel(
            "Note: This DRS calculates calibrated ratios in the units stored in the reference material database. "
            "Make sure all values are in the appropriate units and consistent between RMs before running the DRS. "
            "Iolite interprets uncertainties in the reference material values as 2 standard deviations."
        )
        note.setWordWrap(True)
        self.mainLayout.addWidget(note)
        self.mainLayout.addSpacing(20)

        # --- outputs definition ---
        self.setup_outputs_group()
        self.mainLayout.addSpacing(20)

        # --- Preview ---
        self.setup_preview_group()

        # --- Time-varying regression ---
        self.setup_regression_group()

        # --- Bottom row (Index/Mask) ---
        self.setup_bottom_row()

    def setup_outputs_group(self):
        self.outputsContainer = QtGui.QWidget()
        self.outputsLayout = QtGui.QVBoxLayout(self.outputsContainer)
        self.outputsLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addWidget(self.outputsContainer)

        self.outputRows = []
        saved_sets = drs.setting("OutputSets")
        for state in saved_sets:
            self.add_output_row(state)

    def add_output_row(self, state=None):
        idx = len(self.outputRows) + 1
        # If adding a new row dynamically, duplicate the prior row's config
        # (maybe should just be a default? This is easier...)
        if not state and self.outputRows:
            state = self.outputRows[-1].getState()

        row = OutputSetRow(self, idx, state)
        row.dataChanged.connect(self.save_outputs_state)
        row.addRequested.connect(self.add_output_row)
        row.removeRequested.connect(self.remove_output_row)

        self.outputRows.append(row)
        self.outputsLayout.addWidget(row)
        self.outputsLayout.addSpacing(10)
        self.update_row_indices()
        self.save_outputs_state()

    def remove_output_row(self, row):
        if len(self.outputRows) <= 1:
            return
        self.outputRows.remove(row)
        self.outputsLayout.removeWidget(row)
        row.deleteLater()
        self.update_row_indices()
        self.save_outputs_state()

    def update_row_indices(self):
        for i, row in enumerate(self.outputRows):
            row.setIndex(i + 1)

    def save_outputs_state(self):
        sets = [row.getState() for row in self.outputRows]
        drs.setSetting("OutputSets", sets)

        # Recalculate raw pairs on change
        idx_ch_name = drs.setting("IndexChannel")
        if idx_ch_name:
            idx_ch = data.timeSeries(idx_ch_name)
            unique_pairs = set()
            for s in sets:
                for a in s["analytes"]:
                    unique_pairs.add((a, s["normaliser"]))
            calc_raw_ratios(unique_pairs, idx_ch)

        self.update_previewed_ratio_combo()

    def setup_preview_group(self):
        previewGroup = QtGui.QGroupBox("Regression Preview")
        plotAreaLayout = QtGui.QHBoxLayout(previewGroup)

        # Left side RM layout
        rmLayout = QtGui.QVBoxLayout()

        previewControlsRow = QtGui.QHBoxLayout()
        self.previewRatioSelector = QtGui.QComboBox(self.contentWidget)
        self.previewBlockSelector = QtGui.QComboBox(self.contentWidget)
        self.previewBlockSelector.addItems(["Overview", "Block: 1"])

        previewControlsRow.addWidget(self.previewRatioSelector, 1)
        previewControlsRow.addWidget(self.previewBlockSelector)

        rmLayout.addWidget(QtGui.QLabel("Preview:"))
        rmLayout.addLayout(previewControlsRow)
        rmLayout.addSpacing(10)
        rmLayout.addWidget(QtGui.QLabel("Reference materials:"))

        self.previewRMsList = QtGui.QListWidget(self.contentWidget)
        self.previewRMsList.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.previewRMsList.setMinimumHeight(200)
        rmLayout.addWidget(self.previewRMsList)

        self.missingUncLabel = QtGui.QLabel("* missing uncertainty in ref. value")
        self.missingUncLabel.setStyleSheet("font-style: italic; font-size: 9pt;")
        self.missingUncLabel.setVisible(False)
        rmLayout.addWidget(self.missingUncLabel)
        rmLayout.setSpacing(10)

        # List buttons
        listBtnsLayout = QtGui.QHBoxLayout()
        self.selAllBtn = QtGui.QPushButton("Select All")
        self.deSelAllBtn = QtGui.QPushButton("Deselect All")
        listBtnsLayout.addWidget(self.selAllBtn)
        listBtnsLayout.addWidget(self.deSelAllBtn)
        rmLayout.addLayout(listBtnsLayout)

        self.resetBtn = QtGui.QPushButton("Reset all ratios")
        self.resetBtn.setToolTip("Select all RMs for all ratios")
        rmLayout.addWidget(self.resetBtn)

        # Missing uncertainty controls
        uncCtrlLayout = QtGui.QVBoxLayout()
        uncCtrlLayout.addSpacing(10)
        uncCtrlLayout.setSpacing(2)

        self.fixUncCheck = QtGui.QCheckBox("Assume relative error if missing")
        self.fixUncCheck.setChecked(drs.setting("FixMissingUnc"))

        uncInputLayout = QtGui.QHBoxLayout()
        uncInputLayout.addWidget(QtGui.QLabel("2RSD (%):"))
        self.uncSpinBox = QtGui.QDoubleSpinBox()
        self.uncSpinBox.setRange(0.1, 1000.0)
        self.uncSpinBox.setValue(drs.setting("MissingUnc2RSD"))
        self.uncSpinBox.setEnabled(self.fixUncCheck.isChecked())

        uncInputLayout.addWidget(self.uncSpinBox)
        uncCtrlLayout.addWidget(self.fixUncCheck)
        uncCtrlLayout.addLayout(uncInputLayout)
        rmLayout.addLayout(uncCtrlLayout)

        plotAreaLayout.addLayout(rmLayout)
        plotAreaLayout.addSpacing(30)

        # Right hand side plot
        PLOT.setMinimumHeight(500)
        PLOT.setMaximumHeight(700)
        PLOT.setMinimumWidth(500)
        PLOT.setMaximumWidth(700)
        plotAreaLayout.addWidget(PLOT, 1)

        self.mainLayout.addWidget(previewGroup)
        self.mainLayout.addSpacing(20)

    def setup_regression_group(self):
        tsrGroup = QtGui.QGroupBox("Time-varying regression")
        tsrLayout = QtGui.QVBoxLayout(tsrGroup)

        tsrLabel = QtGui.QLabel(
            "Fit regressions to blocks of reference materials, then interpolate with splines..."
        )
        tsrLabel.setWordWrap(True)
        tsrLayout.addWidget(tsrLabel)

        tsrControlsLayout = QtGui.QHBoxLayout()

        # Min RMs
        tsrControlsLayout.addWidget(QtGui.QLabel("Min RMs per block:"))
        self.minRMsSpinBox = QtGui.QSpinBox()
        self.minRMsSpinBox.setRange(1, 10)
        self.minRMsSpinBox.setValue(drs.setting("MinRMsPerBlock"))
        tsrControlsLayout.addWidget(self.minRMsSpinBox)
        tsrControlsLayout.addSpacing(20)

        # Detection sensitivity
        tsrControlsLayout.addWidget(QtGui.QLabel("Block detection sensitivity:"))
        self.blockDetectionSensitivitySpinBox = QtGui.QDoubleSpinBox()
        self.blockDetectionSensitivitySpinBox.setRange(0.1, 10.0)
        self.blockDetectionSensitivitySpinBox.setSingleStep(0.1)
        self.blockDetectionSensitivitySpinBox.setValue(
            drs.setting("BlockDetectionSensitivity")
        )
        tsrControlsLayout.addWidget(self.blockDetectionSensitivitySpinBox)
        tsrControlsLayout.addSpacing(20)

        # Spline type
        tsrControlsLayout.addWidget(QtGui.QLabel("Spline type:"))
        self.splineTypeSelector = QtGui.QComboBox()
        # no clever way I can see to get this list so just hardcode as 3DTE does
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
        self.splineTypeSelector.addItems(spline_types)
        self.splineTypeSelector.setCurrentText(drs.setting("SplineType"))
        tsrControlsLayout.addWidget(self.splineTypeSelector)
        tsrControlsLayout.addStretch()

        tsrLayout.addLayout(tsrControlsLayout)
        self.mainLayout.addWidget(tsrGroup)
        self.mainLayout.addSpacing(30)

    def setup_bottom_row(self):
        bottomRowLayout = QtGui.QHBoxLayout()

        indexGroup = QtGui.QGroupBox("Index")
        indexLayout = QtGui.QHBoxLayout(indexGroup)
        indexLayout.addWidget(QtGui.QLabel("Channel:"))
        self.indexComboBox = QtGui.QComboBox(self.contentWidget)
        self.indexComboBox.addItems(self.timeSeriesNames)
        self.indexComboBox.setCurrentText(drs.setting("IndexChannel"))
        indexLayout.addWidget(self.indexComboBox)
        bottomRowLayout.addWidget(indexGroup)
        bottomRowLayout.addSpacing(20)

        maskGroup = QtGui.QGroupBox("Mask (click Crunch Data to apply)")
        maskLayout = QtGui.QHBoxLayout(maskGroup)
        self.maskCheckBox = QtGui.QCheckBox("Enabled")
        self.maskCheckBox.setChecked(drs.setting("Mask"))
        maskLayout.addWidget(self.maskCheckBox)

        maskLayout.addSpacing(20)
        maskLayout.addWidget(QtGui.QLabel("Channel:"))
        self.maskChannelComboBox = QtGui.QComboBox()
        self.maskChannelComboBox.addItems(data.timeSeriesNames(data.Input))
        self.maskChannelComboBox.setCurrentText(drs.setting("MaskChannel"))
        maskLayout.addWidget(self.maskChannelComboBox)

        maskLayout.addSpacing(10)
        maskLayout.addWidget(QtGui.QLabel("Cutoff:"))
        self.maskCutoffInput = QtGui.QLineEdit(str(drs.setting("MaskCutoff")))
        maskLayout.addWidget(self.maskCutoffInput)

        maskLayout.addSpacing(10)
        maskLayout.addWidget(QtGui.QLabel("Trim:"))
        self.maskTrimInput = QtGui.QLineEdit(str(drs.setting("MaskTrim")))
        maskLayout.addWidget(self.maskTrimInput)
        maskLayout.addStretch()

        bottomRowLayout.addWidget(maskGroup, 1)

        self.mainLayout.addLayout(bottomRowLayout)

        self.on_mask_toggle_changed(self.maskCheckBox.isChecked())

    def connect_signals(self):
        """Connect the handlers for all the UI inputs"""
        self.previewRatioSelector.currentIndexChanged.connect(
            self.update_rm_list_for_preview
        )
        self.previewBlockSelector.currentIndexChanged.connect(self.refresh_plot)
        self.previewRMsList.itemChanged.connect(self.on_rm_checked_changed)
        self.selAllBtn.clicked.connect(lambda: self.set_all_rms_checked(True))
        self.deSelAllBtn.clicked.connect(lambda: self.set_all_rms_checked(False))
        self.resetBtn.clicked.connect(self.reset_all_ratios)
        self.fixUncCheck.toggled.connect(self.update_unc_fix_state)
        self.uncSpinBox.valueChanged.connect(
            lambda v: (drs.setSetting("MissingUnc2RSD", float(v)), self.refresh_plot())
        )
        self.minRMsSpinBox.valueChanged.connect(
            lambda v: (drs.setSetting("MinRMsPerBlock", int(v)), self.refresh_plot())
        )
        self.blockDetectionSensitivitySpinBox.valueChanged.connect(
            lambda v: (
                drs.setSetting("BlockDetectionSensitivity", float(v)),
                self.refresh_plot(),
            )
        )
        self.splineTypeSelector.currentTextChanged.connect(
            lambda t: drs.setSetting("SplineType", t)
        )
        self.indexComboBox.currentTextChanged.connect(
            lambda t: drs.setSetting("IndexChannel", t)
        )
        self.maskCheckBox.toggled.connect(self.on_mask_toggle_changed)
        self.maskChannelComboBox.currentTextChanged.connect(
            lambda t: drs.setSetting("MaskChannel", t)
        )
        self.maskCutoffInput.textChanged.connect(
            lambda t: drs.setSetting("MaskCutoff", float(t))
        )
        self.maskTrimInput.textChanged.connect(
            lambda t: drs.setSetting("MaskTrim", float(t))
        )

        self.update_previewed_ratio_combo()

    def set_all_rms_checked(self, state):
        try:
            self.previewRMsList.blockSignals(True)
            n = self.previewRMsList.count
            for i in range(n):
                self.previewRMsList.item(i).setCheckState(
                    Qt.Checked if state else Qt.Unchecked
                )
            self.previewRMsList.blockSignals(False)
            self.save_current_rm_selection()
            self.refresh_plot()
        except Exception as e:
            IoLog.error(f"Error selecting all: {e}")

    def reset_all_ratios(self):
        """Select all RMs in all ratios"""
        try:
            sets = drs.setting("OutputSets") or []
            new_dict = {}
            for o in sets:
                norm = o["normaliser"]
                mode = o["mode"]
                for a in o["analytes"]:
                    out_name = _get_output_name(a, norm, mode)
                    new_dict[out_name] = self.rmNames
            drs.setSetting("RMSelections", new_dict)
            self.update_rm_list_for_preview()
        except Exception as e:
            IoLog.error(f"Error resetting: {e}")

    def save_current_rm_selection(self):
        """Save the current RM selections for the currently selected ratio"""
        try:
            combo_data = self.previewRatioSelector.currentData
            if not combo_data:
                return

            out_name = combo_data[0]
            selected = []
            n = self.previewRMsList.count
            for i in range(n):
                item = self.previewRMsList.item(i)
                if item.checkState() == Qt.Checked:
                    selected.append(item.data(Qt.UserRole))

            current_dict = drs.setting("RMSelections") or {}
            current_dict[out_name] = selected
            drs.setSetting("RMSelections", current_dict)
        except Exception as e:
            IoLog.error(f"Error saving RM selection: {e}")

    def on_rm_checked_changed(self):
        """
        Handle the user checking or unchecking an RM in the list.
        Saves the current selection and refreshes the plot.
        """
        self.save_current_rm_selection()
        self.refresh_plot()

    def on_mask_toggle_changed(self, b):
        """Handle mask being checked or unchecked"""
        drs.setSetting("Mask", bool(b))
        self.maskChannelComboBox.setEnabled(bool(b))
        self.maskCutoffInput.setEnabled(bool(b))
        self.maskTrimInput.setEnabled(bool(b))

    def update_unc_fix_state(self, b):
        """
        Handle the 'assume error if missing' checkbox being toggled.
        Updates the RM list.
        """
        drs.setSetting("FixMissingUnc", bool(b))
        self.uncSpinBox.setEnabled(bool(b))
        self.update_rm_list_for_preview()

    def update_previewed_ratio_combo(self):
        """Update the plot preview ratio dropdown after the user changes the inputs"""
        if not hasattr(self, "previewRatioSelector"):
            return

        old_data = self.previewRatioSelector.currentData
        self.previewRatioSelector.blockSignals(True)
        self.previewRatioSelector.clear()

        sets = drs.setting("OutputSets") or []
        for i, o_set in enumerate(sets):
            norm = o_set["normaliser"]
            mode = o_set["mode"]
            for analyte in o_set["analytes"]:
                out_name = _get_output_name(analyte, norm, mode)
                disp_text = f"{out_name}"
                self.previewRatioSelector.addItem(
                    disp_text, (out_name, analyte, norm, mode)
                )

        idx = -1
        if old_data:
            for i in range(self.previewRatioSelector.count):
                if self.previewRatioSelector.itemData(i) == old_data:
                    idx = i
                    break
        self.previewRatioSelector.setCurrentIndex(max(idx, 0))
        self.previewRatioSelector.blockSignals(False)
        self.update_rm_list_for_preview()

    def update_rm_list_for_preview(self):
        """Update the list of RMs when the user changes the previewed ratio"""
        self.previewRMsList.blockSignals(True)
        self.previewRMsList.clear()

        combo_data = self.previewRatioSelector.currentData
        if not combo_data:
            self.previewRMsList.blockSignals(False)
            return

        out_name, target_ch, norm_channel, mode = combo_data
        rm_dict = drs.setting("RMSelections") or {}

        if out_name not in rm_dict:
            rm_dict[out_name] = self.rmNames
            drs.setSetting("RMSelections", rm_dict)

        selected_for_ratio = rm_dict.get(out_name, [])
        apply_fix_unc = drs.setting("FixMissingUnc")
        has_missing_unc_any = False
        ref_lookup = out_name

        for name in self.rmNames:
            item = QtGui.QListWidgetItem()  # Text set later
            item.setData(Qt.UserRole, name)  # Store clean name
            flags = item.flags() | Qt.ItemIsUserCheckable

            pix = QtGui.QPixmap(12, 12)
            pix.fill(get_color(name))

            has_ref_val = False
            ref_unc = 0.0
            if ref_lookup:
                rm_data = data.referenceMaterialData(name)
                if ref_lookup in rm_data:
                    has_ref_val = True
                    ref_unc = rm_data[ref_lookup].uncertainty()

            display_text = name
            is_missing_unc = has_ref_val and (ref_unc <= 0 or np.isnan(ref_unc))
            if is_missing_unc:
                has_missing_unc_any = True
                display_text += "*"

            if not has_ref_val or (is_missing_unc and not apply_fix_unc):
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
            else:
                item.setFlags(flags | Qt.ItemIsEnabled)
                item.setCheckState(
                    Qt.Checked if name in selected_for_ratio else Qt.Unchecked
                )
                item.setIcon(QtGui.QIcon(pix))

            item.setText(display_text)
            self.previewRMsList.addItem(item)

        self.missingUncLabel.setVisible(has_missing_unc_any)
        self.fixUncCheck.setEnabled(has_missing_unc_any)
        self.uncSpinBox.setEnabled(has_missing_unc_any and self.fixUncCheck.isChecked())

        self.previewRMsList.blockSignals(False)
        self.refresh_plot()

    def refresh_plot(self):
        """Update the regression preview plot after user changes inputs"""
        PLOT.clearItems()
        PLOT.clearGraphs()
        combo_data = self.previewRatioSelector.currentData

        if not combo_data:
            PLOT.replot()
            return

        out_name, target_ch, norm_channel, mode = combo_data

        ratio_channel_name = f"{target_ch}_{norm_channel}_Raw"

        if ratio_channel_name not in data.timeSeriesNames(data.Intermediate):
            PLOT.replot()
            return

        selected_rms = drs.setting("RMSelections").get(out_name, [])
        if len(selected_rms) < 2:
            PLOT.replot()
            return

        blocks = find_rm_blocks()
        if not blocks:
            return

        # save current block selection, update combobox, and restore selection if still valid
        prev_view = self.previewBlockSelector.currentText
        self.previewBlockSelector.blockSignals(True)
        self.previewBlockSelector.clear()
        self.previewBlockSelector.addItem("Overview")
        for i in range(len(blocks)):
            self.previewBlockSelector.addItem(f"Block: {i + 1}")
        self.previewBlockSelector.setCurrentText(
            prev_view
            if prev_view
            in [
                self.previewBlockSelector.itemText(i)
                for i in range(self.previewBlockSelector.count)
            ]
            else "Overview"
        )
        self.previewBlockSelector.blockSignals(False)

        # prepare data for plotting
        all_x_global, all_y_global = [], []
        block_plot_data, block_fit_lines = [], []
        block_colors = [PLOT_COLOURS[i % len(PLOT_COLOURS)] for i in range(len(blocks))]

        for block_idx, block in enumerate(blocks):
            result = get_block_regression_data(
                block,
                target_ch,
                norm_channel,
                mode,
                selected_rms,
                drs.setting("MinRMsPerBlock"),
            )
            if not result:
                continue

            block_fit = result["fit"]
            block_data = result["data"]

            all_x_global.extend(block_data["x"])
            all_y_global.extend(block_data["y"])
            block_plot_data.append(
                (block_idx, block_data["raw_stats"], block_data["rm_names"])
            )

            max_x = np.max(block_data["x"])
            min_x = np.min(block_data["x"])
            x_range = np.linspace(min(0, min_x) * 10, max_x * 10, 50)
            if mode == "Delta notation":
                x_range = np.linspace(-max_x * 10, max_x * 10, 50)
            y_fit = block_fit["slope"] * x_range + block_fit["intercept"]

            block_fit_lines.append(
                {
                    "idx": block_idx,
                    "x": x_range,
                    "y": y_fit,
                    "color": block_colors[block_idx],
                    **block_fit,
                }
            )

        view_text = self.previewBlockSelector.currentText
        show_overview = view_text == "Overview"
        fit_index = None if show_overview else int(view_text.split(":")[-1].strip()) - 1

        if mode == "Delta notation":
            PLOT.left().label = f"Reference {out_name} (‰)"
            PLOT.bottom().label = f"Measured {target_ch}/{norm_channel}"
        else:
            PLOT.left().label = f"Reference {out_name}"
            PLOT.bottom().label = f"Measured {out_name}"

        def plot_point_with_error(x, y, x_err, y_err, color, size, hollow=True):
            g = PLOT.addGraph()
            g.setLineStyle("lsNone")
            g.setScatterStyle(
                "ssCircle" if hollow else "ssDisc",
                size,
                color,
                Qt.transparent if hollow else color,
            )
            g.setData(np.array([x]), np.array([y]))

            if not hollow:
                try:
                    eb_x = QCPErrorBars(PLOT.bottom(), PLOT.left())
                    eb_x.setDataPlottable(g)
                    eb_x.errorType = QCPErrorBars.etKeyError
                    eb_x.setData(np.array([x_err]))
                    eb_x.pen = QPen(color)
                    eb_x.removeFromLegend()
                except Exception as e:
                    IoLog.warning(f"Failed to plot X errors: {e}")

                try:
                    eb_y = QCPErrorBars(PLOT.bottom(), PLOT.left())
                    eb_y.setDataPlottable(g)
                    eb_y.errorType = QCPErrorBars.etValueError
                    eb_y.setData(np.array([y_err]))
                    eb_y.pen = QPen(color)
                    eb_y.removeFromLegend()
                except Exception as e:
                    IoLog.warning(f"Failed to plot Y errors: {e}")
            return g

        # plot/replot the data
        ann_fit_data = None
        if show_overview:
            # Showing overview, so plot all block results plus means
            for line in block_fit_lines:
                fg = PLOT.addGraph()
                pen = QPen(line["color"])
                pen.setWidth(1.5)
                fg.pen = pen
                fg.setData(line["x"], line["y"])

            # plot hollow points for all blocks
            for b_idx, stats, names in block_plot_data:
                for (meas_mean, ref_val, _, _), name in zip(stats, names):
                    plot_point_with_error(
                        meas_mean, ref_val, 0, 0, get_color(name), 6.0, True
                    )

            for rm_name in selected_rms:
                stats = gather_ratio_stats(
                    target_ch, norm_channel, mode, rm_group=data.selectionGroup(rm_name)
                )
                if not stats:
                    continue

                meas_mean, ref_val, meas_err, ref_err = stats
                plot_point_with_error(
                    meas_mean,
                    ref_val,
                    meas_err,
                    ref_err,
                    get_color(rm_name),
                    6.0,
                    False,
                )

            if block_fit_lines:
                ann_fit_data = sorted(block_fit_lines, key=lambda x: x["slope"])[
                    len(block_fit_lines) // 2
                ]
        else:
            # Showing single block so plot just this block's data
            if fit_index is not None and 0 <= fit_index < len(block_plot_data):
                line = next((l for l in block_fit_lines if l["idx"] == fit_index), None)
                if line:
                    fg = PLOT.addGraph()
                    pen = QPen(line["color"])
                    pen.setWidth(1.5)
                    fg.pen = pen
                    fg.setData(line["x"], line["y"])
                    ann_fit_data = line

                b_idx, stats, names = block_plot_data[fit_index]
                for (meas_mean, ref_val, meas_err, ref_err), name in zip(stats, names):
                    plot_point_with_error(
                        meas_mean,
                        ref_val,
                        meas_err,
                        ref_err,
                        get_color(name),
                        6.0,
                        False,
                    )

        # Annotate and scale axes
        if ann_fit_data:
            ann = PLOT.annotate(
                "", 0.015, 0.01, "ptAxisRectRatio", Qt.AlignLeft | Qt.AlignTop
            )

        # Build annotation text with ODR parameters
        view_label = "overview" if show_overview else f"block {fit_index + 1}"
        ann_text = f'<p style="color:black;font-size:9pt;line-height:1.15;"><b>{out_name}</b> ({view_label})'

        if ann_fit_data is not None:
            ann_text += (
                "<br />Typical fit (median slope):"
                if show_overview
                else "<br />ODR fit parameters:"
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
        if ann_fit_data:
            ann.text = ann_text

        def _get_robust_ax_lims(data, margin=0.2, pad_factor=(1.0, 1.0)):
            lo = min(data)
            if mode != "Delta notation":
                lo = min(lo, 0.0)
            hi = max(data)
            pad = (hi - lo) * margin
            return lo - pad * pad_factor[0], hi + pad * pad_factor[1]

        PLOT.rescaleAxes()

        if all_x_global and all_y_global:
            y_lims = _get_robust_ax_lims(all_y_global, pad_factor=(1.0, 2))
            x_lims = _get_robust_ax_lims(all_x_global)
            if mode != "Delta notation":
                y_lims = (0, y_lims[1])
                x_lims = (0, x_lims[1])
            PLOT.left().setRange(QCPRange(*y_lims))
            PLOT.bottom().setRange(QCPRange(*x_lims))

        PLOT.replot()


def settingsWidget():
    widget = R3SettingsWidget()
    drs.setSettingsWidget(widget)
