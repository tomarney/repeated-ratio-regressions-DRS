# Repeated Ratio Regressions (R3)

This is an [iolite](https://iolite.xyz/index.php) Data Reduction Scheme (DRS) which calibrates measured ratios in unknowns by fitting regressions of measured and reference ratios in multiple reference materials (RMs), and then interpolating the regression parameters over the course of the analytical session.

## Installation

Download the [latest release](https://github.com/tomarney/repeated-ratio-regressions-DRS/releases/latest) and unzip it. Then, in the iolite DRS tab, click "import" (top left), find where you unzipped the `r3.py` file, and click "open". I recommend clicking "import once" on the next screen for now, while the plugin is still changing a lot.

## How to use

1. Make sure iolite's internal reference material (RM) database (Data tab [left bar] > Reference Materials tab [top bar]) is set up properly:
   - All the RMs you will use must be present in the database (i.e. show up in the side bar), ideally with names very similar to what you usually call them in the run files you will import (for automatic detection later - but you can manually match them). The group they're in doesn't matter.
   - Each RM should have an entry for the elemental ratio you want to calibrate, named with a slash and without units (e.g., `Sr/Ca`). Unfortunately, iolite only accepts a very restricted set of units in that field, so you can't (for instance) enter `mmol/mol` into the Units column.
   - Make sure you use consistent units for any particular ratio across all RMs. If one RM is very high or very low on the regression plot compared to the others, this may be why.
2. After importing a session and creating your selections, make sure you have your primary standards in the "Reference materials" group. If you want to use one of them as a secondary standard, put it into the "Unknowns" group.
3. In the DRS tab, import the `r3.py` file if you haven't already. Then click the "Repeated Ratio Regressions (R3)" name to open the DRS.
4. The DRS will try to guess the preset you might want based on the channels present in your data. If it's wrong, just choose a different preset and click "Apply". Alternatively, edit the settings yourself: choose the analytes (isotopes/elements) you want to calculate ratios for from the drop-down list, the normalizer to use, and how you would like the results processed and presented (elemental/isotopic ratios, delta notation). If you would like different combinations of settings (e.g. two different normalizers, or one channel as an elemental ratio and another in delta notation), add a second output set and configure it.
5. Decide whether you want your results to be normalised to one of your standards. If you do, select the measured selection group and then the values you want to normalise to. They can be different - for instance, you could save different published values as new RMs in the data tab and then normalise to each one here.
7. Preview the regression for each analyte/normalizer ratio:
   - if any regressions look like a poor fit because a standard seems to have consistently mismatching data, you can excluse it from the fit for that element ratio by unticking it from the list. The plot will update, and that RM will not be used in the final processing for that ratio.
   - The initial view is all lines of best fit and all data points for all standards blocks: an overview. You can examine each detected standards block individually by choosing it in the drop-down (tip: you can use your mousewheel when hovering over it or your arrow keys after selecting it to scroll quickly).
   - If there's a few fits that seem off but most of the data points seem fine, you won't want to exclude the entire RM. Go back to the data ("Time Series" tab) to check the individual outlier selections for inclusions, spikes, or other bad data.
8. Choose your spline type carefully: set it to "StepLinear", click "Crunch Data!", go back to the "Time Series" tab, choose one of the slope and/or intercept intermediate channels (e.g. `Sr_Ca43_slope`), select the RM closest to the middle of the blocks, and then change the spline type to see how it looks. Note that this DRS doesn't use the per-RM spline type as defined in this screen; this is just a preview, and it won't change anything for this DRS. Also note that the axis limits will scale so that any variation is maximisied, so double check to see if there's a real change over the session. Once you've found a spline you're happy with, go back to the DRS screen and select that type from the drop-down. I will try and make this easier in the future!
9. Click "Crunch Data!"

## Feedback

Please report anything unexpected (however small) using the issues tab on Github or by emailing <t.arney@soton.ac.uk>. If you notice it, please report it! Real users are the best way to find problems. I will either help explain what's happening or I'll try and fix it. Thank you!
