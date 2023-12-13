#!/usr/bin/env python

"""profile_plot

Plot data from a Slurm HDF5 profile file generated with sh5util.

Writes a 4-panel plot into a PNG file.
 - CPU utilisation
 - Memory Utilisation (RSS)
 - I/O Rate
 - Cumulative I/O

Usage:
  profile_plot [--cpu=c] [--jobid=j] FILE [<step_name>...]

Options:
  --cpu=c    Max size of CPU scale.
  --jobid=j  Job ID from which to look up step names.

"""
import matplotlib
from subprocess import check_output
from matplotlib import pyplot as plt, colors, ticker
from docopt import docopt
import numpy
import math
import re
import h5py

to_rgba = colors.ColorConverter().to_rgba
matplotlib.use("agg")


class NotNegFormatter(ticker.ScalarFormatter):
    # For I/O rate I use negative Y axis positions to represent positive output rates.
    def __call__(self, x, pos=None):
        return super(type(self), self).__call__(abs(x), pos)


def the_first(seq):
    seq = iter(seq)
    item = next(seq)
    return item


def pick_time_scale(max_value):
    scale = 1
    unit = "seconds"
    if max_value / scale > 120:
        scale *= 60
        unit = "minutes"
    if max_value / scale > 120:
        scale *= 60
        unit = "hours"
    return (scale, unit)


def pick_mega_scale(max_value, max_number=1000):
    # Given a value in MB (which is what Slurm uses), a tidy unit to plot with is...
    base = 1024
    power = max(0, int(1 + math.log(max(1, max_value / max_number), base)))
    return (base**power, "MGTP"[power])


# list(f['Steps']['0']['Nodes']['compute-b1-065']['Tasks']['0'])[0]
# fields = f.values()[0].values()[0].values()[0].values()[0].values()[0].values()[0].dtype.fields
# dtype([('ElapsedTime', '<u8'), ('EpochTime', '<u8'),
# ('CPUFrequency', '<u8'), ('CPUTime', '<f8'), ('CPUUtilization', '<f8'),
# ('RSS', '<u8'), ('VMSize', '<u8'), ('Pages', '<u8'), ('ReadMB', '<f8'), ('WriteMB', '<f8')])
# (30, 1450264708, 2701000, 23.62, 78.73333333333333, 5290260, 5490448, 0, 517.7017669677734, 0.01776123046875)

# COMMAND LINE ARGUMENTS

args = docopt(__doc__)
fn = args["FILE"]
specified_steps = args["<step_name>"]
max_cpu = int(args["--cpu"] or 128)
job_id = args["--jobid"] or None

if job_id is None:
    m = re.match("job_([0-9_]+).h5", fn)
    if m is not None:
        (job_id,) = m.groups()

# Message so user knows whats happening.
print(f"Plotting '{fn}'...")

# OVERALL COUNT OF STEPS / TASKS

f = h5py.File(fn, "r")
steps = f["Steps"]
n_tasks = n_nodes = 0
step_order = []
batch_step = None
GPU = False
for step_name, step in steps.items():
    if specified_steps and step_name not in specified_steps:
        continue
    if step_name == "batch" or batch_step is None:
        batch_step = step
    if step_name not in step_order:
        step_order.append(step_name)
    for node in step["Nodes"].values():
        n_nodes += 1
        for task in node.get("Tasks", {}).values():
            n_tasks += 1
            if (not GPU) and ("GPUUtilization" in task.dtype.fields):
                GPU = True

assert step_order, "no steps found"

# BATCH (or, failing that, first) STEP AS BASELINE

# The batch step's start is time zero, and its length determines the downsampling
batch_task = the_first(the_first(batch_step["Nodes"].values())["Tasks"].values())
base_start_time = batch_task[0, "EpochTime"]
try:
    (time_scale, time_unit) = pick_time_scale(
        batch_task[-2, "EpochTime"] - base_start_time
    )
except ValueError:
    (time_scale, time_unit) = (60, "minute")

downsample = max(1, batch_task.shape[0] // 3000)
if downsample > 1:
    print("Downsampling by factor of", downsample)
if "batch" in step_order:
    step_order.remove("batch")
    step_order.insert(0, "batch")

# COLOUR STYLE DEPENDS ON STEPS > 1 OR NOT

# alpha = 1.0 / n_tasks
alpha = 0.5 + 0.5**n_tasks

n_steps = len(step_order)
step_name_map = {}
if n_steps > 1:
    colour_by_step = True
    n_colors = n_steps
    if job_id is not None:
        try:
            step_name_map = dict(
                (n, ((n + " " + v) if len(n) < 4 else n))
                for (n, v) in (
                    line.split(".", 1)[-1].split(None, 1)
                    for line in check_output(
                        ["sacct", "-n", "-j", job_id, "--format", "jobid%20,jobname"],
                        universal_newlines=True,
                    )
                    .strip()
                    .split("\n")
                )
            )
        except Exception as detail:
            pass
else:
    colour_by_step = False
    n_colors = n_tasks
cmap = plt.get_cmap("hsv", n_colors + 1)

# FIGURE LAYOUT

(
    fig,
    (cpu_subplot, mem_subplot, io_rate_subplot, cumulative_io_subplot, *gpu_subplots),
) = plt.subplots(
    (6 if GPU else 4), 1, sharex=True, squeeze=True, figsize=(11.69 - 1, 8.27 - 1)
)
if GPU:
    (gpu_subplot, gpumem_subplot) = gpu_subplots
fig.suptitle(fn.split("/")[-1])
fig.subplots_adjust(hspace=0.15)

# Written bytes not actually negative:
io_rate_subplot.yaxis.set_major_formatter(NotNegFormatter())
io_rate_subplot.axhline(y=0, c="black")

# PLOT SERIES

max_io_rate = max_cumulative_io = 0
io_rate_plots = []
cumulative_io_plots = []
i_colour = 0
legend_data = []
for i_step, step_name in enumerate(step_order):
    step_label = step_name_map.get(step_name, step_name)
    for i_node, node in enumerate(steps[step_name]["Nodes"].values()):
        for task in node.get("Tasks", {}).values():
            if colour_by_step:
                i_colour = i_step
                series_name = step_label
            else:
                i_colour += 1
                series_name = str(task)
            c = to_rgba(cmap(i_colour), alpha)

            if len(task) < 3:
                continue

            # If there are too many timepoints then downsample
            indicies = numpy.arange(0, len(task), downsample)

            # If there are non-distinct time-points then skip them
            useful = numpy.nonzero(numpy.diff(task["EpochTime"][indicies]))
            indicies = numpy.concatenate([indicies[useful], [len(task) - 1]])

            x = (task["EpochTime"][indicies] - base_start_time) / time_scale

            # time points not guaranteed to be evenly spaced.
            dx = numpy.diff(x)
            dx = numpy.concatenate(
                [[numpy.median(dx)], dx]
            )  # pad and avoid div by zero

            cpu = numpy.add.reduceat(task["CPUUtilization"], indicies) / downsample
            # Sometimes spurious negative or very high numbers have appeared, so filter those out
            ok = numpy.logical_and(
                cpu >= 0, cpu <= max_cpu * 100
            )  # Max 128 CPUS per task
            cpu_subplot.plot(
                x[ok], cpu[ok] / 100, color=c, label=series_name, drawstyle="steps"
            )

            if "GPUUtilization" in task.dtype.fields:
                gpu = numpy.maximum.reduceat(task["GPUUtilization"], indicies)
                ok = gpu < 1e9  # nodes without NVML returning very high numbers
                ok[0] = False  # first value is spurious
                gpu_subplot.plot(
                    x[ok], gpu[ok] / 100, color=c, label=series_name, drawstyle="steps"
                )

            rss = numpy.maximum.reduceat(task["RSS"], indicies) / (1024**2)
            mem_subplot.plot(x, rss, color=c, drawstyle="steps")

            if "GPUMemMB" in task.dtype.fields:
                gpumem = numpy.maximum.reduceat(task["GPUMemMB"], indicies) / 1024
                ok = gpumem < 1e9  # nodes without NVML returning very high numbers
                ok[0] = False  # first value is spurious
                gpumem_subplot.plot(x[ok], gpumem[ok], color=c, drawstyle="steps")

            for series_name, line_style, sign in [
                ("ReadMB", "-", 1),
                ("WriteMB", ":", -1),
            ]:
                io_data = numpy.add.reduceat(task[series_name], indicies)
                io_data[io_data < 0] = 0  # Why do negative numbers occur here?
                io_rate = io_data / (dx * time_scale)  # to make rates per-second.
                cumulative_io = numpy.add.accumulate(io_data)
                # Postpone I/O plotting until all I/O data has been seen so as to pick GB vs MB scale.
                max_io_rate = max(max_io_rate, io_rate.max())
                max_cumulative_io = max(max_cumulative_io, cumulative_io.max())
                io_rate_plots.append((line_style, sign, c, x, dx, io_rate))
                cumulative_io_plots.append((line_style, sign, c, x, dx, cumulative_io))
            # print(max_io_rate, max_cumulative_io)

# AXIS LABELS.  I/O PLOTS DONE LATE SO AS TO PICK THEIR Y-AXIS UNITS

cpu_subplot.set_ylabel("CPUs")
mem_subplot.set_ylabel("RSS (GB)")
if GPU:
    gpu_subplot.set_ylabel("GPUs")
    gpumem_subplot.set_ylabel("GPU (GB)")

(io_rate_scale, si_prefix) = pick_mega_scale(max_io_rate)
io_rate_subplot.set_ylabel("I/O ({}B/s)".format(si_prefix))
for line_style, sign, colour, x, dx, y in io_rate_plots:
    b = y != 0
    io_rate_subplot.bar(
        x[b] - dx[b], y[b] * sign / io_rate_scale, dx[b], color=colour, linewidth=0
    )
    # io_rate_subplot.plot(x, y*sign/io_rate_scale, color=colour, drawstyle="steps")

(cumulative_io_scale, si_prefix) = pick_mega_scale(max_cumulative_io)
cumulative_io_subplot.set_ylabel("I/O ({}B)".format(si_prefix))
for line_style, sign, colour, x, dx, y in cumulative_io_plots:
    cumulative_io_subplot.plot(
        x, y / cumulative_io_scale, color=colour, linestyle=line_style
    )

plt.autoscale(axis="x", tight=True)
plt.xlabel(time_unit)

# LEGEND

if colour_by_step:
    plt.subplots_adjust(right=0.85)
    handles = []
    labels = []
    for h, l in zip(*cpu_subplot.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, borderaxespad=0.1, loc="center right", frameon=False)

# OUTPUT
outname = fn.split("/")[-1].split(".")[0] + "_profile.png"
plt.savefig(outname)
print(f"Output saved to '{outname}'.")
