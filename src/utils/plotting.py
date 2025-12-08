import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import binned_statistic, norm

# pyplot config
pyplot_cfg = {
    # "axes.formatter.limits": [-2, 2.99],
    "axes.titlesize": 14,
    "font.size": 14,
    "font.family": "serif",
    "legend.fontsize": 12,
    "figure.subplot.top": 0.95,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.left": 0.15,
    "figure.subplot.right": 0.96,
    "savefig.pad_inches": 0.1,
    "savefig.dpi": 300,
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{amsmath}" r"\usepackage[bitstream-charter]{mathdesign}"
    ),
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(pyplot_cfg)

BLACK = "#323232"


def add_reweighting_histogram(ax, exp, sim, sim_weights, num_bins=32, discrete=False):

    exp = exp[np.isfinite(exp)]
    sim_mask = np.isfinite(sim) & np.isfinite(sim_weights)
    sim = sim[sim_mask]
    sim_weights = sim_weights[sim_mask]

    lo = min(exp.min(), sim.min())
    hi = max(exp.max(), sim.max())
    bins = (
        np.arange(lo, hi + 1, discrete) - discrete / 2
        if discrete
        else np.linspace(lo, hi, num_bins)
    )

    # histogram values
    y_exp = np.histogram(exp, bins=bins)[0]
    y_sim = np.histogram(sim, bins=bins)[0]

    y_rew = np.histogram(sim, bins=bins, weights=sim_weights)[0]

    # histogram errors
    err_exp = np.sqrt(y_exp)
    err_sim = np.sqrt(y_sim)
    sum_w2 = binned_statistic(sim, sim_weights**2, "sum", bins=bins)[0]
    err_rew = np.sqrt(sum_w2)

    ys = (y_exp, y_sim, y_rew)
    errs = (err_exp, err_sim, err_rew)
    labels = ("Data", "Sim", "Reweighted Sim")
    colors = ("C0", "C2", "C3")
    for y, err, label, color in zip(ys, errs, labels, colors):
        scale = 1
        dup_last = lambda a: np.append(a, a[-1])
        ax.step(
            bins,
            dup_last(y) * scale,
            where="post",
            color=color,
            label=label,
            lw=1.0,
        )
        ax.fill_between(
            bins,
            dup_last(y - err) * scale,
            dup_last(y + err) * scale,
            alpha=0.2,
            step="post",
            facecolor=color,
        )
    ax.set_ylabel("Counts")


def add_histogram(
    ax,
    vals,
    weights=None,
    bins=32,
    discrete=False,
    label=None,
    color=None,
    density=False,
):

    if isinstance(bins, int):
        lo, hi = vals.min(), vals.max()
        bins = (
            np.arange(lo, hi + 1, discrete) - discrete / 2
            if discrete
            else np.linspace(lo, hi, bins)
        )

    weights = np.ones(len(vals)) if weights is None else weights

    # histogram values
    y = np.histogram(vals, bins=bins, weights=weights)[0]

    # histogram errors
    sum_w2 = binned_statistic(vals, weights**2, "sum", bins=bins)[0]
    err = np.sqrt(sum_w2)

    dup_last = lambda a: np.append(a, a[-1])
    scale = 1 / y.sum() if density else 1
    ax.step(
        bins,
        dup_last(y) * scale,
        where="post",
        color=color,
        label=label,
        lw=1.0,
    )
    ax.fill_between(
        bins,
        dup_last(y - err) * scale,
        dup_last(y + err) * scale,
        alpha=0.2,
        step="post",
        color=color,
    )
    ax.set_ylabel("Density" if density else "Counts")

    return y, err


def plot_reweighting(
    exp,
    sim,
    weights_list,
    variance_list,
    names_list,
    xlabel,
    figsize,
    num_bins=45,
    discrete=False,
    title=None,
    logx=False,
    logy=False,
    qlims=(0.005, 0.995),
    xlims=None,
    quantiles_from_sim=False,
    name_exp="Data",
    name_sim="Sim",
    denom_idx=0,
    ratio_lims=(0.85, 1.15),
    density=True,
    add_chi2=True,
    add_legend=True,
    exp_weights=None,
):

    # make figure with ratio axis
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 1.5], hspace=0.0)
    main_ax = plt.subplot(grid[0])
    ratio_ax = plt.subplot(grid[1])

    # # remove entries with nans
    # exp = exp[np.isfinite(exp)]
    # sim_mask = np.isfinite(sim)
    # for ws in weights_list:
    #     sim_mask = sim_mask & np.isfinite(ws)
    # sim = sim[sim_mask]
    # weights_list = [ws[sim_mask] for ws in weights_list]

    # lo, hi = np.quantile(sim if quantiles_from_sim else exp, qlims)
    lo, hi = xlims or np.quantile(np.hstack([sim, exp]), qlims)
    bins = (
        np.arange(lo, hi + discrete + 1, discrete) - discrete / 2
        if discrete
        else np.linspace(lo, hi, num_bins)
    )

    # counts and errors
    if exp_weights is None:
        y_exp = np.histogram(exp, bins=bins)[0]
        err_exp = np.sqrt(y_exp)
    else:
        y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
        sum_w2s = binned_statistic(exp, exp_weights**2, "sum", bins=bins)[0]
        err_exp = np.sqrt(sum_w2s)
    y_sim = np.histogram(sim, bins=bins)[0]
    err_sim = np.sqrt(y_sim)

    # weighted counts and errors
    y_rews, err_rews = [], []
    for ws, vs in zip(weights_list, variance_list):
        if ws.ndim > 1:  # set of weights from bayesian sample

            y_rews.append(np.histogram(sim, bins=bins, weights=ws.mean(0))[0])
            sum_w2s = binned_statistic(sim, (ws**2).mean(0), "sum", bins=bins)[0]
            err = np.sqrt(sum_w2s)

        else:

            if vs is None:  # assume zero variance if none given
                vs = np.zeros_like(ws)

            mom1 = np.exp(np.log(ws) + vs / 2)
            y_rews.append(np.histogram(sim, bins=bins, weights=mom1)[0])

            mom2 = np.exp(2 * (np.log(ws) + vs))
            sum_w2s = binned_statistic(sim, mom2, "sum", bins=bins)[0]
            err = np.sqrt(sum_w2s)

        err_rews.append(err)

    ys = (y_exp, y_sim, *y_rews)
    errs = (err_exp, err_sim, *err_rews)
    labels_rew = names_list  # [f"{n}[{name_sim}]" for n in names_list]
    labels = (name_exp, name_sim, *names_list)

    # colors = (BLACK, "C0", "C2", "C3", "C4", 'C5')
    # colors = [BLACK, "#1D6A9E", "#ea6702", "#009826"] + [f"C{i}" for i in range(3,8)]
    colors = [
        BLACK,
        "#1D6A9E",
        "#7E3B91",
        "#009826",
        "C1",
        "C3",
        "#13b2b7",
    ]  # + [f"C{i}" for i in [1] + list(range(3, 8))]
    denom = ys[denom_idx]
    dup_last = lambda a: np.append(a, a[-1])
    legend_objs, legend_labels = [], []
    for y, err, label, color in zip(ys, errs, labels, colors):

        if add_chi2 and (label in labels_rew):
            if density:
                pull = (y / y.sum() - y_exp / y_exp.sum()) / np.sqrt(
                    (err / y.sum()) ** 2 + (err_exp / y_exp.sum()) ** 2
                )
            else:
                pull = (y - y_exp) / np.sqrt(err**2 + err_exp**2)

            nonempty = (y != 0) & (y_exp != 0)  # exclude empty bins
            chi2 = (pull[nonempty] ** 2).sum() / num_bins
            label += f" ({chi2:.2f})"

        legend_labels.append(label)

        scale = (
            1
            if not density
            else 1 / y_sim.sum() if (label in labels_rew) else 1 / y.sum()
        )

        (line_obj,) = main_ax.step(
            bins,
            dup_last(y) * scale,
            where="post",
            color=color,
            label=label,
            lw=1.0,
        )
        fill_obj = main_ax.fill_between(
            bins,
            dup_last(y - err) * scale,
            dup_last(y + err) * scale,
            alpha=0.2,
            step="post",
            facecolor=color,
        )
        # ratio_scale = denom.sum() / y.sum() if density else 1
        ratio_scale = (
            1
            if not density
            else denom.sum() / y_sim.sum() if (label in labels_rew) else denom.sum() / y.sum()
        )
        ratio_ax.step(
            bins,
            dup_last(y / denom) * ratio_scale,
            where="post",
            color=color,
            lw=1.0,
        )
        ratio_ax.fill_between(
            bins,
            dup_last((y - err) / denom) * ratio_scale,
            dup_last((y + err) / denom) * ratio_scale,
            alpha=0.2,
            step="post",
            facecolor=color,
        )

        legend_objs.append((line_obj, fill_obj))

    # scales
    if logx:
        main_ax.semilogx()
        ratio_ax.semilogx()
    if logy:
        main_ax.semilogy()

    # limits
    # main_ax.set_ylim(
    #     main_ax.get_ylim()[0],
    #     main_ax.get_ylim()[1] ** 1.15 if logy else 1.3 * main_ax.get_ylim()[1],
    # )
    ratio_ax.set_ylim(*ratio_lims)
    main_ax.set_xlim(bins[0], bins[-1])
    ratio_ax.set_xlim(bins[0], bins[-1])

    # labels
    main_ax.set_ylabel("Density" if density else "Count")
    ratio_ax.set_ylabel("Ratio")
    ratio_ax.set_xlabel(xlabel)

    if add_legend:
        main_ax.legend(
            legend_objs,
            legend_labels,
            frameon=False,
            handlelength=1.4,
            # ncols=3,
            columnspacing=0.5,
            # loc="upper center",
        )

    if title is not None:
        main_ax.set_title(title)

    # ticks
    main_ax.set_xticklabels([])
    # main_ax.tick_params("x", bottom=True)
    # main_ax.tick_params("x", direction="in")
    # ratio_ax.tick_params("x", top=True)
    # ratio_ax.tick_params("x", direction="in")

    return fig, (main_ax, ratio_ax)


def plot_reweighting_pulls(
    exp,
    sim,
    weights_list,
    names_list,
    figsize,
    num_bins_obs=5000,
    num_bins_pull=20,
    title=None,
    exp_weights=None,
):

    # make figure with ratio axis
    fig, ax = plt.subplots(figsize=figsize)
    lo, hi = np.quantile(exp, (0, 1))
    bins_obs = np.linspace(lo, hi, num_bins_obs)

    # histogram values
    if exp_weights is None:
        y_exp = np.histogram(exp, bins=bins_obs)[0]
    else:
        y_exp = np.histogram(exp, bins=bins_obs, weights=exp_weights)[0]
    y_rews = []
    for ws in weights_list:
        if ws.ndim > 1:  # set of weights from bayesian sample
            y_rews.append(np.histogram(sim, bins=bins_obs, weights=ws.mean(0))[0])
        else:  # single
            y_rews.append(np.histogram(sim, bins=bins_obs, weights=ws)[0])

    # histogram errors
    if exp_weights is None:
        err_exp = np.sqrt(y_exp)
    else:
        sum_w2s = binned_statistic(exp, exp_weights**2, "sum", bins=bins_obs)[0]
        err_exp = np.sqrt(y_exp)

    err_rews = []
    for ws in weights_list:
        if ws.ndim > 1:  # set of weights from bayesian sample
            sum_w2s = binned_statistic(
                sim, (ws**2).sum(0) / len(ws), "sum", bins=bins_obs
            )[0]
            err = np.sqrt(sum_w2s)
        else:  # single
            sum_w2s = binned_statistic(sim, ws**2, "sum", bins=bins_obs)[0]
            err = np.sqrt(sum_w2s)
        err_rews.append(err)

    # calculate pulls
    pulls = []
    for y, sigma in zip(y_rews, err_rews):
        nonempty = (y != 0) & (y_exp != 0)  # exclude empty bins
        pull = (y - y_exp) / np.sqrt(sigma**2 + err_exp**2)
        pulls.append(pull[nonempty])

    bins_pull = np.linspace(-4, 4, num_bins_pull)
    for name, pull in zip(names_list, pulls):
        plt.hist(pull, bins=bins_pull, alpha=0.3, density=True, label=name)

    xs = np.linspace(bins_pull[[0, 1]].mean(), bins_pull[[-1, -2]].mean(), 100)
    plt.plot(xs, norm.pdf(xs), color=BLACK)

    # labels
    ax.set_ylabel("Density")
    ax.set_xlabel(r"$t_\mathrm{stat}$")
    ax.legend(frameon=False, handlelength=1.4)
    if title is not None:
        ax.set_title(title)

    # clean
    fig.tight_layout(pad=0.2)

    return fig, ax


def plot_reweighting_calibration(
    exp,
    sim,
    weights_list,
    names_list,
    figsize,
    num_bins_obs=5000,
    num_bins_pull=20,
    title=None,
    exp_weights=None,
):

    # make figure with ratio axis
    fig, ax = plt.subplots(figsize=figsize)
    lo, hi = np.quantile(exp, (0, 1))
    bins_obs = np.linspace(lo, hi, num_bins_obs)

    # histogram values
    if exp_weights is None:
        y_exp = np.histogram(exp, bins=bins_obs)[0]
    else:
        y_exp = np.histogram(exp, bins=bins_obs, weights=exp_weights)[0]
    y_rews = []
    for ws in weights_list:
        if ws.ndim > 1:  # set of weights from bayesian sample
            y_rews.append(np.histogram(sim, bins=bins_obs, weights=ws.mean(0))[0])
        else:  # single
            y_rews.append(np.histogram(sim, bins=bins_obs, weights=ws)[0])

    # histogram errors
    if exp_weights is None:
        err_exp = np.sqrt(y_exp)
    else:
        sum_w2s = binned_statistic(exp, exp_weights**2, "sum", bins=bins_obs)[0]
        err_exp = np.sqrt(y_exp)
    err_rews = []
    for ws in weights_list:
        if ws.ndim > 1:  # set of weights from bayesian sample
            sum_w2s = binned_statistic(
                sim, (ws**2).sum(0) / len(ws), "sum", bins=bins_obs
            )[0]
            err = np.sqrt(sum_w2s)
        else:  # single
            sum_w2s = binned_statistic(sim, ws**2, "sum", bins=bins_obs)[0]
            err = np.sqrt(sum_w2s)
        err_rews.append(err)

    # calculate pulls
    pulls = []
    for y, sigma in zip(y_rews, err_rews):
        nonempty = (y != 0) & (y_exp != 0)  # exclude empty bins
        pull = (y - y_exp) / np.sqrt(sigma**2 + err_exp**2)
        pulls.append(pull[nonempty])

    ax.plot([0, 1], [0, 1], color=BLACK)
    for name, pull in zip(names_list, pulls):
        quantiles = np.linspace(1e-8, 1, num_bins_pull + 1)
        bins = norm.ppf(quantiles)
        bin_widths = bins[1:] - bins[:-1]
        pdf = np.histogram(pull, bins=bins, density=True)[0] * bin_widths
        cdf = np.cumsum(pdf)
        ax.plot(quantiles[1:], cdf, label=name)

    ax.set_aspect(0.95)
    ax.margins(0.02)

    # labels
    ax.set_ylabel("Empirical CDF")
    ax.set_xlabel("Gaussian CDF")
    ax.legend(frameon=False, handlelength=1.4)
    if title is not None:
        ax.set_title(title)

    # clean
    fig.tight_layout(pad=0.2)

    return fig, ax


def plot_syst_calibration(
    pred_logweights,
    pred_logweight_vars,
    ref_logweights,
    pred_weight_label,
    ref_weight_label,
    pull_label,
    quantile_obs=None,
    num_quantiles=1,
    num_bins_pull=128,
    num_points_scatter=500,
    ref_lims=(-2.5, 2.5),
    diff_lims=(-0.5, 0.5),
    pull_lims=(-3.5, 3.5),
):

    # calculate pull
    pull = (pred_logweights - ref_logweights) / np.sqrt(pred_logweight_vars)

    # hist bins in Gaussian quantiles
    quantile_edges = np.linspace(1e-8, 1, num_bins_pull + 1)
    bins = norm.ppf(quantile_edges)
    bin_widths = bins[1:] - bins[:-1]
    pdf = np.histogram(pull, bins=bins, density=True)[0] * bin_widths
    cdf = np.insert(np.cumsum(pdf), 0, 0)

    # open figure
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3))

    # determine quantiles
    quantile_obs = ref_logweights if quantile_obs is None else quantile_obs
    quantiles = np.quantile(quantile_obs, np.linspace(0, 1, num_quantiles + 1))

    # reference lines
    ax[0].plot(ref_lims, [0, 0], color=BLACK)
    bins_pull = np.linspace(-5, 5, num_bins_pull)
    xs = np.linspace(bins_pull[[0, 1]].mean(), bins_pull[[-1, -2]].mean(), 100)
    ax[1].plot(xs, norm.pdf(xs), color=BLACK)
    ax[2].plot([0, 1], [0, 1], color=BLACK)

    # labels
    ax[0].set_xlabel(rf"$\log {ref_weight_label}$")
    # ax[0].set_ylabel(rf"$\log \displaystyle\frac{{{pred_weight_label}}}{{{ref_weight_label}}}$")
    ax[0].set_ylabel(rf"$\log {pred_weight_label} - \log {ref_weight_label} $")
    ax[1].set_ylabel("Density")
    ax[1].set_xlabel(f"${pull_label}$")
    ax[2].set_ylabel("Empirical CDF")
    ax[2].set_xlabel("Gaussian CDF")

    colors = ["#1D6A9E", "#009826", "C1", "C3"]
    bin_widths = bins[1:] - bins[:-1]
    for i in range(num_quantiles):

        # pred
        pull_mask = np.digitize(quantile_obs, quantiles) == i + 1
        N = int(num_points_scatter / num_quantiles)
        ax[0].errorbar(
            ref_logweights[pull_mask][:N],
            pred_logweights[pull_mask][:N] - ref_logweights[pull_mask][:N],
            yerr=np.sqrt(pred_logweight_vars)[pull_mask][:N],
            fmt="o",
            ms=1,
            elinewidth=0.1,
            alpha=0.6,
            color=colors[i],
        )

        # pull hist
        ax[1].hist(
            pull[pull_mask],
            bins=bins_pull,
            density=True,
            histtype="step" if num_quantiles > 1 else "stepfilled",
            alpha=1 if num_quantiles > 1 else 0.4,
            color=colors[i],
        )

        # calib
        pdf = np.histogram(pull[pull_mask], bins=bins, density=True)[0] * bin_widths
        cdf = np.cumsum(pdf)
        ax[2].plot(quantile_edges[1:], cdf)

    # axis limits
    ax[0].set_xlim(*ref_lims)
    ax[0].set_ylim(*diff_lims)
    ax[1].set_xlim(*pull_lims)

    # clean
    fig.tight_layout(pad=0.2)

    return fig, ax


def plot_reweighting_ensemble(
    exp,
    sim,
    weights_list,
    variance_list,
    names_list,
    xlabel,
    figsize,
    num_bins=45,
    discrete=False,
    title=None,
    logx=False,
    logy=False,
    qlims=(0.005, 0.995),
    xlims=None,
    quantiles_from_sim=False,
    name_exp="Data",
    name_sim="Sim",
    denom_idx=0,
    ratio_lims=(0.85, 1.15),
    density=True,
    add_chi2=True,
    exp_weights=None,
):

    # make figure with ratio axis
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 1.5], hspace=0.0)
    main_ax = plt.subplot(grid[0])
    ratio_ax = plt.subplot(grid[1])

    # # remove entries with nans
    # exp = exp[np.isfinite(exp)]
    # sim_mask = np.isfinite(sim)
    # for ws in weights_list:
    #     sim_mask = sim_mask & np.isfinite(ws)
    # sim = sim[sim_mask]
    # weights_list = [ws[sim_mask] for ws in weights_list]

    # lo, hi = np.quantile(sim if quantiles_from_sim else exp, qlims)
    lo, hi = xlims or np.quantile(np.hstack([sim, exp]), qlims)
    bins = (
        np.arange(lo, hi + discrete + 1, discrete) - discrete / 2
        if discrete
        else np.linspace(lo, hi, num_bins)
    )

    # counts and errors
    if exp_weights is None:
        y_exp = np.histogram(exp, bins=bins)[0]
        err_exp = (y_exp - np.sqrt(y_exp), y_exp + np.sqrt(y_exp))
    else:
        y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
        sum_w2s = binned_statistic(exp, exp_weights**2, "sum", bins=bins)[0]
        err_exp = np.sqrt(sum_w2s)
    y_sim = np.histogram(sim, bins=bins)[0]
    err_sim = (y_sim - np.sqrt(y_sim), y_sim + np.sqrt(y_sim))

    # weighted counts and errors
    y_rews, err_rews = [], []
    for ws, vs in zip(weights_list, variance_list):
        assert ws.ndim > 1  # set of weights from bayesian sample

        all_y = np.apply_along_axis(
            lambda w: np.histogram(sim, bins=bins, weights=w)[0], 1, ws
        )
        all_sum_w2s = np.apply_along_axis(
            lambda w: binned_statistic(sim, w**2, "sum", bins=bins)[0], 1, ws
        )

        y_rew = np.quantile(all_y, 0.5, axis=0)
        err = (
            np.quantile(all_y - np.sqrt(all_sum_w2s), 0.0, axis=0),
            np.quantile(all_y + np.sqrt(all_sum_w2s), 1.0, axis=0)
        )

        y_rews.append(y_rew)
        err_rews.append(err)

    ys = (y_exp, y_sim, *y_rews)
    errs = (err_exp, err_sim, *err_rews)
    # errs = [err_exp, err_sim] + err_rews
    labels_rew = names_list  # [f"{n}[{name_sim}]" for n in names_list]
    labels = (name_exp, name_sim, *names_list)

    # colors = (BLACK, "C0", "C2", "C3", "C4", 'C5')
    # colors = [BLACK, "#1D6A9E", "#ea6702", "#009826"] + [f"C{i}" for i in range(3,8)]
    colors = [
        BLACK,
        "#1D6A9E",
        "#7E3B91",
        "#009826",
        "C1",
        "C3",
        "#13b2b7",
    ]  # + [f"C{i}" for i in [1] + list(range(3, 8))]
    denom = ys[denom_idx]
    dup_last = lambda a: np.append(a, a[-1])
    legend_objs, legend_labels = [], []
    for y, err, label, color in zip(ys, errs, labels, colors):

        # if add_chi2 and (label in labels_rew):
        #     if density:
        #         pull = (y / y.sum() - y_exp / y_exp.sum()) / np.sqrt(
        #             (err / y.sum()) ** 2 + (err_exp / y_exp.sum()) ** 2
        #         )
        #     else:
        #         pull = (y - y_exp) / np.sqrt(err**2 + err_exp**2)

        #     nonempty = (y != 0) & (y_exp != 0)  # exclude empty bins
        #     chi2 = (pull[nonempty] ** 2).sum() / num_bins
        #     label += f" ({chi2:.2f})"

        legend_labels.append(label)

        scale = (
            1
            if not density
            else 1 / y_sim.sum() if (label in labels_rew) else 1 / y.sum()
        )

        (line_obj,) = main_ax.step(
            bins,
            dup_last(y) * scale,
            where="post",
            color=color,
            label=label,
            lw=1.0,
        )
        fill_obj = main_ax.fill_between(
            bins,
            dup_last(err[0]) * scale,
            dup_last(err[1]) * scale,
            alpha=0.2,
            step="post",
            facecolor=color,
        )
        ratio_scale = (
            1
            if not density
            else denom.sum() / y_sim.sum() if (label in labels_rew) else denom.sum() / y.sum()
        )
        ratio_ax.step(
            bins,
            dup_last(y / denom) * ratio_scale,
            where="post",
            color=color,
            lw=1.0,
        )
        ratio_ax.fill_between(
            bins,
            dup_last(err[0] / denom) * ratio_scale,
            dup_last(err[1] / denom) * ratio_scale,
            alpha=0.2,
            step="post",
            facecolor=color,
        )

        legend_objs.append((line_obj, fill_obj))

    # scales
    if logx:
        main_ax.semilogx()
        ratio_ax.semilogx()
    if logy:
        main_ax.semilogy()

    # limits
    # main_ax.set_ylim(
    #     main_ax.get_ylim()[0],
    #     main_ax.get_ylim()[1] ** 1.15 if logy else 1.3 * main_ax.get_ylim()[1],
    # )
    ratio_ax.set_ylim(*ratio_lims)
    main_ax.set_xlim(bins[0], bins[-1])
    ratio_ax.set_xlim(bins[0], bins[-1])

    # labels
    main_ax.set_ylabel("Density" if density else "Count")
    ratio_ax.set_ylabel("Ratio")
    ratio_ax.set_xlabel(xlabel)

    main_ax.legend(
        legend_objs,
        legend_labels,
        frameon=False,
        handlelength=1.4,
        # ncols=3,
        columnspacing=0.5,
        # loc="upper center",
    )

    if title is not None:
        main_ax.set_title(title)

    # ticks
    main_ax.set_xticklabels([])
    # main_ax.tick_params("x", bottom=True)
    # main_ax.tick_params("x", direction="in")
    # ratio_ax.tick_params("x", top=True)
    # ratio_ax.tick_params("x", direction="in")

    return fig, (main_ax, ratio_ax)
