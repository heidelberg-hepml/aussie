import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import binned_statistic

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
        r"\usepackage{amsmath}"
        r"\usepackage[bitstream-charter]{mathdesign}"
        r"\DeclareSymbolFont{usualmathcal}{OMS}{cmsy}{m}{n}"
        r"\DeclareSymbolFontAlphabet{\mathcal}{usualmathcal}"
    ),
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(pyplot_cfg)

BLACK = "#323232"


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
    show_sim=True,
    denom_idx=0,
    ratio_lims=(0.85, 1.15),
    density=True,
    add_chi2=True,
    add_legend=True,
    exp_weights=None,
    colors=None,
):

    # make figure with ratio axis
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 1.5], hspace=0.0)
    main_ax = plt.subplot(grid[0])
    ratio_ax = plt.subplot(grid[1])

    ratio_ax.axhline(1.0, color="gray", lw=1.0)

    # remove entries with nans
    exp_mask = np.isfinite(exp)
    sim_mask = np.isfinite(sim)
    if exp_weights is not None:
        exp_weights = exp_weights[exp_mask]
    exp = exp[exp_mask]
    sim = sim[sim_mask]
    weights_list = [ws[..., sim_mask] for ws in weights_list]

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
        if ws.ndim > 1:  # set of weights from ensemble

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
    colors = colors or [
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
    for i, (y, err, label, color) in enumerate(zip(ys, errs, labels, colors)):

        if (i == 1) and not show_sim:
            continue

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

        # (line_obj,) = main_ax.step(
        #     bins,
        #     dup_last(y) * scale,
        #     where="post",
        #     color=color,
        #     label=label,
        #     lw=1.0,
        # )
        # fill_obj = main_ax.fill_between(
        #     bins,
        #     dup_last(y - err) * scale,
        #     dup_last(y + err) * scale,
        #     alpha=0.2,
        #     step="post",
        #     facecolor=color,
        # )
        # # ratio_scale = denom.sum() / y.sum() if density else 1
        # ratio_scale = (
        #     1
        #     if not density
        #     else (
        #         denom.sum() / y_sim.sum()
        #         if (label in labels_rew)
        #         else denom.sum() / y.sum()
        #     )
        # )
        # ratio_ax.step(
        #     bins,
        #     dup_last(y / denom) * ratio_scale,
        #     where="post",
        #     color=color,
        #     lw=1.0,
        # )
        # ratio_ax.fill_between(
        #     bins,
        #     dup_last((y - err) / denom) * ratio_scale,
        #     dup_last((y + err) / denom) * ratio_scale,
        #     alpha=0.2,
        #     step="post",
        #     facecolor=color,
        # )

        # legend_objs.append((line_obj, fill_obj))

        # if i > 0:
        #     (line_obj,) = main_ax.step(
        #         bins,
        #         dup_last(y) * scale,
        #         where="post",
        #         color=color,
        #         label=label,
        #         lw=1.0,
        #         zorder=i,
        #     )
        #     fill_obj = main_ax.fill_between(
        #         bins,
        #         dup_last(y - err) * scale,
        #         dup_last(y + err) * scale,
        #         alpha=0.2,
        #         step="post",
        #         facecolor=color,
        #         zorder=i,
        #     )
        #     ratio_scale = denom.sum() / y.sum() if density else 1
        #     ratio_ax.step(
        #         bins,
        #         dup_last(y / denom) * ratio_scale,
        #         where="post",
        #         color=color,
        #         zorder=i,
        #         lw=1.0,
        #     )
        #     ratio_ax.fill_between(
        #         bins,
        #         dup_last((y - err) / denom) * ratio_scale,
        #         dup_last((y + err) / denom) * ratio_scale,
        #         alpha=0.2,
        #         step="post",
        #         facecolor=color,
        #         zorder=i,
        #     )

        #     legend_objs.append((line_obj, fill_obj))

        # else:
        #     bin_centers = 0.5 * (bins[1:] + bins[:-1])
        #     point_obj = main_ax.errorbar(
        #         bin_centers,
        #         y * scale,
        #         yerr=err * scale,
        #         color=color,
        #         label=label,
        #         fmt="o",
        #         zorder=len(ys) + 1,
        #         ms=1.5,
        #         elinewidth=1,
        #     )
        #     ratio_scale = denom.sum() / y.sum() if density else 1
        #     ratio_ax.errorbar(
        #         bin_centers,
        #         y / denom * ratio_scale,
        #         yerr=err / denom * ratio_scale,
        #         color=color,
        #         fmt="o",
        #         zorder=len(ys) + 1,
        #         ms=1.5,
        #         elinewidth=1,
        #     )

        #     legend_objs.append(point_obj)

        fill_obj = main_ax.fill_between(
            bins,
            dup_last(y - err) * scale,
            dup_last(y + err) * scale,
            alpha=0.15,
            step="post",
            facecolor=color,
            zorder=i,
        )
        ratio_scale = denom.sum() / y.sum() if density else 1
        ratio_ax.fill_between(
            bins,
            dup_last((y - err) / denom) * ratio_scale,
            dup_last((y + err) / denom) * ratio_scale,
            alpha=0.15,
            step="post",
            facecolor=color,
            zorder=i,
        )
        if i > 0:
            (line_obj,) = main_ax.step(
                bins,
                dup_last(y) * scale,
                where="post",
                color=color,
                label=label,
                lw=0.75,
                zorder=i,
            )
            ratio_ax.step(
                bins,
                dup_last(y / denom) * ratio_scale,
                where="post",
                color=color,
                zorder=i,
                lw=0.75,
            )

            legend_objs.append((line_obj, fill_obj))

        else:
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            point_obj = main_ax.scatter(
                bin_centers,
                y * scale,
                # yerr=err * scale,
                color=color,
                label=label,
                # fmt="o",
                zorder=len(ys) + 1,
                s=1.5,
                # elinewidth=1,
            )
            ratio_ax.scatter(
                bin_centers,
                y / denom * ratio_scale,
                # yerr=err / denom * ratio_scale,
                color=color,
                # fmt="o",
                zorder=len(ys) + 1,
                s=1.5,
                # elinewidth=1,
            )

            legend_objs.append(point_obj)

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
    main_ax.set_ylabel("Prob." if density else "Count")
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
    show_sim=True,
    denom_idx=0,
    ratio_lims=(0.85, 1.15),
    density=True,
    add_chi2=True,
    exp_weights=None,
    add_legend=False,
    colors=None,
):

    # make figure with ratio axis
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 1.5], hspace=0.0)
    main_ax = plt.subplot(grid[0])
    ratio_ax = plt.subplot(grid[1])

    ratio_ax.axhline(1.0, color="gray", lw=1.0)

    # # remove entries with nans
    # exp_mask = np.isfinite(exp)
    # sim_mask = np.isfinite(sim)
    # if exp_weights is not None:
    #     exp_weights = exp_weights[exp_mask]
    # exp = exp[exp_mask]
    # sim = sim[sim_mask]
    # weights_list = [ws[..., sim_mask] for ws in weights_list]

    # lo, hi = np.quantile(sim if quantiles_from_sim else exp, qlims)
    lo, hi = xlims or np.quantile(np.hstack([sim, exp]), qlims)
    bins = (
        np.arange(lo, hi + discrete + 1, discrete) - discrete / 2
        if discrete
        else np.linspace(lo, hi, num_bins)
    )

    # counts and errors
    if exp_weights is None:
        exp_weights = np.ones_like(exp)

    y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
    sum_w2s = binned_statistic(exp, exp_weights**2, "sum", bins=bins)[0]
    if density:
        norm_exp = y_exp.sum()
        y_exp = y_exp / norm_exp
        sum_w2s = sum_w2s / (norm_exp**2)
    err_exp = (
        y_exp - np.sqrt(sum_w2s),
        y_exp + np.sqrt(sum_w2s),
    )  # only if not using plt.errorbar for data
    # err_exp = np.sqrt(sum_w2s)

    y_sim = np.histogram(sim, bins=bins)[0]
    err_sim = (y_sim - np.sqrt(y_sim), y_sim + np.sqrt(y_sim))
    if density:
        norm_sim = y_sim.sum()
        err_sim = (
            (y_sim - np.sqrt(y_sim)) / norm_sim,
            (y_sim + np.sqrt(y_sim)) / norm_sim,
        )
        y_sim = y_sim / norm_sim

    # weighted counts and errors
    y_rews, err_rews = [], []
    for ws, vs in zip(weights_list, variance_list):
        assert ws.ndim > 1  # set of weights from ensemble

        all_y = np.apply_along_axis(
            lambda w: np.histogram(sim, bins=bins, weights=w)[0], 1, ws
        )
        all_sum_w2s = np.apply_along_axis(
            lambda w: binned_statistic(sim, w**2, "sum", bins=bins)[0], 1, ws
        )
        if density:
            all_y = all_y / norm_sim
            all_sum_w2s = all_sum_w2s / (norm_sim**2)

        y_rew = np.quantile(all_y, 0.5, axis=0)
        err = (
            np.quantile(all_y - np.sqrt(all_sum_w2s), 0.0, axis=0),
            np.quantile(all_y + np.sqrt(all_sum_w2s), 1.0, axis=0),
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
    colors = colors or [
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
    for i, (y, err, label, color) in enumerate(zip(ys, errs, labels, colors)):

        if (i == 1) and not show_sim:
            continue

        legend_labels.append(label)

        scale = (
            1
            # if (not density) or (label in labels_rew)
            # else 1 / y.sum()
            # else 1 / y_sim.sum() if (label in labels_rew) else 1 / y.sum()
        )

        # legend_objs.append((line_obj, fill_obj))

        # if i > 0:
        #     (line_obj,) = main_ax.step(
        #         bins,
        #         dup_last(y) * scale,
        #         where="post",
        #         color=color,
        #         label=label,
        #         lw=1.0,
        #         zorder=i,
        #     )
        #     fill_obj = main_ax.fill_between(
        #         bins,
        #         # dup_last(y - err) * scale,
        #         # dup_last(y + err) * scale,
        #         dup_last(err[0]) * scale,
        #         dup_last(err[1]) * scale,
        #         alpha=0.2,
        #         step="post",
        #         facecolor=color,
        #         zorder=i,
        #     )
        #     ratio_scale = denom.sum() / y.sum() if density else 1
        #     ratio_ax.step(
        #         bins,
        #         dup_last(y / denom) * ratio_scale,
        #         where="post",
        #         color=color,
        #         zorder=i,
        #         lw=1.0,
        #     )
        #     ratio_ax.fill_between(
        #         bins,
        #         dup_last(err[0] / denom) * ratio_scale,
        #         dup_last(err[1] / denom) * ratio_scale,
        #         alpha=0.2,
        #         step="post",
        #         facecolor=color,
        #         zorder=i,
        #     )

        #     legend_objs.append((line_obj, fill_obj))

        # else:
        #     bin_centers = 0.5 * (bins[1:] + bins[:-1])
        #     point_obj = main_ax.errorbar(
        #         bin_centers,
        #         y * scale,
        #         yerr=err * scale,
        #         color=color,
        #         label=label,
        #         fmt="o",
        #         zorder=len(ys) + 1,
        #         ms=1.5,
        #         elinewidth=1,
        #     )
        #     ratio_scale = 1  # denom.sum() / y.sum() if density else 1
        #     ratio_ax.errorbar(
        #         bin_centers,
        #         y / denom * ratio_scale,
        #         yerr=err / denom * ratio_scale,
        #         color=color,
        #         fmt="o",
        #         zorder=len(ys) + 1,
        #         ms=1.5,
        #         elinewidth=1,
        #     )

        #     legend_objs.append(point_obj)

        fill_obj = main_ax.fill_between(
            bins,
            dup_last(err[0]) * scale,
            dup_last(err[1]) * scale,
            alpha=0.1 if i == 0 else 0.2,
            step="post",
            facecolor=color,
            zorder=i,
        )
        ratio_scale = denom.sum() / y.sum() if density else 1
        ratio_ax.fill_between(
            bins,
            dup_last(err[0] / denom) * ratio_scale,
            dup_last(err[1] / denom) * ratio_scale,
            alpha=0.1 if i == 0 else 0.2,
            step="post",
            facecolor=color,
            zorder=i,
        )
        if i > 0:
            (line_obj,) = main_ax.step(
                bins,
                dup_last(y) * scale,
                where="post",
                color=color,
                label=label,
                lw=1.0,
                zorder=i,
            )
            ratio_ax.step(
                bins,
                dup_last(y / denom) * ratio_scale,
                where="post",
                color=color,
                zorder=i,
                lw=1.0,
            )

            legend_objs.append((line_obj, fill_obj))

        else:
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            point_obj = main_ax.scatter(
                bin_centers,
                y * scale,
                # yerr=err * scale,
                color=color,
                label=label,
                # fmt="o",
                zorder=len(ys) + 1,
                s=1.5,
                # elinewidth=1,
            )
            ratio_ax.scatter(
                bin_centers,
                y / denom * ratio_scale,
                # yerr=err / denom * ratio_scale,
                color=color,
                # fmt="o",
                zorder=len(ys) + 1,
                s=1.5,
                # elinewidth=1,
            )

            legend_objs.append(point_obj)

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


def plot_reweighting_multi_ratio(
    exp,
    sim,
    weights_list,
    variance_list,
    names_list,
    ratio_idx,
    ratio_names,
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
    show_sim=True,
    denom_idx=0,
    ratio_lims=(0.85, 1.15),
    density=True,
    add_chi2=True,
    exp_weights=None,
    add_legend=False,
    colors=None,
    legend_loc=None,
    ypad=1,
):

    num_ratios = np.max(ratio_idx) + 1

    # make figure with ratio axis
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = gridspec.GridSpec(
        1 + num_ratios,
        1,
        figure=fig,
        height_ratios=[4] + [1.5] * num_ratios,
        hspace=0.0,
    )
    main_ax = plt.subplot(grid[0])
    ratio_axes = [plt.subplot(grid[i + 1]) for i in range(num_ratios)]

    for i in range(num_ratios):
        ratio_axes[i].axhline(1.0, color="gray", lw=1.0)

    # remove entries with nans
    exp_mask = np.isfinite(exp)
    sim_mask = np.isfinite(sim)
    if exp_weights is not None:
        exp_weights = exp_weights[exp_mask]
    exp = exp[exp_mask]
    sim = sim[sim_mask]
    weights_list = [ws[..., sim_mask] for ws in weights_list]

    # lo, hi = np.quantile(sim if quantiles_from_sim else exp, qlims)
    lo, hi = xlims or np.quantile(np.hstack([sim, exp]), qlims)
    bins = (
        np.arange(lo, hi + discrete + 1, discrete) - discrete / 2
        if discrete
        else np.linspace(lo, hi, num_bins)
    )

    # counts and errors
    if exp_weights is None:
        exp_weights = np.ones_like(exp)

    y_exp = np.histogram(exp, bins=bins, weights=exp_weights)[0]
    sum_w2s = binned_statistic(exp, exp_weights**2, "sum", bins=bins)[0]
    if density:
        norm_exp = y_exp.sum()
        y_exp = y_exp / norm_exp
        sum_w2s = sum_w2s / (norm_exp**2)
    err_exp = (y_exp - np.sqrt(sum_w2s), y_exp + np.sqrt(sum_w2s))

    y_sim = np.histogram(sim, bins=bins)[0]
    err_sim = (y_sim - np.sqrt(y_sim), y_sim + np.sqrt(y_sim))
    if density:
        norm_sim = y_sim.sum()
        err_sim = (
            (y_sim - np.sqrt(y_sim)) / norm_sim,
            (y_sim + np.sqrt(y_sim)) / norm_sim,
        )
        y_sim = y_sim / norm_sim

    # weighted counts and errors
    y_rews, err_rews = [], []
    for ws, vs in zip(weights_list, variance_list):
        assert ws.ndim > 1  # set of weights from ensemble

        all_y = np.apply_along_axis(
            lambda w: np.histogram(sim, bins=bins, weights=w)[0], 1, ws
        )
        all_sum_w2s = np.apply_along_axis(
            lambda w: binned_statistic(sim, w**2, "sum", bins=bins)[0], 1, ws
        )
        if density:
            all_y = all_y / norm_sim
            all_sum_w2s = all_sum_w2s / (norm_sim**2)

        central = np.quantile(all_y, 0.5, axis=0)
        var_across = np.var(all_y, axis=0)
        mean_within = np.mean(all_sum_w2s, axis=0)
        total_var = var_across + mean_within
        z = 1.0  # 1.0 <--> 68%, 1.96 <--> 95%
        err = (central - z * np.sqrt(total_var), central + z * np.sqrt(total_var))
        y_rew = central

        y_rews.append(y_rew)
        err_rews.append(err)

    ys = (y_exp, y_sim, *y_rews)
    errs = (err_exp, err_sim, *err_rews)
    labels_rew = names_list  # [f"{n}[{name_sim}]" for n in names_list]
    labels = (name_exp, name_sim, *names_list)

    colors = colors or [
        "#323232",
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

    ratio_idcs = [np.arange(len(ratio_idx))] * 2 + [[i] for i in ratio_idx]

    for i, (y, err, label, color) in enumerate(zip(ys, errs, labels, colors)):

        if (i == 1) and not show_sim:
            continue

        legend_labels.append(label)

        scale = (
            1
            # if (not density) or (label in labels_rew)
            # else 1 / y.sum()
            # else 1 / y_sim.sum() if (label in labels_rew) else 1 / y.sum()
        )

        if i == 1:
            fill_obj = None
        else:
            fill_obj = main_ax.fill_between(
                bins,
                dup_last(err[0]) * scale,
                dup_last(err[1]) * scale,
                alpha=0.1 if i == 0 else 0.2,
                step="post",
                facecolor=color,
                zorder=i,
            )
        ratio_scale = denom.sum() / y.sum() if density else 1
        for ir in range(num_ratios):
            if ir in ratio_idcs[i]:
                if i == 1:
                    continue
                ratio_axes[ir].fill_between(
                    bins,
                    dup_last(err[0] / denom) * ratio_scale,
                    dup_last(err[1] / denom) * ratio_scale,
                    alpha=0.1 if i == 0 else 0.2,
                    step="post",
                    facecolor=color,
                    zorder=i,
                )
        if i > 0:
            (line_obj,) = main_ax.step(
                bins,
                dup_last(y) * scale,
                where="post",
                color=color,
                label=label,
                lw=0.5 if i == 1 else 1.0,
                ls="--" if i == 1 else "-",
                zorder=i,
            )
            for ir in range(num_ratios):
                if ir in ratio_idcs[i]:
                    ratio_axes[ir].step(
                        bins,
                        dup_last(y / denom) * ratio_scale,
                        where="post",
                        color=color,
                        zorder=i,
                        lw=0.5 if i == 1 else 1.0,
                        ls="--" if i == 1 else "-",
                    )

            legend_objs.append(
                (fill_obj, line_obj) if fill_obj is not None else line_obj
            )

        else:
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            point_obj = main_ax.scatter(
                bin_centers,
                y * scale,
                color=color,
                label=label,
                zorder=len(ys) + 1,
                s=1.5,
            )
            for ir in range(num_ratios):
                if ir in ratio_idcs[i]:
                    ratio_axes[ir].scatter(
                        bin_centers,
                        y / denom * ratio_scale,
                        color=color,
                        zorder=len(ys) + 1,
                        s=1.5,
                    )

            legend_objs.append((fill_obj, point_obj))

    # scales
    if logx:
        main_ax.semilogx()
        for ir in range(num_ratios):
            ratio_axes[ir].semilogx()
    if logy:
        main_ax.semilogy()

    # limits
    main_ax.set_ylim(
        main_ax.get_ylim()[0],
        main_ax.get_ylim()[1] ** ypad if logy else ypad * main_ax.get_ylim()[1],
    )
    main_ax.set_xlim(bins[0], bins[-1])
    main_ax.set_ylabel("Density" if density else "Count")

    for ir in range(num_ratios):
        ratio_axes[ir].set_ylim(*ratio_lims)
        ratio_axes[ir].set_xlim(bins[0], bins[-1])
        ratio_axes[ir].set_ylabel(
            rf"$\frac{{\mathrm{{{ratio_names[ir]}}}}}{{\mathrm{{Data}}}}$"
        )
        if ir < num_ratios - 1:
            ratio_axes[ir].set_xticklabels([])

    ratio_axes[-1].set_xlabel(xlabel)

    if add_legend:
        main_ax.legend(
            legend_objs,
            legend_labels,
            frameon=False,
            handlelength=1.4,
            columnspacing=0.5,
            fontsize=10,
            scatteryoffsets=[0.5],
            loc=legend_loc,
        )

    if title is not None:
        main_ax.set_title(title)

    # ticks
    main_ax.set_xticklabels([])

    plt.subplots_adjust(bottom=0.15)

    return fig, (main_ax, ratio_axes)
