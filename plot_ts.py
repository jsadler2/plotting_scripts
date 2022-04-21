import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


def seg_time_data(df,
                  spatial_id,
                  spatial_idx_name,
                  time_idx_name,
                  start_time=None,
                  end_time=None):
    df = df[df[spatial_idx_name] == spatial_id]
    df.set_index(time_idx_name, inplace=True)
    df = df.loc[start_time:end_time]
    # df.reset_index(inplace=True)
    return df


def combine_model_data_seg_times(
    model_files, spatial_id, variable, model_ids, start_time=None, end_time=None
):
    df_preds_list = []
    for i, exp_data_file in enumerate(model_files):
        df = pd.read_feather(exp_data_file)
        df_seg = seg_time_data(df, spatial_id, start_time, end_time)
        new_col_name = model_ids[i]
        df_seg.rename(
            columns={get_pred_name(variable): new_col_name}, inplace=True
        )
        df_preds_list.append(df_seg[new_col_name])
    df_preds = pd.concat(df_preds_list, axis=1)
    return df_preds


def get_format_obs(obs_file, seg, variable, start_time, end_time):
    df_obs = (
        xr.open_zarr(obs_file)[get_obs_name(variable)]
        .to_dataframe()
        .reset_index()
    )
    df_obs = seg_time_data(df_obs, seg, start_time, end_time)
    df_obs.rename(
        columns={get_obs_name(variable): "observations"}, inplace=True
    )
    return df_obs[["observations"]]


def get_fmt_sntemp(sntemp_file, variable, seg, start_time=None, end_time=None):
    df_sntemp = pd.read_feather(sntemp_file)
    df_sntemp["spatial_id_nat"] = df_sntemp["spatial_id_nat"].astype(int)

    if variable == "temp_degC":
        sntemp_var = "seg_tave_water"
    else:
        sntemp_var = "seg_outflow"
    df_sntemp_preds = seg_time_data(df_sntemp, seg, start_time, end_time)[
        sntemp_var
    ]
    df_sntemp_preds.rename("uncal prms/sntemp", inplace=True)
    return df_sntemp_preds


def ts_plot(
    out_model_files,
    obs_file,
    seg,
    start_time=None,
    end_time=None,
    sntemp_file=None,
    variable="temp",
    model_labels=None,
    top_perc=None,
    bot_perc=None,
    plot_month=None,
):
    """
    :param out_model_files: [list] paths to .feather output model files
    :param obs_file: [str] paths to .csv observation file
    :param seg: [int] which segment you want to plot
    :param start_time: [str] date you want plot to start at YYYY-MM-DD
    :param end_time: [str] date you want plot to end at YYYY-MM-DD
    :param sntemp_file: [str] sntemp file
    :param variable: [str] variable to plot. either 'flow' or 'temp'
    :param model_labels: [list] labels for models
    """

    if not model_labels:
        model_labels = list(range(len(out_model_files)))
    df_preds = combine_model_data_seg_times(
        out_model_files, seg, variable, model_labels, start_time, end_time
    )
    if sntemp_file:
        sntemp_preds = get_fmt_sntemp(
            sntemp_file, variable, seg, start_time, end_time
        )
        df_preds = df_preds.join(sntemp_preds)
    df_obs = get_format_obs(obs_file, seg, variable, start_time, end_time)
    df = df_preds.join(df_obs)
    if plot_month:
        df = df[df.index.month == plot_month]
    if top_perc or bot_perc:
        if bot_perc:
            df = filter_by_perc(df, bot_perc, bottom=True)
        else:
            df = filter_by_perc(df, top_perc, bottom=False)
    return df


def filter_by_perc(df, perc, bottom=True):
    percentile_val = np.nanpercentile(df["observations"], perc)
    if bottom:
        return df.where(df["observations"] < percentile_val, np.nan)
    else:
        print(percentile_val)
        return df.where(df["observations"] > percentile_val, np.nan)


def plot_scatter_obs_preds(
    out_model_files,
    obs_file,
    seg,
    start_time=None,
    end_time=None,
    sntemp_file=None,
    variable="temp",
    model_labels=None,
    top_perc=None,
    bot_perc=None,
    plot_month=None,
    palette=None,
):
    """
    scatter plot of predictions against observations
    :param out_model_files: [list] paths to .feather output model files
    :param obs_file: [str] paths to .csv observation file
    :param seg: [int] which segment you want to plot
    :param start_time: [str] date you want plot to start at YYYY-MM-DD
    :param end_time: [str] date you want plot to end at YYYY-MM-DD
    :param sntemp_file: [str] sntemp file
    :param variable: [str] variable to plot. either 'flow' or 'temp'
    :param model_labels: [list] labels for models
    """
    df = ts_plot(
        out_model_files,
        obs_file,
        seg,
        start_time,
        end_time,
        sntemp_file,
        variable,
        model_labels,
        top_perc,
        bot_perc,
        plot_month,
    )
    df = df.set_index("observations")
    nfiles = len(out_model_files)
    if sntemp_file:
        nfiles += 1
    ax = sns.scatterplot(
        data=df,
        alpha=0.5,
        style=None,
        markers=["o"] * nfiles,
        s=10,
        palette=palette,
    )
    max_val = df.max().max() * 1.05
    min_val = df.min().min() * 0.95
    ax.plot([min_val, max_val], [min_val, max_val])
    ax.set_title(seg)
    return ax


def plot_one_seg_times(
    out_model_files,
    obs_file,
    seg,
    start_time=None,
    end_time=None,
    out_file=None,
    sntemp_file=None,
    variable="temp",
    figsize=None,
    model_labels=None,
    interactive=False,
    scatter_obs=False,
    top_perc=None,
    bot_perc=None,
):
    """
    plot a time series of one segment for a given set of dates
    :param out_model_files: [list] paths to .feather output model files
    :param obs_file: [str] paths to .csv observation file
    :param seg: [int] which segment you want to plot
    :param start_time: [str] date you want plot to start at YYYY-MM-DD
    :param end_time: [str] date you want plot to end at YYYY-MM-DD
    :param sntemp_file: [str] sntemp file
    :param variable: [str] variable to plot. either 'flow' or 'temp'
    :param figsize: [tuple] figure size
    :param model_labels: [list] labels for models
    :param interactive: [bool] if True, plot will be a hvplot. Only good if
    in a notebook setting
    :param scatter_obs: [bool] if True, observations will be plotted as scatter
    points. If false, they will be plotted as a line
    :return:
    """
    df = ts_plot(
        out_model_files,
        obs_file,
        seg,
        start_time,
        end_time,
        sntemp_file,
        variable,
        model_labels,
        top_perc,
        bot_perc,
    )
    if interactive:
        pd.options.plotting.backend = "holoviews"
        ax = df.plot()
    else:
        ax = df.iloc[:, :-1].plot(alpha=0.5, figsize=figsize)
        if scatter_obs:
            ax.scatter(
                x=df.index,
                y=df["observations"],
                s=2,
                c="k",
                zorder=10,
                alpha=0.6,
                label="obs",
            )
        else:
            df["observations"].plot(c="k", ax=ax, legend=True)
        ax = plt.gca()
        ax.set_title(f"seg id: {seg}")
        ax.set_ylabel(f'{variable} {get_yunits(variable, "rmse")}')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels)
        plt.tight_layout()
        plt.grid()
        if out_file:
            plt.savefig(
                out_file,
                dpi=300,
                bbox_extra_artists=(lgd,),
                bbox_inches="tight",
            )
    return ax
