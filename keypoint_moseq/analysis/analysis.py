import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
from tqdm import tqdm

import uuid
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from collections import defaultdict, OrderedDict
from copy import deepcopy
from cytoolz import sliding_window
from keypoint_moseq.util import filter_angle
import seaborn as sns

def compute_moseq_df(results_dict, *, use_bodyparts, smooth_heading=True, **kwargs):
    session_name = []
    centroid = []
    velocity = []
    estimated_coordinates = []
    heading = []
    syllables = []
    syllables_reindexed = []
    frame_index = []
    s_uuid = []
    for k, v in results_dict.items():
        centroid.append(v['centroid'])
        velocity.append(np.concatenate(([0], np.sqrt(np.square(np.diff(v['centroid'], axis = 0)).sum(axis=1)) *30)))
        n_frame= v['centroid'].shape[0]
        session_name.append([str(k)] *n_frame)
        s_uuid.append([str(uuid.uuid4())]*n_frame)
        frame_index.append(np.arange(n_frame))
        estimated_coordinates.append(v['estimated_coordinates'])
        if smooth_heading:
            heading.append(filter_angle(v['heading']))
        else:
            heading.append(v['heading'])
        syllables.append(v['syllables'])
        syllables_reindexed.append(v['syllables_reindexed'])
    
    # build data frame
    # coor_col = []
    # for part in use_bodyparts:
    #     coor_col.append(part+'_x')
    #     coor_col.append(part+'_y')
    # estimated_coordinates =np.concatenate(estimated_coordinates)
    # n, x, y = estimated_coordinates.shape
    # coor_df = pd.DataFrame(np.reshape(estimated_coordinates, (n, x*y)), columns=coor_col)
    
    moseq_df = pd.DataFrame(np.concatenate(centroid), columns=['centroid_x', 'centroid_y'])
    # moseq_df = pd.concat([moseq_df, coor_df], axis = 1)
    moseq_df['heading'] = np.concatenate(heading)
    moseq_df['velocity_px_s'] = np.concatenate(velocity)
    moseq_df['syllable'] = np.concatenate(syllables)
    moseq_df['syllables_reindexed'] = np.concatenate(syllables_reindexed)
    moseq_df['frame_index'] = np.concatenate(frame_index)
    moseq_df['session_name'] = np.concatenate(session_name)
    moseq_df['uuid'] = np.concatenate(s_uuid)

    # TODO: velecity, body span etc?

    # compute syllable onset
    change = np.diff(moseq_df['syllable']) !=0
    indices = np.where(change)[0]
    indices += 1
    indices = np.concatenate(([0], indices))

    onset = np.full(moseq_df.shape[0], False)
    onset[indices] = True
    moseq_df['onset'] = onset
    return moseq_df

def compute_stats_df(moseq_df, threshold=0.005, groupby=['group','uuid', 'session_name'], fps=30, syll_key='syllables_reindexed', normalize=True, **kwargs):
    raw_usage = (moseq_df.groupby('syllable').count()['frame_index']/moseq_df.shape[0]).reset_index().rename(columns={'frame_index':'counts'})
    syll_include = raw_usage[raw_usage['counts'] > threshold]['syllable']
    filtered_df = moseq_df[moseq_df['syllable'].isin(syll_include)].copy()

    usages = (filtered_df.groupby(groupby)[syll_key]
                  .value_counts(normalize=normalize)
                  .unstack(fill_value=0)
                  .reset_index()
                  .melt(id_vars = groupby)
                  .set_index(groupby + [syll_key]))
    usages.columns = ['usage']

    # TODO: hard-coded heading for now, could add other scalars
    features = filtered_df.groupby(groupby + [syll_key])[['heading', 'velocity_px_s']].agg(['mean', 'std', 'min', 'max'])

    features.columns = ['_'.join(col).strip() for col in features.columns.values]

    # get durations
    trials = filtered_df['onset'].cumsum()
    trials.name = 'trials'
    durations = filtered_df.groupby(groupby + [syll_key] + [trials])['onset'].count()
    # average duration in seconds
    durations = durations.groupby(groupby + [syll_key]).mean() / fps
    durations.name = 'duration'
    durations.fillna(0)
    
    stats_df = usages.join(durations).join(features).reset_index()
    stats_df = stats_df.rename(columns={'syllables_reindexed':'syllable'})
    return stats_df


## fingerprint
def robust_min(v):
    return v.quantile(0.01)


def robust_max(v):
    return v.quantile(0.99)


def _apply_to_col(df, fn, **kwargs):
    return df.apply(fn, axis=0, **kwargs)


def create_fingerprint_dataframe(scalar_df, mean_df, stat_type='mean', n_bins=100, 
                                 groupby_list=['group','uuid'], range_type='robust', 
                                 scalars=['heading', 'velocity_px_s']):
    # deep copy the dfs
    scalar_df = scalar_df.copy()
    mean_df = mean_df.copy()
    # rescale velocity to cm/s
    vel_cols = [c for c in scalars if 'velocity' in c]
    vel_cols_stats = [f'{c}_{stat_type}' for c in scalars if 'velocity' in c]
    
    if len(vel_cols) > 0:
        scalar_df[vel_cols] *= 30
        mean_df[vel_cols_stats] *=30
    
    # pivot mean_df to be groupby x syllable
    syll_summary = mean_df.pivot_table(index=groupby_list, values='usage', columns='syllable')
    syll_summary.columns = pd.MultiIndex.from_arrays([['MoSeq'] * syll_summary.shape[1], syll_summary.columns])
    min_p = syll_summary.min().min()
    max_p = syll_summary.max().max()
    
    ranges = scalar_df.reset_index(drop=True)[scalars].agg(['min', 'max', robust_min, robust_max])
    # add syllable ranges to this df
    ranges['MoSeq'] = [min_p, max_p, min_p, max_p]
    range_idx = ['min', 'max'] if range_type == 'full' else ['robust_min', 'robust_max']

    def bin_scalars(data: pd.Series, n_bins=50, range_type='full'):
        _range = ranges.loc[range_idx, data.name]
        bins = np.linspace(_range.iloc[0], _range.iloc[1], n_bins)

        binned_data = data.value_counts(normalize=True, sort=False, bins=bins)
        binned_data = binned_data.sort_index().reset_index(drop=True)
        binned_data.index.name = 'bin'
        return binned_data

    # use total number of syllables 
    if n_bins is None:
        n_bins = syll_summary.shape[1] + 1 # num of bins (default to match the total number of syllables)

    binned_scalars = scalar_df.groupby(groupby_list)[scalars].apply(_apply_to_col, fn=bin_scalars, range_type=range_type, n_bins=n_bins)

    scalar_fingerprint = binned_scalars.pivot_table(index=groupby_list, columns='bin', values=binned_scalars.columns)

    fingerprints = scalar_fingerprint.join(syll_summary, how='outer')

    return fingerprints, ranges.loc[range_idx]


def plotting_fingerprint(summary,range_dict, preprocessor_type='minmax', num_level=1, level_names=['Group'], vmin=None, vmax=None,
                         figsize=(10,15), plot_columns=['heading','velocity_px_s', 'MoSeq'],
                         col_names=[('Heading','a.u.'),('velocity','px/s'), ('MoSeq','Syllable ID')]):
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    assert preprocessor_type in ['minmax','standard','none']
    if preprocessor_type=='minmax': preprocessor = MinMaxScaler()
    elif preprocessor_type=='standard': preprocessor = StandardScaler()
    else: preprocessor = None
    
    # ensure number of groups is not over the number of available levels
    if num_level > len(summary.index.names):
        raise Exception('Too many levels to unpack. num_level should be less than', len(summary.index.names))

    name_map = dict(zip(plot_columns, col_names))
    
    levels = []
    level_plot = []
    level_ticks = []
    for i in range(num_level):
        level = summary.index.get_level_values(i)
        level_label = LabelEncoder().fit_transform(level)
        find_mid = (np.diff(np.r_[0,np.argwhere(np.diff(level_label)).ravel(),len(level_label)])/2).astype('int32')
        # store level value
        levels.append(level)
        level_plot.append(level_label)
        level_ticks.append(np.r_[0,np.argwhere(np.diff(level_label)).ravel()] + find_mid)
    
    # col_num = number of grouping/level + column in summary
    col_num = num_level + len(plot_columns)

    # https://matplotlib.org/stable/tutorials/intermediate/gridspec.html
    fig = plt.figure(1, figsize=figsize, facecolor='white')

    gs = GridSpec(2, col_num, wspace=0.1, hspace=0.1,
              width_ratios=[1]*num_level+[8]*(col_num-num_level), height_ratios = [10,0.1], figure=fig)

    # plot the level(s)
    for i in range(num_level):
        temp_ax = fig.add_subplot(gs[0,i])
        temp_ax.set_title(level_names[i], fontsize=20)
        temp_ax.imshow(level_plot[i][:,np.newaxis], aspect = 'auto', cmap = 'Set3')
        plt.yticks(level_ticks[i], levels[i][level_ticks[i]], fontsize=20)
        
        temp_ax.get_xaxis().set_ticks([])
    
    # compile data to plot while recording vmin and vmax in the data
    plot_dict = {}
    # initialize vmin and vmax
    temp_vmin = np.Inf
    temp_vmax = -np.Inf
    
    for col in plot_columns:
        data = summary[col].to_numpy()
        # process data with preprocessor
        if preprocessor is not None:
            data = preprocessor.fit_transform(data.T).T

        if np.min(data) < temp_vmin:
            temp_vmin = np.min(data)
        if np.max(data) > temp_vmax:
            temp_vmax = np.max(data)

        plot_dict[col] = data
    
    if vmin is None:
        vmin = temp_vmin
    if vmax is None:
        vmax = temp_vmax

    # plot the data
    for i, col in enumerate(plot_columns):
        name = name_map[col]
        temp_ax = fig.add_subplot(gs[0, i + num_level])
        temp_ax.set_title(name[0], fontsize=20)
        data = plot_dict[col]
        
        # top to bottom is 0-20 for y axis
        if col == 'MoSeq':
            extent = [summary[col].columns[0], summary[col].columns[-1], len(summary) - 1, 0]
        else:
            extent = [range_dict[col].iloc[0], range_dict[col].iloc[1], len(summary) - 1, 0]

        pc = temp_ax.imshow(data, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, extent=extent)
        temp_ax.set_xlabel(name[1], fontsize=10)
        temp_ax.set_xticks(np.linspace(np.ceil(extent[0]), np.floor(extent[1]), 6).astype(int))
        # https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
        temp_ax.set_yticks([])
        temp_ax.axis = 'tight'
    
    # plot colorbar
    cb = fig.add_subplot(gs[1,-1])
    plt.colorbar(pc, cax=cb, orientation='horizontal')

    # specify labels for feature scaling
    if preprocessor:
        cb.set_xlabel('Min Max')
    else:
        cb.set_xlabel('Percentage Usage')

## usage plot stuff
def sort_syllables_by_stat_difference(complete_df, ctrl_group, exp_group, max_sylls=None, stat='usage'):
    if max_sylls is not None:
        complete_df = complete_df[complete_df.syllable < max_sylls]

    # Prepare DataFrame
    mutation_df = complete_df.groupby(['group', 'syllable']).mean()

    # Get groups to measure mutation by
    control_df = mutation_df.loc[ctrl_group]
    exp_df = mutation_df.loc[exp_group]

    # compute mean difference at each syll usage and reorder based on difference
    ordering = (exp_df[stat] - control_df[stat]).sort_values(ascending=False).index

    return list(ordering)


def sort_syllables_by_stat(complete_df, stat='usage', max_sylls=None):
    if max_sylls is not None:
        complete_df = complete_df[complete_df.syllable < max_sylls]

    tmp = complete_df.groupby('syllable').mean().sort_values(by=stat, ascending=False).index

    # Get sorted ordering
    ordering = list(tmp)

    # Get order mapping
    relabel_mapping = {o: i for i, o in enumerate(ordering)}

    return ordering, relabel_mapping
    
def _validate_and_order_syll_stats_params(complete_df, stat='usage', ordering='stat', max_sylls=40, groups=None, ctrl_group=None, exp_group=None, colors=None, figsize=(10, 5)):

    if not isinstance(figsize, (tuple, list)):
        print('Invalid figsize. Input a integer-tuple or list of len(figsize) = 2. Setting figsize to (10, 5)')
        figsize = (10, 5)

    unique_groups = complete_df['group'].unique()

    if groups is None or len(groups) == 0:
        groups = unique_groups
    elif isinstance(groups, str):
        groups = [groups]

    if isinstance(groups, (list, tuple, np.ndarray)):
        diff = set(groups) - set(unique_groups)
        if len(diff) > 0:
            groups = unique_groups

    if stat.lower() not in complete_df.columns:
        raise ValueError(f'Invalid stat entered: {stat}. Must be a column in the supplied dataframe.')

    if ordering is None:
        ordering = np.arange(max_sylls)
    elif ordering == "stat":
        ordering, _ = sort_syllables_by_stat(complete_df, stat=stat, max_sylls=max_sylls)
    elif ordering == "diff":
        if ctrl_group is None or exp_group is None or not np.all(np.isin([ctrl_group, exp_group], groups)):
            raise ValueError(f'Attempting to sort by {stat} differences, but {ctrl_group} or {exp_group} not in {groups}.')
        ordering = sort_syllables_by_stat_difference(complete_df, ctrl_group, exp_group,
                                                     max_sylls=max_sylls, stat=stat)
    if colors is None:
        colors = []
    if len(colors) == 0 or len(colors) != len(groups):
        colors = sns.color_palette(n_colors=len(groups))

    return ordering, groups, colors, figsize

def plot_syll_stats_with_sem(scalar_df, syll_info=None, sig_sylls=None, stat='usage', ordering='stat', max_sylls=40,
                             groups=None, ctrl_group=None, exp_group=None, colors=None, join=False, figsize=(8, 4)):

    xlabel = f'Syllables sorted by {stat}'
    if ordering == 'diff':
        xlabel += ' difference'
    ordering, groups, colors, figsize = _validate_and_order_syll_stats_params(scalar_df,
                                                                              stat=stat,
                                                                              ordering=ordering,
                                                                              max_sylls=max_sylls,
                                                                              groups=groups,
                                                                              ctrl_group=ctrl_group,
                                                                              exp_group=exp_group,
                                                                              colors=colors,
                                                                              figsize=figsize)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot each group's stat data separately, computes groupwise SEM, and orders data based on the stat/ordering parameters
    hue = 'group' if groups is not None else None
    ax = sns.pointplot(data=scalar_df, x='syllable', y=stat, hue=hue, order=ordering,
                       join=join, dodge=True, errorbar=('ci', 68), ax=ax, hue_order=groups,
                       palette=colors)

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # add syllable labels if they exist
    if syll_info is not None:
        mean_xlabels = []
        for o in (ordering):
            mean_xlabels.append(f'{syll_info[o]["label"]} - {o}')

        plt.xticks(range(max_sylls), mean_xlabels, rotation=90)

    # if a list of significant syllables is given, mark the syllables above the x-axis
    if sig_sylls is not None:
        markings = []
        for s in sig_sylls:
            markings.append(ordering.index(s))
        plt.scatter(markings, [-.005] * len(markings), color='r', marker='*')

        # manually define a new patch
        patch = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                              markersize=9, label='Significant Syllable')
        handles.append(patch)

    # add legend and axis labels
    legend = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1, 1))
    plt.xlabel(xlabel, fontsize=12)
    sns.despine()

    return fig, legend

# transition matrix
def get_transitions(label_sequence):
    arr = deepcopy(label_sequence)

    # get syllable transition locations
    locs = np.where(arr[1:] != arr[:-1])[0] + 1
    transitions = arr[locs]

    return transitions, locs

def get_syllable_statistics(data, fill_value=-5, max_syllable=100, count='usage'):
    usages = defaultdict(int)
    durations = defaultdict(list)

    use_usage = count == 'usage'
    if not use_usage and count != 'frames':
        print('Inputted count is incorrect or not supported. Use "usage" or "frames".')
        print('Calculating statistics by syllable usage')
        use_usage = True

    for s in range(max_syllable):
        usages[s] = 0
        durations[s] = []

    if isinstance(data, list) or (isinstance(data, np.ndarray) and data.dtype == np.object):

        for v in data:
            seq_array, locs = get_transitions(v)
            to_rem = np.where(np.logical_or(seq_array > max_syllable,
                                            seq_array == fill_value))

            seq_array = np.delete(seq_array, to_rem)
            locs = np.delete(locs, to_rem)
            durs = np.diff(np.insert(locs, len(locs), len(v)))

            for s, d in zip(seq_array, durs):
                if use_usage:
                    usages[s] = usages[s] + 1
                else:
                    usages[s] = usages[s] + d
                durations[s].append(d)

    else:#elif type(data) is np.ndarray and data.dtype == 'int16':

        seq_array, locs = get_transitions(data)
        to_rem = np.where(seq_array > max_syllable)[0]

        seq_array = np.delete(seq_array, to_rem)
        locs = np.delete(locs, to_rem)
        durs = np.diff(np.insert(locs, len(locs), len(data)))

        for s, d in zip(seq_array, durs):
            if use_usage:
                usages[s] = usages[s] + 1
            else:
                usages[s] = usages[s] + d
            durations[s].append(d)


    usages = OrderedDict(sorted(usages.items())[:max_syllable])
    durations = OrderedDict(sorted(durations.items())[:max_syllable])
    return usages, durations

def n_gram_transition_matrix(labels, n=2, max_label=99):
    trans_mat = np.zeros((max_label, ) * n, dtype='float')
    for loc in sliding_window(n, labels):
        if any(l >= max_label for l in loc):
            continue
        trans_mat[loc] += 1
    return trans_mat
    

def normalize_transition_matrix(init_matrix, normalize):
    if normalize is None or normalize not in ('bigram', 'rows', 'columns'):
        return init_matrix
    if normalize == 'bigram':
        init_matrix /= init_matrix.sum()
    elif normalize == 'rows':
        init_matrix /= init_matrix.sum(axis=1, keepdims=True)
    elif normalize == 'columns':
        init_matrix /= init_matrix.sum(axis=0, keepdims=True)

    return init_matrix

def get_transition_matrix(labels, max_syllable=100, normalize='bigram',
                          smoothing=0.0, combine=False, disable_output=False) -> list:
    if not isinstance(labels[0], (list, np.ndarray, pd.Series)):
        labels = [labels]

    # Compute a singular transition matrix
    if combine:
        init_matrix = []

        for v in tqdm(labels, disable=disable_output, desc='Computing bigram transition probabilities'):
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = n_gram_transition_matrix(transitions, n=2, max_label=max_syllable)
            init_matrix.append(trans_mat)

        init_matrix = np.sum(init_matrix, axis=0) + smoothing
        all_mats = normalize_transition_matrix(init_matrix, normalize)
    else:
        # Compute a transition matrix for each session label list
        all_mats = []
        for v in tqdm(labels, disable=disable_output, desc='Computing bigram transition probabilities'):
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = n_gram_transition_matrix(transitions, n=2, max_label=max_syllable) + smoothing

            # Normalize matrix
            init_matrix = normalize_transition_matrix(trans_mat, normalize)
            all_mats.append(init_matrix)

    return all_mats

def get_group_trans_mats(labels, label_group, group, max_sylls, normalize='bigram'):
    trans_mats = []
    usages = []

    # Computing transition matrices for each given group
    for plt_group in group:
        # Get sessions to include in trans_mat
        use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
        trans_mats.append(get_transition_matrix(use_labels,
                                                normalize=normalize,
                                                combine=True,
                                                max_syllable=max_sylls))

        # Getting usage information for node scaling
        usages.append(get_syllable_statistics(use_labels, max_syllable=max_sylls)[0])
    return trans_mats, usages