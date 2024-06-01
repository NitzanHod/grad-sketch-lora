import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np

sns.set()


def scatter_rank_vs_memory(model_size=60, project_name=f"gnn-depth/lora"):
    wandb.login()
    api = wandb.Api()
    runs = api.runs(path=project_name, filters={"config.group": f"tracking_time_mem_llama{model_size}"})
    # runs = api.runs(path=project_name, filters={"config.group": f"time_mem_llama{model_size}"})
    pqs = list()
    times_opt = list()
    times_dur = list()
    times_fb = list()
    totals = list()
    ranks = list()
    projs = list()

    # to calc overhead we want to know for each rank in std, what is the equivalent rank in gaussian_std.
    # hence per rank, what is the total memory consumption
    for idx, run in enumerate(runs):
        try:
            if 'opt_step_time' in run.summary and run.config['rank'] > 8:
                t_dur = run.summary['_wandb']['runtime']
                t_opt = run.summary['opt_step_time'].mean
                t_fb = run.history()['fb_step_time'].mean()  # forgot to put it in summary as mean so we fetch history
                pq = run.config['optim_states']['pq']
                total = run.config['optim_states']['total']
                proj = run.config['proj_type']
                rank = run.config['rank']
                totals.append(total)
                times_dur.append(t_dur)
                times_opt.append(t_opt)
                times_fb.append(t_fb)
                ranks.append(rank)
                projs.append(proj)
        except:
            pass
    projs = np.array(projs)
    totals = np.array(totals)
    ranks = np.array(ranks)
    times_dur = np.array(times_dur)
    times_opt = np.array(times_opt)
    times_fb = np.array(times_fb)
    for metric, yname, title_name in [[totals, 'Memory (Total)', 'Memory'],
                                      [times_opt, 'Opt Step Time (s)', 'Opt Step Time'],
                                      [times_dur, 'Run Duration (s)', 'Wall Time'],
                                      [times_fb, 'F+B Time (s)', 'F+B Time'],
                                      ]:

        plt.xlabel('Rank')
        plt.ylabel(yname)
        plt.xticks(ranks)
        plt.title(f'LLaMa-{model_size}m - {title_name} vs. Rank')
        for proj in list(set(projs)):
            plt.scatter(ranks[projs == proj],
                        metric[projs == proj],
                        label=proj)
        plt.legend(loc='upper left')
        plt.show()


def plot_spectrum(rank_histograms):
    plt.figure(figsize=(10, 6))
    for threshold, ranks in rank_histograms.items():
        # Define custom bins for better readability
        bins = 10
        plt.hist(ranks, bins=bins, alpha=0.5, label=f'Threshold {threshold}')

    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Histogram of Ranks per Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    raise ValueError('ASDF Done!')


if __name__ == "__main__":
    for size in [60]:
        scatter_rank_vs_memory(size)
