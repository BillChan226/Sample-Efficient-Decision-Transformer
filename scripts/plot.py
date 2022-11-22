import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot(args):

    # env_d4rl_name = 'halfcheetah-medium-v2'
    # log_dir = 'dt_runs/'
    # x_key = "num_updates"
    # y_key = "eval_d4rl_score"
    # y_smoothing_win = 5
    # plot_avg = False
    # save_fig = False

    env_d4rl_name = args.env_d4rl_name
    log_dir = args.log_dir
    x_key = args.x_key
    y_key = args.y_key
    y_smoothing_win = args.smoothing_window
    plot_avg = args.plot_avg
    save_fig = args.save_fig

    if plot_avg:
        save_fig_path = env_d4rl_name + "_avg.png"
    else:
        save_fig_path = env_d4rl_name + ".png"

    all_files = glob.glob(log_dir + f'/dt_{env_d4rl_name}*.csv')
    all_files_2 = glob.glob('dt_runs/' + f'/dt_{env_d4rl_name}*.csv')

    ax = plt.gca()
    #ax.set_title("Walker 2d")

    if plot_avg:
        name_list = []
        df_list = []

        name_list2 = []
        df_list2 = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            df_list.append(frame)
        
        for filename in all_files_2:
            frame2 = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame2.shape)
            frame2['y_smooth'] = frame2[y_key].rolling(window=y_smoothing_win).mean()
            df_list2.append(frame2)

        df_concat = pd.concat(df_list)
        df_concat2 = pd.concat(df_list2)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()
        df_concat_groupby2 = df_concat2.groupby(df_concat2.index)
        data_avg2 = df_concat_groupby2.mean()
        print('Mean:', data_avg['y_smooth'])
        data_var = df_concat_groupby.var()
        data_var2 = df_concat_groupby2.var()
        print('Var:', data_var['y_smooth'])

        font = font1 = {'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 20,
                    }
        def million_formatter(x, pos):
            return '%.1fM' % (x * 1e-6)

        formatter = FuncFormatter(million_formatter)
        ax.set_facecolor((0.95, 0.95, 0.95))
        #ax.xaxis.set_major_formatter(formatter)
        ax.grid(color='white', linestyle='-', linewidth=2, alpha=0.6)
        #data_avg.plot(x=x_key, y='y_smooth', ax=ax)
        #print("data_avg",data_avg)
        # data_var['y_smooth'] = data_var['y_smooth']/18
        # ax.plot(data_avg['num_updates'], data_avg['y_smooth'], color='red', linewidth=2)
        
        
        # data_var2['y_smooth'] = data_var2['y_smooth']/18
        # ax.plot(data_avg2['num_updates'], data_avg2['y_smooth'], color='blue', linewidth=2)
        # ax.fill_between(data_avg['num_updates'], data_avg['y_smooth']+data_var['y_smooth'], data_avg['y_smooth']-data_var['y_smooth'], facecolor='red', alpha=0.3)
        # ax.fill_between(data_avg2['num_updates'], data_avg2['y_smooth']+data_var2['y_smooth'], data_avg2['y_smooth']-data_var2['y_smooth'], facecolor='blue', alpha=0.3)
        data_var['eval_avg_reward'] = data_var['eval_avg_reward']/700
        ax.plot(data_avg['num_updates'], data_avg['eval_avg_reward'], color='blue', linewidth=2)
        data_var2['eval_avg_reward'] = data_var2['eval_avg_reward']/700
        ax.plot(data_avg2['num_updates'], data_avg2['eval_avg_reward'], color='red', linewidth=2)
        ax.fill_between(data_avg['num_updates'], data_avg['eval_avg_reward']+data_var['eval_avg_reward'], data_avg['eval_avg_reward']-data_var['eval_avg_reward'], facecolor='blue', alpha=0.3)
        ax.fill_between(data_avg2['num_updates'], data_avg2['eval_avg_reward']+data_var2['eval_avg_reward'], data_avg2['eval_avg_reward']-data_var2['eval_avg_reward'], facecolor='red', alpha=0.3)
        

        #ax.set_xlabel(x_key, fontdict=font)
        ax.set_xlabel("Ant", fontdict=font)
        #ax.set_ylabel(y_key, fontdict=font)
        ax.legend(['DT', 'SE-DT'], loc='lower right')

        if save_fig:
            plt.savefig(save_fig_path)

        plt.show()

    else:
        name_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            frame.plot(x=x_key, y='y_smooth', ax=ax)
            name_list.append(filename.split('/')[-1])

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend(name_list, loc='lower right')

        if save_fig:
            plt.savefig(save_fig_path)

        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_d4rl_name', type=str, default='ant-medium-v2')
    #parser.add_argument('--env_d4rl_name', type=str, default='walker2d-medium-v2')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--x_key', type=str, default='num_updates')
    parser.add_argument('--y_key', type=str, default='eval_d4rl_score')
    parser.add_argument('--smoothing_window', type=int, default=1)
    parser.add_argument("--plot_avg", action="store_true", default=False,
                    help="plot avg of all logs else plot separately")
    parser.add_argument("--save_fig", action="store_true", default=False,
                    help="save figure if true")

    args = parser.parse_args()

    plot(args)
