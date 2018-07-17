import numpy as np
import matplotlib.pyplot as plt


def read_data():
    ret_list = []
    ret_min = 1e10
    in_file = open("log/rl_log","r")
    for i, line in enumerate(in_file.readlines()):
        if i % 10 == 8:
            ret_n = line.split()
            ret_n = float(ret_n[1])
            ret_min = min(ret_n, ret_min)
            if len(ret_list) == 0:
                ret_list.append(ret_n)
            else:
                ret_list.append(ret_list[-1] * 0.99 + 0.01 * ret_n)
    return ret_list, ret_min


def run():
    params = {
        'axes.labelsize': 20,
        'font.size': 20,
        'legend.fontsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.titlesize': 20,
        'axes.spines.left'   : False,
        'axes.spines.bottom' : True,
        'axes.spines.top'    : False,
        'axes.spines.right'  : False,
        'figure.autolayout': True,
        'axes.grid' :        True,
        'figure.figsize': (8.4, 6),
        'legend.facecolor'     : '0.9',  # inherit from axes.facecolor; or color spec
        'legend.edgecolor'     : '0.9'      # background patch boundary color
        # expressed as a fraction of the average axis width
        # figure.subplot.hspace  : 0.2    # the amount of height reserved for white space between subplots,
        # expressed as a fraction of the average axis height
        # 'text.usetex': False,
        # 'font.family': 'monospace'
    }
    # fig, ax = plt.subplots()
    plt.rcParams.update(params)
    # plt.xlim(0.5,3)
    # plt.ylim(100, 140)
    plt.xlabel("Iterations")
    plt.ylabel("Average MakeSpan")
    # x, y = plot_data(l)
    # plt.plot(x,y, linewidth=5, linestyle='--', color='#006BB2')
    # plt.axvline(x=1, linewidth=5, color='k', alpha=0.4)
    # plt.legend(["ECSched-dp vs. Swarm", "ECSched-ml vs. Swarm"], loc = "best")
    # plt.savefig("test.eps" , bbox_inches='tight', form='eps', dpi=1200)
    x, y = read_data()
    print y
    # plt.axhline(y=104.0, linewidth=5, color='k', alpha=0.9)
    # plt.axhline(y=113.4, linewidth=5, color='g', alpha=0.9)
    # plt.axhline(y=103.4, linewidth=5, color='b', alpha=0.9)
    plt.plot(x, linewidth=5, color='#B22400')
    # plt.legend(["pack","shortest task first","critical path first","reinforcement learning"])
    plt.show()

if __name__ == "__main__":
    run()