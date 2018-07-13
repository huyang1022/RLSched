import numpy as np
import matplotlib.pyplot as plt


def read_data():
    ret_list = []
    in_file = open("log/rl_log","r")
    for i, line in enumerate(in_file.readlines()):
        if i % 11 == 9:
            ret_n = line.split()
            ret_list.append(float(ret_n[1]))
    return ret_list


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
    # plt.ylim(50,120.5)
    plt.xlabel("Iterations")
    # plt.ylabel("Fraction of Containers (%)")
    # x, y = plot_data(l)
    # plt.plot(x,y, linewidth=5, linestyle='--', color='#006BB2')
    # plt.axvline(x=1, linewidth=5, color='k', alpha=0.4)
    # plt.legend(["ECSched-dp vs. Swarm", "ECSched-ml vs. Swarm"], loc = "best")
    # plt.savefig("test.eps" , bbox_inches='tight', form='eps', dpi=1200)
    x = read_data()
    print min(x)
    plt.axhline(y=14, linewidth=5, color='k', alpha=0.5)
    plt.plot(x, linewidth=5, color='#B22400')
    plt.show()

if __name__ == "__main__":
    run()