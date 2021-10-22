import os
from matplotlib import pyplot as plt
from spectrl.util.io import (plot_error_bar, extract_plot_data, save_plot,
                             parse_command_line_options, extract_plot_data_rm)

# mypy: ignore-errors

'''
Note: In order to plot RM baselines results have to be copied from
the results folder within reward_machines folder to folder with
DiRL results.
Comment out baselines for which results are not available to obtain
partial plots.
'''

'''
0: Samples
1: Time
-1: Probability
'''

x_index = 0
start_itno = 0
x_max = 1e9

flags = parse_command_line_options()
folder_n = flags['folder']
itno = flags['itno']
spec = flags['spec_num']
folder = os.path.join(folder_n, 'spec{}'.format(spec))

plot_name = "spec"+str(spec)

hierarchy_folder = os.path.join(folder, "hierarchy")
spectrl_folder = os.path.join(folder, "spectrl")
tltl_folder = os.path.join(folder, "tltl")
ddpg_basic = os.path.join(folder, "ddpg/basic")
ddpg_cr = os.path.join(folder, "ddpg/cr")
dhrm_basic = os.path.join(folder, "dhrm/basic")
dhrm_cr = os.path.join(folder, "dhrm/cr")


xh, _, _, _ = extract_plot_data(hierarchy_folder, x_index, start_itno, itno)
yh = extract_plot_data(hierarchy_folder, -1, start_itno, itno)
plot_error_bar(xh, yh, 'blue', 'DiRL (Ours)', points=True)
x_max = min(x_max, xh[-1]+3000)


xs, _, _, _ = extract_plot_data(spectrl_folder, x_index, start_itno, itno)
ys = extract_plot_data(spectrl_folder, -1, start_itno, itno)
plot_error_bar(xs, ys, 'seagreen', 'SPECTRL')
x_max = min(x_max, xs[-1])

xt, _, _, _ = extract_plot_data(tltl_folder, x_index, start_itno, itno)
yt = extract_plot_data(tltl_folder, -1, start_itno, itno)
plot_error_bar(xt, yt, 'tomato', 'TLTL')
x_max = min(x_max, xt[-1])

ddpg_basic = extract_plot_data_rm(ddpg_basic, -5, 1, start_itno, itno, True)
plot_error_bar(ddpg_basic[0], ddpg_basic[1:], 'orange', 'QRM')
x_max = min(x_max, ddpg_basic[0][-1])  # no mypy


ddpg_cr = extract_plot_data_rm(ddpg_cr, -5, 1, start_itno, itno, True)
plot_error_bar(ddpg_cr[0], ddpg_cr[1:], 'pink', 'QRM+CR')
x_max = min(x_max, ddpg_cr[0][-1])


dhrm_basic = extract_plot_data_rm(dhrm_basic, 3, 2, start_itno, itno, True)
plot_error_bar(dhrm_basic[0], dhrm_basic[1:], 'red', 'HRM')
x_max = min(x_max, dhrm_basic[0][-1])

dhrm_cr = extract_plot_data_rm(dhrm_cr, 3, 2, start_itno, itno, True)
plot_error_bar(dhrm_cr[0], dhrm_cr[1:], 'purple', 'HRM+CR')
x_max = min(x_max, dhrm_cr[0][-1])

# print(x_max)

plt.xlim(right=x_max, left=-0.01)

save_plot(folder_n, plot_name, False)

