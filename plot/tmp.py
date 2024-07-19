import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

tv_plot_colors = ['darkorange','moccasin','lightgray','darkgray','k']
#tv_plot_colors = ['olivedrab','yellowgreen','k']
#tv_plot_colors = ['dodgerblue','deepskyblue','darkkhaki','khaki','gold'] + ['lightgray','k']

labels = ['TV-Gaze','TV-Exposure','Missing','Powered-off','No device'] 
#labels = ['Mobile','Unknown','No device']
#labels = ['Wear-time','Sleeping','Sedentary','Light', 'MVPA'] + ['Non-wear','No device']


matplotlib.rcParams['font.size'] = 14.0
fig, ax = plt.subplots()

patches = []
for i in range(len(tv_plot_colors)):
    patches.append(mpatches.Patch(color=tv_plot_colors[i], label=labels[i]))

ax.legend(handles=patches)
plt.show()

