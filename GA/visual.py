import matplotlib.pyplot as plt
import numpy as np

from utils import iter_time
from matplotlib_venn import venn2 
from scipy.stats import pearsonr

# [deprecated] data preprocess
def plot_speed_NFDs(v, k, start_date, end_date, size=10, savefig='.//speed-density.png'):
    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    dates = iter_time(start_date, end_date)
    for date in dates:
        plt.scatter(k[date], v[date], s=size, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('density (veh/km/lane)')
    plt.ylabel('speed (km/h)')
    plt.title('speed-density')
    
    # Add a legend
    plt.legend(dates, loc='upper right')
    plt.savefig(savefig)
    # Show the plot
    plt.show()

def plot_clean_stat(invalid, total, start_date, end_date, savefig='.//clean.png'): 
    # Generate some sample data
    labels    = iter_time(start_date, end_date)
    invalid_n = [invalid[date] for date in labels]
    total_n   = [total[date] for date in labels]
    # Create the bar chart
    plt.figure(figsize=(8, 6))
    
    bar_width = 0.3 
    index_invalid = np.arange(len(labels))  
    index_total   = index_invalid + bar_width 
    
   
    plt.bar(index_invalid, height=invalid_n, width=bar_width, label='invalid')
    plt.bar(index_total, height=total_n, width=bar_width, label='total')
    
    plt.legend()  
    plt.xticks(index_invalid + bar_width/2, labels)  
    plt.ylabel('#observations')  
    plt.title('invalid observations statistics') 
    
    plt.savefig(savefig)
    # Show the plot
    plt.show()

def plot_missing_data_venn(set_a, set_b, savefig='.//venn.png'):
    v = venn2((set_a, set_b))
    labels = ['10', '01']
    custom_text = ["from routes", "with detectors"]
    for label_id, text in zip(labels, custom_text):
        v.get_label_by_id(label_id).set_text(text)
    for text in v.set_labels:
        text.set_fontsize(10)
    for text in v.subset_labels:
        text.set_fontsize(5)
    plt.savefig(savefig)
    # Show the plot
    plt.show()

# dodme benchmark
def plot_ape(data, var_name='X', savefig='APE.png'): 
    # TODO: adjust num of bins
    tmp = [i * 100 for i in data]
    plt.hist(tmp, bins=40, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'APE of {var_name}')
    plt.xlabel('APE(%)')
    plt.ylabel('count')
    # plt.grid(axis='y', alpha=0.75)
    plt.savefig(var_name + '-' + savefig)
    plt.show()

def plot_geh(data, var_name='X', savefig='GEH.png'): 
    plt.boxplot(data, labels=var_name, sym='') # ignore outliers
    plt.title(f'GEH of {var_name}')
    plt.savefig(var_name + '-' + savefig)
    plt.show()

def plot_pearson_correlation_scatter(df_a, df_b, upper, var_name, savefig = 'r2.png'): 
    ''' input: data frame A, B '''
    r2, _ = pearsonr(df_a, df_b)
    plt.figure(figsize=(8, 8))
    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.scatter(df_a, df_b, color = 'blue', alpha = 0.5)
    plt.title(f'Pearson Correlation: {r2:.3f}') # TODO: rmse
    # plt.xlabel('Variable A')
    # plt.ylabel('Variable B')

    x = np.arange(0, upper)
    y = x
    plt.plot(x, y, color = 'r', linestyle='dashed')

    plt.grid()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(var_name + '-' + savefig)
    plt.show()



def plot_historical_objective_function_value(x, y_list, label_list, savefig='historical_of.png'): 
    for idx, y in enumerate(y_list):
        plt.plot(x, y, 'o-', label = label_list[idx])
    plt.ylabel('Objective Value')
    plt.xlabel('#iterations')
    plt.xticks(ticks = x, labels = [str(i) for i in x])
    plt.savefig(savefig)
    plt.show()
    plt.close()