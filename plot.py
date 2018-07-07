import numpy
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def calc_sma(y_values):
    """Calculate the moving average for one line (given as two lists, one
    for its x-values and one for its y-values).
    Args:
        x_values: x-coordinate of each value.
        y_values: y-coordinate of each value.
    Returns:
        Tuple (x_values, y_values), where x_values are the x-values of
        the line and y_values are the y-values of the line.
    """
    result_y, last_ys = [], []
    running_sum = 0
    period = 20
    # use a running sum here instead of avg(), should be slightly faster
    for y_val in y_values:
        last_ys.append(y_val)
        running_sum += y_val
        if len(last_ys) > period:
            poped_y = last_ys.pop(0)
            running_sum -= poped_y
        result_y.append(float(running_sum) / float(len(last_ys)))
    return result_y

def plotter(args, correction, color, path):
    samples = {'x':[], 'y':[]}
    avg = {'x':[], 'y':[]}

    filename = '../logs/' + path + '/' + args.dataset[0]
    filename += '_' + str(args.num_workers)
    filename += '_' + correction
    if not args.number is None:
        filename += ' (' + str(args.number) + ')'
    filename += '.log'

    with open(filename, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line[:4] == 'Test':
                if args.accuracy is False:
                    loss = float(line.split('loss: ')[1].split(',')[0])
                else:
                    loss = float(line.split('(')[1].split('%')[0])
                samples['y'].append(loss)
                samples['x'].append(i * 10 + 10)

    avg['x'] = samples['x']
    avg['y'] = calc_sma(samples['y'])
    if args.clean is False:
        plt.plot(samples['x'], samples['y'], color+'-')
        note = ''
    else:
        plt.plot(avg['x'], avg['y'], color+'-')
        note = ' (avg 20)'

    line = mlines.Line2D([], [], color=color, markersize=15, label=str(args.num_workers) + ' Workers, ' + correction.capitalize() + note)
    return line


parser = argparse.ArgumentParser(description='Plot parser')
parser.add_argument('dataset', nargs=1, choices=['CIFAR10', 'MNIST'],
                    help='Working dataset')
parser.add_argument('-nw', '--num-workers', type=int, default=1, metavar='N',
                    help='Training batch size')
parser.add_argument('-acc', '--accuracy', action='store_true',
                    help='Graph accuracy or loss. Default loss')
parser.add_argument('-c', '--clean', action='store_true',
                    help='Clean graph using running average')
parser.add_argument('-n', '--number', type=int, default=None,
                    help='Number trial')
parser.add_argument('-com', '--compare-one', action='store_true',
                    help='Compare to single computer trial')

args = parser.parse_args()
args.clean = args.clean or args.accuracy
args.compare_one = args.compare_one or args.accuracy

clean = 'Clean' if args.clean else 'Noisy'
acc = 'Accuracy' if args.accuracy else 'Loss'
num = '0' if args.number is None else str(args.number)
filename = '../logs/' + str(args.num_workers) + ' Workers - ' + acc + ' - 3rd Attempt - ' + num + ' - ' + clean

plt.xlabel('Epoch Number')
plt.xticks(range(0, 390 * 180, 10*390), (str(i) for i in range(0, 180, 10)))
if args.accuracy:
    plt.ylabel('Accuracy')
    plt.yticks(range(0, 100, 5), (str(i) for i in range(0, 100, 5)))
    title = 'Accuracy'
else:
    plt.ylabel('Loss')
    plt.yticks([x/10 for x in range(0, 40, 2)], (str(i/10) for i in range(0, 40, 2)))
    title = 'Loss'
plt.title(args.dataset[0] + ', ' + str(args.num_workers) + ' Workers, ' + title)
plt.grid(True)

handles = []
handles += [plotter(args, 'none', 'r', 'resnet_1st')]
handles += [plotter(args, 'master', 'b', 'resnet_1st')]
handles += [plotter(args, 'master', 'm', 'resnet_3rd')]
handles += [plotter(args, 'master', 'c', 'resnet_4th')]
handles += [plotter(args, 'worker', 'g', 'resnet_2nd')] 

 if args.compare_one:
     args.num_workers = 1
     handles += [plotter(args, 'none', 'y', 'resnet_1st')]

plt.legend(handles=handles)
plt.show()
# plt.savefig(filename)
