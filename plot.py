import numpy
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plotter(args, correction, color):
    samples = {'x':[], 'y':[]}
    df = {'x':[], 'y':[]}

    filename = '../logs/server_logs/' + args.dataset[0]
    filename += '_' + str(args.num_workers)
    filename += '_' + correction
    if not args.number is None:
        filename += ' (' + str(args.number) + ')'
    filename += '.log'

    with open(filename, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line[:4] == 'Test':
                loss = float(line.split('loss: ')[1].split(',')[0])
                samples['y'].append(loss)
                samples['x'].append(i * 10 + 10)

    df['x'] = samples['x']
    func = numpy.polyfit(numpy.log(samples['x']), samples['y'], 1)
    for i in df['x']:
        df['y'].append(func[0] * numpy.log(i) + func[1])
    #plt.plot(samples['x'], samples['y'], color+'-')
    plt.plot(df['x'], df['y'], color+'-')

    #function = 'y=' + str(func[0]) + 'log(x) + ' + str(func[1])
    line = mlines.Line2D([], [], color=color, markersize=15, label='Regression ' + str(args.num_workers) + ', ' + correction)
    #red_line = mlines.Line2D([], [], color='red', markersize=15, label='Logarithmic Regression\n'+function)
    #plt.legend(handles=[blue_line, red_line])
    return line


parser = argparse.ArgumentParser(description='Plot parser')
parser.add_argument('dataset', nargs=1, choices=['CIFAR10', 'MNIST'],
                    help='Working dataset')
parser.add_argument('-nw', '--num-workers', type=int, default=1, metavar='N',
                    help='Training batch size')
#parser.add_argument('-gc', '--gradient-correction', choices=['none', 'worker', 'master'], default='none',
#                    help='Correction type to use. Possible options: master, worker, none')
parser.add_argument('-n', '--number', type=int, default=None,
                    help='Number trial')

args = parser.parse_args()

plt.xlabel('Epoch Number')
plt.xticks(range(0, 390 * 11, 390), (str(i) for i in range(11)))
plt.ylabel('Loss')
plt.title(args.dataset[0] + ', workers: ' + str(args.num_workers))

nLine = plotter(args, 'none', 'r')
mLine = plotter(args, 'master', 'b')
wLine = plotter(args, 'worker', 'y')
args.number = None
args.num_workers = 1
oLine = plotter(args, 'none', 'g')

plt.legend(handles=[oLine, nLine, mLine, wLine])

plt.show()
