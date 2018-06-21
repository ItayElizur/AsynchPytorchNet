#!/bin/bash 
COUNTER=0
while [ $COUNTER -lt 3 ]; do
	srun -c2 -N1 --gres=gpu:1 --job-name="just_a_test" python master.py CIFAR10 -nw 1
    srun -c2 -N1 --gres=gpu:1 --job-name="just_a_test" python master.py CIFAR10 -nw 4
	srun -c2 -N1 --gres=gpu:1 --job-name="just_a_test" python master.py CIFAR10 -nw 4 -gc worker
	srun -c2 -N1 --gres=gpu:1 --job-name="just_a_test" python master.py CIFAR10 -nw 4 -gc master
	srun -c2 -N1 --gres=gpu:1 --job-name="just_a_test" python master.py CIFAR10 -nw 8
	srun -c2 -N1 --gres=gpu:1 --job-name="just_a_test" python master.py CIFAR10 -nw 8 -gc worker
	srun -c2 -N1 --gres=gpu:1 --job-name="just_a_test" python master.py CIFAR10 -nw 8 -gc master
	let COUNTER=COUNTER+1
done