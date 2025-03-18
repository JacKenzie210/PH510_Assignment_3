#The "job script" to run it on local pc

Num_cores = 5

from IPython import get_ipython

ip = get_ipython()

ip.run_cell(f"!mpiexec -n {Num_cores} python monte_carlo.py")