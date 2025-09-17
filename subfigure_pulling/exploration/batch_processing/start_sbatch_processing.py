import json
from pathlib import Path
import subprocess as sp

workers = 50

for worker in range(workers):
  # not looping since the map has the regions in is already
    #for i in range(0,5):
  sp.run(f"sbatch -A visteam -J {worker}_processing -o slurm_{worker}_%A.out -p standard -t 4:00:00 -n 8 -N 1 coordinator.sh {worker}",shell=True) 
    #sp.run(f"bash coordinator.sh {region}",shell=True) 
