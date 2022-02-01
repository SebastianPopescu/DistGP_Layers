import os
import numpy as np
import os
import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type = int)
    args = parser.parse_args()

    cpu1=args.num_gpu*4
    cpu2=args.num_gpu*4 + 1
    cpu3=args.num_gpu*4 + 2
    cpu4=args.num_gpu*4 + 3
    string_cpus=str(cpu1)+','+str(cpu2)+','+str(cpu3)+','+str(cpu4)

    #for num_layers in range(3,6):
    #num_layers = 2
    #for dim_layers in [1, 5, 10]:

    dim_layers = 5
    dim_convolutionally_warped_gp = 12

    cmd = 'CUDA_VISIBLE_DEVICES='+str(args.num_gpu)+' taskset -c '+string_cpus+' /vol/biomedic2/sgp15/anaconda3/bin/python3 BrainStructure_Seg.py --num_layers='+str(7)+' --dim_filter=5 --dim_layers='+str(dim_layers)+' --num_inducing=100 --num_iterations=50001 --dim_convolutionally_warped_gp='+str(dim_convolutionally_warped_gp)
    os.system(cmd)
