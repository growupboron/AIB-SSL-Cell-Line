#!/bin/sh
#--partition=all_serial
#--time=01:00


cd /work/cvcs_2023_group23/AIB 
. AIB_env/bin/activate


python download_dataset.py
