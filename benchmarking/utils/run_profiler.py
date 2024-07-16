import argparse
import subprocess
import time
import sys

# Time
metrics = "sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,"

# DP
metrics += "sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"

# SP
metrics += "sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"

# HP
metrics += "sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"

# Tensor Core
metrics += "sm__inst_executed_pipe_tensor.sum,"

# DRAM, L2 and L1
metrics += "dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,"

# Throughput
metrics += "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,"


parser = argparse.ArgumentParser(description='Run a Python script with profiling')
parser.add_argument('input_file', help='Path to the input Python file')
parser.add_argument('csv_file', help='Path to the output csv file')
parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments to pass to the input file')

args = parser.parse_args()

input_file = args.input_file
output_file = args.csv_file
input_args = ' '.join(args.args)
profile_str = f"ncu --target-processes all --profile-from-start 0 --nvtx --nvtx-include \"profile\" --metrics {metrics} --csv --print-units base"

print(input_args)
try:
    start = time.process_time()
    subprocess.run(f"{profile_str} python {input_file} {input_args} > {output_file}", shell=True, check=True)
    print(f"Profiling took {(time.process_time() - start):.6f} s")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    sys.exit(1)
