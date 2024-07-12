import argparse
import os
import sys
import numpy as np
import pandas as pd
import numpy as np

from absl import flags
flags.DEFINE_string('profiles_path', 'nsight_profiles', 'Path to the Nsight CSV files')
flags.DEFINE_string('output_path', 'nsight_processed.csv', 'Path to the Processed CSV file')
FLAGS = flags.FLAGS
FLAGS(sys.argv)

files=[x for x in os.listdir(FLAGS.profiles_path) if x.endswith('.csv')]
files.sort()
df_integrated = pd.DataFrame(columns=['irreps_type', 'tensor_product_type', 'lmax', 'GFLOPs/s (mean)', "GFLOPs/s (std)", "GB/s", "GB/s (std)"])

from scipy import stats

def report_mean_and_std(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    return mean, se


for iloc, file in enumerate(files):
    tag, ext = os.path.splitext(os.path.basename(file))
    _, irreps_type, tensor_product_type, lmax = tag.split("_")
    file_path = os.getcwd() + '/' + FLAGS.profiles_path + '/' + file
    df = pd.read_csv(file_path, skiprows=2)

    df['Metric Value'] =pd.to_numeric(df['Metric Value'].str.replace(r',','', regex=True))
    dft=df.groupby(['Kernel Name','Metric Name']).sum()
    dfmetric=pd.pivot_table(dft, index='Kernel Name', columns='Metric Name', values='Metric Value')
    dfmetric['Count']=df.groupby(['Kernel Name']).count()['ID'].div(dfmetric.shape[1])

    dfmetric['Time']=dfmetric['sm__cycles_elapsed.avg'] \
                    / (dfmetric['sm__cycles_elapsed.avg.per_second'] /dfmetric['Count'] )

    dfmetric['CC FLOPs']  = 2 * dfmetric['sm__sass_thread_inst_executed_op_dfma_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_dmul_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_dadd_pred_on.sum'] \
                            + 2 * dfmetric['sm__sass_thread_inst_executed_op_ffma_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_fmul_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_fadd_pred_on.sum'] \
                            + 2 * dfmetric['sm__sass_thread_inst_executed_op_hfma_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_hmul_pred_on.sum'] \
                            + dfmetric['sm__sass_thread_inst_executed_op_hadd_pred_on.sum']

    MAGIC_NUMBER = 2048 # Ampere
    # MAGIC_NUMBER = 512 # Turing

    dfmetric['TC FLOPs']= MAGIC_NUMBER * dfmetric['sm__inst_executed_pipe_tensor.sum']
    # Don't know where that 512 is coming from
    dfmetric['all FLOPs']= dfmetric['CC FLOPs'] + dfmetric['TC FLOPs']

    dfmetric['AI HBM'] = dfmetric['all FLOPs'].div(dfmetric['dram__bytes.sum'])
    dfmetric['AI L2'] = dfmetric['all FLOPs'].div(dfmetric['lts__t_bytes.sum'])
    dfmetric['AI L1'] = dfmetric['all FLOPs'].div(dfmetric['l1tex__t_bytes.sum'])
    dfmetric['all GFLOPs'] = dfmetric['all FLOPs']/1024/1024/1024
    dfmetric['GFLOPs/s'] = dfmetric['all GFLOPs']/ dfmetric['Time']
    dfmetric['TC GFLOPs'] = dfmetric['TC FLOPs']/1024/1024/1024
    dfmetric['TC GFLOPs/s'] = dfmetric['TC GFLOPs']/ dfmetric['Time'].to_list()
    dfmetric['Read Throughput GB/s'] = dfmetric['l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second']/1024**3
    # dfmetric['Write Throughput GB/s'] = dfmetric['l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second']/1024**3
    # dfmetric['Total Throughput GB/s'] = dfmetric['Load Throughput GB/s'] + dfmetric['Read Throughput GB/s']

    
    gflops_s, gflops_s_error = report_mean_and_error(dfmetric['GFLOPs/s'])
    gb_s, gb_s_error = report_mean_and_error(dfmetric['Read Throughput GB/s'])
    df_integrated.loc[iloc] = [irreps_type, tensor_product_type, lmax, gflops_s, gflops_s_error, gb_s, gb_s_error]


df_integrated.to_csv(FLAGS.output_path)


# print("Measured L1 GB:", sum(dfmetric['l1tex__t_bytes.sum'].to_list()) /1024/1024/1024)
# print("L1 cache hit rate:", np.mean(((dfmetric["l1tex__t_sector_hit_rate.pct"])).to_list()), "+/-", np.std(((dfmetric["l1tex__t_sector_hit_rate.pct"])).to_list()))
# print("Measured L2 GB:", sum(dfmetric['lts__t_bytes.sum'].to_list()) /1024/1024/1024)
# print("L2 cache hit rate:", 100*np.mean(((dfmetric["lts__t_sectors_srcunit_tex_lookup_hit.sum"])/(dfmetric["lts__t_sectors_srcunit_tex_lookup_hit.sum"] + dfmetric["lts__t_sectors_srcunit_tex_lookup_miss.sum"])).to_list()), "+/-",
#                             100*np.std(((dfmetric["lts__t_sectors_srcunit_tex_lookup_hit.sum"])/(dfmetric["lts__t_sectors_srcunit_tex_lookup_hit.sum"] + dfmetric["lts__t_sectors_srcunit_tex_lookup_miss.sum"])).to_list()))