import argparse
import os
import sys
import numpy as np
import pandas as pd
import numpy as np
import e3nn_jax as e3nn
from freedom.tensor_products.vector_spherical_harmonics import VSHCoeffs

from absl import flags

flags.DEFINE_string("profiles_path", "nsight_profiles", "Path to the Nsight CSV files")
flags.DEFINE_string(
    "output_path",
    "benchmarking/csv/nsight_profiling.csv",
    "Path to the Processed CSV file",
)
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def n_irreps_normalization(lmax, irreps_type, tensor_product_type):
    """
    Returns # output_irreps for CGTP-dense/CGTP-sparse
    and # output_irreps + # input_irreps - 2 for Matrix-TP/GTP-grid/GTP-fourier/VGTP-grid
    Currently only support MIMO and returns 1 by default
    """
    lmax = int(lmax)
    if irreps_type == "MIMO":
        if tensor_product_type in ["CGTP-dense", "CGTP-sparse"]:
            return e3nn.tensor_product(
                e3nn.s2_irreps(lmax), e3nn.s2_irreps(lmax)
            ).num_irreps
        elif tensor_product_type in ["Matrix-TP", "GTP-grid", "GTP-fourier"]:
            return (
                e3nn.s2_irreps(2 * lmax).num_irreps
                + 2 * e3nn.s2_irreps(lmax).num_irreps
                - 2
            )
    return 1


files = [x for x in os.listdir(FLAGS.profiles_path) if x.endswith(".csv")]

df_integrated = pd.DataFrame(
    columns=[
        "irreps_type",
        "tensor_product_type",
        "lmax",
        "batch",
        "normalization",
        "Time",
        "all GFLOPs",
        "GB",
        "TC GFLOPs/s",
        "CC GFLOPs/s",
        "all GFLOPs/s",
        "GB/s",
    ]
)

for iloc, file in enumerate(files):
    try:
        tag, ext = os.path.splitext(os.path.basename(file))
        # TODO: Put in a _jax while running nsight with JAX
        if len(tag.split("_")) == 7:
            _, irreps_type, tensor_product_type, lmax, batch, lmax_grid, _ = tag.split(
                "_"
            )
        if len(tag.split("_")) == 6:
            _, irreps_type, tensor_product_type, lmax, batch, _ = tag.split("_")
        elif len(tag.split("_")) == 5:
            _, irreps_type, tensor_product_type, lmax, batch = tag.split("_")
        else:
            continue
            _, tp_type, lmax, multiplicity, layers, _ = tag.split("_")
        irrep_normalization = n_irreps_normalization(
            lmax, irreps_type, tensor_product_type
        )
        file_path = os.getcwd() + "/" + FLAGS.profiles_path + "/" + file
        df = pd.read_csv(file_path, skiprows=4)
        df["Metric Value"] = pd.to_numeric(
            df["Metric Value"].str.replace(r",", "", regex=True)
        )
        dft = df.groupby(["Kernel Name", "Metric Name"]).sum()
        dfmetric = pd.pivot_table(
            dft, index="Kernel Name", columns="Metric Name", values="Metric Value"
        )
        dfmetric["Count"] = (
            df.groupby(["Kernel Name"]).count()["ID"].div(dfmetric.shape[1])
        )

        # dfmetric['Time']=dfmetric['sm__cycles_elapsed.avg'] \
        #                 / (dfmetric['sm__cycles_elapsed.avg.per_second'] /dfmetric['Count'] )
        dfmetric["Time"] = (
            dfmetric["gpu__time_duration.sum"] / 10**9
        )  # Because this time is in ns

        dfmetric["CC FLOPs"] = (
            2 * dfmetric["sm__sass_thread_inst_executed_op_dfma_pred_on.sum"]
            + dfmetric["sm__sass_thread_inst_executed_op_dmul_pred_on.sum"]
            + dfmetric["sm__sass_thread_inst_executed_op_dadd_pred_on.sum"]
            + 2 * dfmetric["sm__sass_thread_inst_executed_op_ffma_pred_on.sum"]
            + dfmetric["sm__sass_thread_inst_executed_op_fmul_pred_on.sum"]
            + dfmetric["sm__sass_thread_inst_executed_op_fadd_pred_on.sum"]
            + 2 * dfmetric["sm__sass_thread_inst_executed_op_hfma_pred_on.sum"]
            + dfmetric["sm__sass_thread_inst_executed_op_hmul_pred_on.sum"]
            + dfmetric["sm__sass_thread_inst_executed_op_hadd_pred_on.sum"]
        )

        MAGIC_NUMBER = 2048  # Ampere
        # MAGIC_NUMBER = 512 # Turing

        dfmetric["TC FLOPs"] = (
            MAGIC_NUMBER * dfmetric["sm__inst_executed_pipe_tensor.sum"]
        )
        # Don't know where that 512 is coming from
        dfmetric["all FLOPs"] = dfmetric["CC FLOPs"] + dfmetric["TC FLOPs"]

        # dfmetric['AI HBM'] = dfmetric['all FLOPs'].div(dfmetric['DRAM Bytes'])
        # dfmetric['AI L2'] = dfmetric['all FLOPs'].div(dfmetric['lts__t_bytes.sum'])
        # dfmetric['AI L1'] = dfmetric['all FLOPs'].div(dfmetric['l1tex__t_bytes.sum'])
        dfmetric["all GFLOPs"] = dfmetric["all FLOPs"] / 1024 / 1024 / 1024
        dfmetric["all GFLOPs/s"] = dfmetric["all GFLOPs"] / dfmetric["Time"]

        dfmetric["TC GFLOPs"] = dfmetric["TC FLOPs"] / 1024 / 1024 / 1024
        dfmetric["TC GFLOPs/s"] = dfmetric["TC GFLOPs"] / dfmetric["Time"]

        dfmetric["CC GFLOPs"] = dfmetric["CC FLOPs"] / 1024 / 1024 / 1024
        dfmetric["CC GFLOPs/s"] = dfmetric["CC GFLOPs"] / dfmetric["Time"]

        # dfmetric['Read Throughput GB/s'] = dfmetric['l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second']/1024**3
        # dfmetric['Write Throughput GB/s'] = dfmetric['l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second']/1024**3
        # dfmetric['Total Throughput GB/s'] = dfmetric['Read Throughput GB/s'] + dfmetric['Write Throughput GB/s']
        dfmetric["DRAM GB"] = (
            (
                dfmetric["dram__bytes.sum"]
                if "dram__bytes.sum" in dfmetric
                else (
                    dfmetric["dram__bytes_read.sum"] + dfmetric["dram__bytes_write.sum"]
                )
            )
            / 1024
            / 1024
            / 1024
        )
        dfmetric["DRAM GB/s"] = dfmetric["DRAM GB"] / dfmetric["Time"]

        # total_time_measured = np.sum(dfmetric['gpu__time_duration.sum'])
        total_time = np.sum(dfmetric["Time"])
        total_gflops = np.sum(dfmetric["all GFLOPs"])
        total_gb = np.sum(dfmetric["DRAM GB"])
        mean_tc_gflops_s = np.mean(dfmetric["TC GFLOPs/s"])
        mean_cc_gflops_s = np.mean(dfmetric["CC GFLOPs/s"])
        mean_gflops_s = np.mean(dfmetric["all GFLOPs/s"])
        mean_gb_s = np.mean(dfmetric["DRAM GB/s"])

        # Print summary statistics
        print(file_path)
        df_integrated.loc[iloc] = [
            irreps_type,
            tensor_product_type,
            lmax,
            batch,
            irrep_normalization,
            total_time,
            total_gflops,
            total_gb,
            mean_tc_gflops_s,
            mean_cc_gflops_s,
            mean_gflops_s,
            mean_gb_s,
        ]
    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError) as e:
        print(file_path)

df_integrated.to_csv(FLAGS.output_path)


# print("Measured L1 GB:", sum(dfmetric['l1tex__t_bytes.sum'].to_list()) /1024/1024/1024)
# print("L1 cache hit rate:", np.mean(((dfmetric["l1tex__t_sector_hit_rate.pct"])).to_list()), "+/-", np.std(((dfmetric["l1tex__t_sector_hit_rate.pct"])).to_list()))
# print("Measured L2 GB:", sum(dfmetric['lts__t_bytes.sum'].to_list()) /1024/1024/1024)
# print("L2 cache hit rate:", 100*np.mean(((dfmetric["lts__t_sectors_srcunit_tex_lookup_hit.sum"])/(dfmetric["lts__t_sectors_srcunit_tex_lookup_hit.sum"] + dfmetric["lts__t_sectors_srcunit_tex_lookup_miss.sum"])).to_list()), "+/-",
#                             100*np.std(((dfmetric["lts__t_sectors_srcunit_tex_lookup_hit.sum"])/(dfmetric["lts__t_sectors_srcunit_tex_lookup_hit.sum"] + dfmetric["lts__t_sectors_srcunit_tex_lookup_miss.sum"])).to_list()))
