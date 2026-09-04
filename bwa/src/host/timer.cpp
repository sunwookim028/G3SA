#include "timer.h"
#include "bwa.h"
#include "macro.h"
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cstring>
#include <algorithm>

float tprof[MAX_NUM_GPUS][MAX_NUM_STEPS];
char *step_name[MAX_NUM_STEPS] = {
    (char*)"smem",
    (char*)"reseed_r2",
    (char*)"reseed_r3",
    (char*)"sa_lookup",
    (char*)"sort_seeds",
    (char*)"chain",
    (char*)"sort_chains",
    (char*)"filter_chains",
    (char*)"sw_seed_prep",
    (char*)"local_extend",
    (char*)"filter_regions",
    (char*)"sort_regions",
    (char*)"tb_prep",
    (char*)"traceback",
    (char*)"finalize",
    (char*)"compute_total",
    (char*)"pull_total",
    (char*)"push_total",
    (char*)"gpu_setup",
    (char*)"file_input",
    (char*)"file_output",
    (char*)"aligner_top",
    (char*)"input_first",
    (char*)"samgen_total",
    (char*)"flt_chained_seeds",
    (char*)"flt_cont_regions",
    (char*)"patch_regions",
    (char*)"scan_offsets"
};

// Stage groupings for the profiler
struct stage_group_t {
    const char *name;
    int steps[8];  // indices into tprof, -1 terminated
};

static const stage_group_t stage_groups[] = {
    {"Seeding",     {S_SMEM, S_R2, S_R3, -1}},
    {"Chaining",    {C_SAL, C_SORT_SEEDS, C_CHAIN, C_SORT_CHAINS, C_FILTER, -1}},
    {"Extending",   {E_PAIRGEN, E_EXTEND, E_FILTER_MARK, E_SORT_ALNS, -1}},
    {"Traceback",   {E_T_PAIRGEN, E_TRACEBACK, E_FINALIZE, -1}},
};
static const int NUM_STAGE_GROUPS = 4;

static void print_bar(float pct, int width)
{
    int filled = (int)(pct / 100.0f * width + 0.5f);
    filled = std::min(filled, width);
    for (int i = 0; i < filled; i++) std::cerr << '#';
    for (int i = filled; i < width; i++) std::cerr << '.';
}

void report_stats(float tprof[MAX_NUM_GPUS][MAX_NUM_STEPS], g3_opt_t *g3_opt)
{
    int ngpus = g3_opt->num_use_gpus;
    float avg[MAX_NUM_STEPS];
    for (int i = 0; i < MAX_NUM_STEPS; i++) {
        float sum = 0;
        for (int g = 0; g < ngpus; g++) sum += tprof[g][i];
        avg[i] = sum / ngpus / 1000.0f;  // convert ms -> seconds
    }

    float kernel_total = 0;
    for (int i = S_SMEM; i <= E_FINALIZE; i++) kernel_total += avg[i];
    for (int i = E_FLT_SEEDS; i <= E_OFFSETS; i++) kernel_total += avg[i];

    std::cerr << "\n";
    if (kernel_total == 0) {
        std::cerr << "(kernel timing disabled — set G3_KERNEL_TIMING=1 to enable per-kernel breakdown)\n";
    } else {
        // Header
        std::cerr << "=== KERNEL STAGE PROFILE ===" << "\n";
        std::cerr << std::fixed << std::setprecision(4);
        std::cerr << "+-----------------------+---------+-------+----------------------------+\n";
        std::cerr << "| Kernel                | Time(s) |    %  | Bar                        |\n";
        std::cerr << "+-----------------------+---------+-------+----------------------------+\n";

        for (int i = S_SMEM; i <= E_FINALIZE; i++) {
            float pct = avg[i] / kernel_total * 100.0f;
            std::cerr << "| " << std::left << std::setw(22) << step_name[i]
                      << "| " << std::right << std::setw(7) << avg[i]
                      << " | " << std::setw(4) << std::setprecision(1) << pct << "% | ";
            print_bar(pct, 26);
            std::cerr << " |\n";
            std::cerr << std::setprecision(4);
        }
        for (int i = E_FLT_SEEDS; i <= E_OFFSETS; i++) {
            float pct = avg[i] / kernel_total * 100.0f;
            std::cerr << "| " << std::left << std::setw(22) << step_name[i]
                      << "| " << std::right << std::setw(7) << avg[i]
                      << " | " << std::setw(4) << std::setprecision(1) << pct << "% | ";
            print_bar(pct, 26);
            std::cerr << " |\n";
            std::cerr << std::setprecision(4);
        }
        std::cerr << "+-----------------------+---------+-------+----------------------------+\n";
        std::cerr << "| " << std::left << std::setw(22) << "KERNEL TOTAL"
                  << "| " << std::right << std::setw(7) << kernel_total
                  << " | 100.0% |                            |\n";
        std::cerr << "+-----------------------+---------+-------+----------------------------+\n";

        // Stage group summary
        std::cerr << "\n";
        std::cerr << "=== STAGE GROUP SUMMARY ===" << "\n";
        std::cerr << "+----------------+---------+-------+----------------------------+\n";
        std::cerr << "| Stage          | Time(s) |    %  | Bar                        |\n";
        std::cerr << "+----------------+---------+-------+----------------------------+\n";
        for (int g = 0; g < NUM_STAGE_GROUPS; g++) {
            float group_time = 0;
            for (int j = 0; stage_groups[g].steps[j] >= 0; j++)
                group_time += avg[stage_groups[g].steps[j]];
            float pct = group_time / kernel_total * 100.0f;
            std::cerr << "| " << std::left << std::setw(15) << stage_groups[g].name
                      << "| " << std::right << std::setw(7) << group_time
                      << " | " << std::setw(4) << std::setprecision(1) << pct << "% | ";
            print_bar(pct, 26);
            std::cerr << " |\n";
            std::cerr << std::setprecision(4);
        }
        std::cerr << "+----------------+---------+-------+----------------------------+\n";
    }

    // Pipeline overhead
    std::cerr << "\n";
    std::cerr << "=== PIPELINE OVERHEAD ===" << "\n";
    std::cerr << "+----------------+---------+\n";
    std::cerr << "| Component      | Time(s) |\n";
    std::cerr << "+----------------+---------+\n";
    std::cerr << "| " << std::left << std::setw(15) << "H2D memcpy"
              << "| " << std::right << std::setw(7) << avg[PUSH_TOTAL] << " |\n";
    std::cerr << "| " << std::left << std::setw(15) << "D2H memcpy"
              << "| " << std::right << std::setw(7) << avg[PULL_TOTAL] << " |\n";
    std::cerr << "| " << std::left << std::setw(15) << "SAM gen"
              << "| " << std::right << std::setw(7) << avg[SAMGEN_TOTAL] << " |\n";
    std::cerr << "| " << std::left << std::setw(15) << "File I/O"
              << "| " << std::right << std::setw(7) << avg[FILE_OUTPUT] << " |\n";
    std::cerr << "| " << std::left << std::setw(15) << "GPU setup"
              << "| " << std::right << std::setw(7) << avg[GPU_SETUP] << " |\n";
    std::cerr << "| " << std::left << std::setw(15) << "Compute total"
              << "| " << std::right << std::setw(7) << avg[COMPUTE_TOTAL] << " |\n";
    std::cerr << "| " << std::left << std::setw(15) << "Wall (aligner)"
              << "| " << std::right << std::setw(7) << avg[ALIGNER_TOP] << " |\n";
    std::cerr << "+----------------+---------+\n";

    // Sync overhead estimate: compute_total - kernel_total = time spent in cudaDeviceSynchronize
    float sync_overhead = avg[COMPUTE_TOTAL] - kernel_total;
    if (sync_overhead > 0 && kernel_total > 0) {
        std::cerr << "\n";
        std::cerr << "Sync overhead (compute_total - kernel_total): "
                  << sync_overhead << "s ("
                  << std::setprecision(1) << (sync_overhead / avg[COMPUTE_TOTAL] * 100.0f)
                  << "% of compute)" << "\n";
        std::cerr << std::setprecision(4);
    }
    std::cerr << "\n";
}
