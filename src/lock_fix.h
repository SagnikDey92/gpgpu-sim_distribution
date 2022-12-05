#include <stdlib.h>
#include <stdio.h>
#include <bitset>
#include <utility>
#include <set>
#include <queue>
#include <map>
#include <list>
#include <mutex>

#include "cuda-sim/ptx_sim.h"
#include "abstract_hardware_model.h"

std::map<std::vector<int>, int> delay;
std::map<uint64_t, std::mutex*> dLock;
std::map<std::mutex*, std::vector<int>> lockToThread;
std::map<std::vector<int>, std::mutex*> threadToLock;

int D = 3;  //Fix when thread exits

namespace tool {
    void acquire(uint64_t addr, ptx_thread_info* thread);

    void release(uint64_t addr, ptx_thread_info* thread);

    void activate_locks(ptx_thread_info* thread);

    void write(uint64_t addr, ptx_thread_info* thread);

    void exit_thr(ptx_thread_info* thread);

    void read(uint64_t addr, ptx_thread_info* thread);
}