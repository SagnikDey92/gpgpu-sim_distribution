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

extern std::map<std::vector<int>, int> delay;
extern std::map<uint64_t, std::mutex*> dLock;
extern std::map<std::mutex*, std::vector<int>> lockToThread;
extern std::map<std::vector<int>, std::set<std::mutex*>> threadToLock;

extern int D;  //Fix when thread exits

namespace tool {
    void acquire(uint64_t addr, ptx_thread_info* thread);

    void release(uint64_t addr, ptx_thread_info* thread);

    void activate_locks(ptx_thread_info* thread);

    void write(uint64_t addr, ptx_thread_info* thread, bool w);

    void exit_thr(ptx_thread_info* thread);

    void read(uint64_t addr, ptx_thread_info* thread);
}
