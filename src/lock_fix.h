#include <stdlib.h>
#include <stdio.h>
#include <bitset>
#include <utility>
#include <set>
#include <queue>
#include <map>
#include <list>

#include "cuda-sim/ptx_sim.h"
#include "abstract_hardware_model.h"

namespace tool {
    void acquire(uint64_t addr, ptx_thread_info* thread);

    void release(uint64_t addr, ptx_thread_info* thread);
}