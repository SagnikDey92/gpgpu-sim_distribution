#include "lock_fix.h"

std::map<std::vector<int>, std::set<std::pair<uint64_t, bool>>>  thr_lockset;

namespace tool {
    void acquire(uint64_t addr, ptx_thread_info* thread) {
        dim3 tid = thread->get_tid();
        dim3 cta = thread->get_ctaid();
        std::vector<int> ftid;
        ftid.push_back(tid.x); ftid.push_back(tid.y); ftid.push_back(tid.z);
        ftid.push_back(cta.x); ftid.push_back(cta.y); ftid.push_back(cta.z);
        thr_lockset[ftid].insert(std::make_pair(addr, false));
    }

    void release(uint64_t addr, ptx_thread_info* thread) {
        dim3 tid = thread->get_tid();
        dim3 cta = thread->get_ctaid();
        std::vector<int> ftid;
        ftid.push_back(tid.x); ftid.push_back(tid.y); ftid.push_back(tid.z);
        ftid.push_back(cta.x); ftid.push_back(cta.y); ftid.push_back(cta.z);
        thr_lockset[ftid].erase(std::make_pair(addr, false));
        thr_lockset[ftid].erase(std::make_pair(addr, true));
    }
}