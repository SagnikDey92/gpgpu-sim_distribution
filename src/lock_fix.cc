#include "lock_fix.h"

std::map<std::vector<int>, std::set<std::pair<uint64_t, bool>>>  thr_lockset;
std::map<uint64_t, std::set<uint64_t>> addr_lockset;

std::vector<int> getTID(ptx_thread_info* thread) {
    dim3 tid = thread->get_tid();
    dim3 cta = thread->get_ctaid();
    std::vector<int> ftid;
    ftid.push_back(tid.x); ftid.push_back(tid.y); ftid.push_back(tid.z);
    ftid.push_back(cta.x); ftid.push_back(cta.y); ftid.push_back(cta.z);
}

namespace tool {
    void acquire(uint64_t addr, ptx_thread_info* thread) {
        std::vector<int> ftid = getTID(thread);
        thr_lockset[ftid].insert(std::make_pair(addr, false));
    }

    void release(uint64_t addr, ptx_thread_info* thread) {
        std::vector<int> ftid = getTID(thread);
        thr_lockset[ftid].erase(std::make_pair(addr, false));
        thr_lockset[ftid].erase(std::make_pair(addr, true));
    }

    void activate_locks(ptx_thread_info* thread) {
        std::vector<int> ftid = getTID(thread);
        std::set<std::pair<uint64_t, bool>> temp;
        for (std::pair<uint64_t, bool> p: thr_lockset[ftid]) {
            temp.insert(std::make_pair(p.first, true));
        }
        thr_lockset[ftid].clear();
        for (std::pair<uint64_t, bool> p: temp) {
            thr_lockset[ftid].insert(p);
        }
    }

    void write(uint64_t addr, ptx_thread_info* thread) {
        std::vector<int> ftid = getTID(thread);
        std::set<uint64_t> prev = addr_lockset[addr];
        std::set<uint64_t> curr;
        for (std::pair<uint64_t, bool> p: thr_lockset[ftid]) {
            if (p.second)
                curr.insert(p.first);
        }
        //Intersect into temp
        std::set<uint64_t> temp;
        for (uint64_t l: curr) {
            if (prev.count(l)) {
                temp.insert(l);
            }
        }
        //Place temp into addr lockset
        addr_lockset[addr].clear();
        for (uint64_t l: temp)
            addr_lockset[addr].insert(l);
        if (temp.empty()) {
            printf("Accessed address %x with no locks held!\n", addr);
        }
    }
}