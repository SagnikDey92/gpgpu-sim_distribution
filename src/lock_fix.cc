#include "lock_fix.h"

std::map<std::vector<int>, std::set<std::pair<uint64_t, bool>>>  thr_lockset;
std::map<uint64_t, std::set<uint64_t>> addr_lockset;

std::map<std::vector<int>, int> delay;
std::map<uint64_t, std::mutex*> dLock;
std::map<std::mutex*, std::vector<int>> lockToThread;
std::map<std::vector<int>, std::set<std::mutex*>> threadToLock;

int D = 3;  //Fix when thread exits

std::vector<int> getTID(ptx_thread_info* thread, unsigned warp_size) {
    int lane_id = thread->get_hw_tid() % warp_size; 
    dim3 tid = thread->get_tid();
    dim3 cta = thread->get_ctaid();
    std::vector<int> ftid;
    ftid.push_back(tid.x); ftid.push_back(tid.y); ftid.push_back(tid.z);
    ftid.push_back(cta.x); ftid.push_back(cta.y); ftid.push_back(cta.z);
    ftid.push_back(lane_id);
    return ftid;
}

namespace tool {
    void acquire(uint64_t addr, ptx_thread_info* thread, unsigned warp_size) {
        std::vector<int> ftid = getTID(thread, warp_size);
        thr_lockset[ftid].insert(std::make_pair(addr, false));
    }

    void release(uint64_t addr, ptx_thread_info* thread, unsigned warp_size) {
        std::vector<int> ftid = getTID(thread, warp_size);
        thr_lockset[ftid].erase(std::make_pair(addr, false));
        thr_lockset[ftid].erase(std::make_pair(addr, true));
    }

    void activate_locks(ptx_thread_info* thread, unsigned warp_size) {
        std::vector<int> ftid = getTID(thread, warp_size);
        std::set<std::pair<uint64_t, bool>> temp;
        for (std::pair<uint64_t, bool> p: thr_lockset[ftid]) {
            temp.insert(std::make_pair(p.first, true));
        }
        thr_lockset[ftid].clear();
        for (std::pair<uint64_t, bool> p: temp) {
            thr_lockset[ftid].insert(p);
        }
    }

    void write(uint64_t addr, ptx_thread_info* thread, bool w, unsigned warp_size) {
        std::vector<int> ftid = getTID(thread, warp_size);
        printf("tool: (");
        for (int i = 0; i<7; ++i) {
            printf("%d,", ftid[i]);
        }
        if (w)
            printf("): write at %x\n", addr);
        else
            printf("): read at %x\n", addr);
        std::set<uint64_t> prev;
        std::set<uint64_t> curr;
        bool all = false;
        if (addr_lockset.find(addr) != addr_lockset.end())
            prev = addr_lockset[addr];
        else 
            all = true; //Lockset has everything

        if (delay.find(ftid) != delay.end() && delay[ftid]!=0) {
            delay[ftid]--;  // Reduce delay
            if (delay[ftid] == 0) {
                //Unlock
                printf("tool: cta: %d is releasing locks: ", ftid[3]);
                for (std::mutex* L: threadToLock[ftid]) {
                    printf("%x ", L);
                    L->unlock();
                }
                printf("\n");
                threadToLock[ftid].clear();
            }
        }

        for (std::pair<uint64_t, bool> p: thr_lockset[ftid]) {
            if (p.second)
                curr.insert(p.first);
        }

        //Intersect into temp
        std::set<uint64_t> temp;
        for (uint64_t l: curr) {
            if (all || prev.count(l)) {
                temp.insert(l);
            }
        }

        //Place temp into addr lockset
        addr_lockset[addr].clear();
        for (uint64_t l: temp)
            addr_lockset[addr].insert(l);
        if (temp.empty()) {
            if (dLock.find(addr) == dLock.end()) {
               std::mutex* L = new std::mutex(); 
               dLock[addr] = L;
               printf("tool: Found addr: %x without locks. Associated lock %x to it.\n", addr, L);
            }
            std::mutex* L = dLock[addr];
	    if (lockToThread[L] != ftid) {
		    if(L->try_lock()) {
		        printf("tool: cta: %d got lock %x\n", ftid[3], L);
			delay[ftid] = D;
		        lockToThread[L] = ftid;
		        threadToLock[ftid].insert(L);
		        thread->m_loop = false;
		    } else {
		        printf("tool: cta: %d is looping!\n", ftid[3]);
		        thread->m_loop = true;
		    }
	    }
        }
    }

    void read(uint64_t addr, ptx_thread_info* thread, unsigned warp_size) {
        write(addr, thread, false, warp_size);
    }

    void exit_thr(ptx_thread_info* thread, unsigned warp_size) {
        std::vector<int> ftid = getTID(thread, warp_size);
        //Unlock
        printf("tool: cta: %d is releasing locks after thread exit: ", ftid[3]);
        for (std::mutex* L: threadToLock[ftid]) {
            printf("%x ", L);
            L->unlock();
        }
        printf("\n");
        threadToLock[ftid].clear();
    }
}
