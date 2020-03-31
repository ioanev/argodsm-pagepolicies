#include "data_distribution.hpp"
#include "backend/mpi/swdsm.h"
#include <cstdlib>
#include <mpi.h>

/** @brief Page size local to this file for the implementations */
static constexpr std::size_t granularity = 0x1000UL;

/*Policies*/
/** @brief  Holds the owner of a page */
extern std::size_t *globalOwners;
/** @brief  Allocator offset for the node */
extern std::size_t ownerOffset;
/** @brief  MPI window for communicating owner directory */
extern MPI_Win ownerWindow;
/** @brief  Protects the owner directory */
extern pthread_mutex_t ownermutex;
/** @brief  More useful variables */
extern int workrank;

namespace argo {
	namespace data_distribution {
        template<>
        std::size_t naive_data_distribution<0>::firstTouch (const std::size_t& addr) {
            node_id_t homenode;
            const std::size_t id = 1 << getID();
            const std::size_t index = 2 * (addr / granularity);

            pthread_mutex_lock(&ownermutex);
            //sem_wait(&ibsem);

            //printf("(pr_bef) Process: %i, Addr: 0x%X, [0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X]\n", workrank, addr,
            //		globalOwners[0], globalOwners[1], globalOwners[2], globalOwners[3], globalOwners[4], globalOwners[5],
            //		globalOwners[6], globalOwners[7], globalOwners[8], globalOwners[9], globalOwners[10], globalOwners[11]);
            //fflush(stdout);

            // Variables for CAS.
            std::size_t result;
            constexpr std::size_t compare = 0;
            
            // Check/try to acquire ownership of the page.
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, ownerWindow);
            // CAS to process' 0 index.
            MPI_Compare_and_swap(&id, &compare, &result, MPI_LONG, 0, index, ownerWindow);
            // Force local and remote completion with MPI_Win_unlock().
            MPI_Win_unlock(0, ownerWindow);

            // This process was the first one to deposit the id.
            if (result == 0) {
                homenode = id;

                // Mark the page in the local window.
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, workrank, 0, ownerWindow);
                globalOwners[index] |= id; // OR operation for debugging...
                globalOwners[index+1] |= ownerOffset; // OR operation for debugging...
                MPI_Win_unlock(workrank, ownerWindow);

                // Mark the page in the public windows.
                int n;
                for(n = 0; n < nodes; n++)
                    if (n != workrank) {
                        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, n, 0, ownerWindow);
                        MPI_Accumulate(&id, 1, MPI_LONG, n, index, 1, MPI_LONG, MPI_BOR, ownerWindow); // MPI_BOR operation for debugging...
                        MPI_Accumulate(&ownerOffset, 1, MPI_LONG, n, index+1, 1, MPI_LONG, MPI_BOR, ownerWindow); // MPI_BOR operation for debugging...
                        MPI_Win_unlock(n, ownerWindow);
                    }
                
                // Since a new page was acquired increase the homenode offset.
                ownerOffset += granularity;
            } else
                homenode = result;
            
            //printf("(pr_aft) Process: %i, Addr: 0x%X, [0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X, 0x%X]\n", workrank, addr,
            //		globalOwners[0], globalOwners[1], globalOwners[2], globalOwners[3], globalOwners[4], globalOwners[5],
            //		globalOwners[6], globalOwners[7], globalOwners[8], globalOwners[9], globalOwners[10], globalOwners[11]);
            //fflush(stdout);
            
            //sem_post(&ibsem);
            pthread_mutex_unlock(&ownermutex);

            return homenode;
        }

        template<>
        node_id_t naive_data_distribution<0>::homenode (char* const ptr) {
            #if   ARGO_MEM_ALLOC_POLICY == 0
                const std::size_t addr = ptr - start_address;
                const node_id_t homenode = addr / size_per_node;
                if(homenode >=nodes){
                    exit(EXIT_FAILURE);
                }
                return homenode;
            #elif ARGO_MEM_ALLOC_POLICY == 1
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / granularity;
                const node_id_t homenode = pagenum % nodes;
                if(homenode >=nodes){
                    exit(EXIT_FAILURE);
                }
                return homenode;
            #elif ARGO_MEM_ALLOC_POLICY == 2
                static const std::size_t pageblock = PAGE_BLOCK * granularity;
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / pageblock;
                const node_id_t homenode = pagenum % nodes;
                if(homenode >=nodes){
                    exit(EXIT_FAILURE);
                }
                return homenode;
            #elif ARGO_MEM_ALLOC_POLICY == 3
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / granularity;
                const node_id_t homenode = (pagenum + pagenum / nodes) % nodes;
                if(homenode >=nodes){
                    exit(EXIT_FAILURE);
                }
                return homenode;
            #elif ARGO_MEM_ALLOC_POLICY == 4
                static const std::size_t pageblock = PAGE_BLOCK * granularity;
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / pageblock;
                const node_id_t homenode = (pagenum + pagenum / nodes) % nodes;
                if(homenode >=nodes){
                    exit(EXIT_FAILURE);
                }
                return homenode;
            #elif ARGO_MEM_ALLOC_POLICY == 5
                static const std::size_t prime = (3 * nodes) / 2;
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / granularity;
                const node_id_t homenode = ((pagenum % prime) >= nodes)
                ? ((pagenum / prime) * (prime - nodes) + ((pagenum % prime) - nodes)) % nodes
                : pagenum % prime;
                if(homenode >=nodes){
                    exit(EXIT_FAILURE);
                }
                return homenode;
            #elif ARGO_MEM_ALLOC_POLICY == 6
                static const std::size_t pageblock = PAGE_BLOCK * granularity;
                static const std::size_t prime = (3 * nodes) / 2;
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / pageblock;
                const node_id_t homenode = ((pagenum % prime) >= nodes)
                ? ((pagenum / prime) * (prime - nodes) + ((pagenum % prime) - nodes)) % nodes
                : pagenum % prime;
                if(homenode >=nodes){
                    exit(EXIT_FAILURE);
                }
                return homenode;
            #elif ARGO_MEM_ALLOC_POLICY == 7
                const std::size_t addr = ptr - start_address;
                const std::size_t index = 2 * (addr / granularity);
                node_id_t homenode = globalOwners[index];
                
                if (!homenode)
                    homenode = firstTouch(addr);

                int n;
                for(n = 0; n < nodes; n++)
                    if((1 << n) == homenode)
                        homenode = n;

                if(homenode >=nodes){
                    exit(EXIT_FAILURE);
                }
                return homenode;
            #endif
        }

        template<>
        std::size_t naive_data_distribution<0>::local_offset (char* const ptr) {
            #if   ARGO_MEM_ALLOC_POLICY == 0
                const std::size_t addr = ptr - start_address;
                const std::size_t offset = addr - (homenode(ptr)) * size_per_node;
                if(offset >=size_per_node){
                    exit(EXIT_FAILURE);
                }
                return offset;
            #elif ARGO_MEM_ALLOC_POLICY == 1
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / granularity;
                const std::size_t offset = (addr > 0 && homenode(ptr) == 0)
                ? pagenum / nodes * granularity + granularity
                : pagenum / nodes * granularity;
                if(offset >=size_per_node){
                    exit(EXIT_FAILURE);
                }
                return offset;
            #elif ARGO_MEM_ALLOC_POLICY == 2
                static const std::size_t pageblock = PAGE_BLOCK * granularity;
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / pageblock;
                const std::size_t offset = (addr > 0 && homenode(ptr) == 0)
                ? pagenum / nodes * pageblock + lessaddr % pageblock + granularity
                : pagenum / nodes * pageblock + lessaddr % pageblock;
                if(offset >=size_per_node){
                    exit(EXIT_FAILURE);
                }
                return offset;
            #elif ARGO_MEM_ALLOC_POLICY == 3
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / granularity;
                const std::size_t offset = (addr > 0 && homenode(ptr) == 0)
                ? pagenum / nodes * granularity + granularity
                : pagenum / nodes * granularity;
                if(offset >=size_per_node){
                    exit(EXIT_FAILURE);
                }
                return offset;
            #elif ARGO_MEM_ALLOC_POLICY == 4
                static const std::size_t pageblock = PAGE_BLOCK * granularity;
                const std::size_t addr = ptr - start_address;
                const std::size_t lessaddr = (addr >= granularity) ? addr - granularity : 0;
                const std::size_t pagenum = lessaddr / pageblock;
                const std::size_t offset = (addr > 0 && homenode(ptr) == 0)
                ? pagenum / nodes * pageblock + lessaddr % pageblock + granularity
                : pagenum / nodes * pageblock + lessaddr % pageblock;
                if(offset >=size_per_node){
                    exit(EXIT_FAILURE);
                }
                return offset;
            #elif ARGO_MEM_ALLOC_POLICY == 5
                static const std::size_t prime = (3 * nodes) / 2;
                std::size_t addr = ptr - start_address;
                std::size_t offset, lessaddr = (addr >= granularity) ? addr - granularity : 0;
                std::size_t pagenum = lessaddr / granularity;
                if ((addr <= (nodes * granularity)) || ((pagenum % prime) >= nodes))
                    offset = (pagenum / nodes) * granularity + (addr > 0 && !homenode(ptr)) * granularity;
                else {
                    node_id_t currhome;
                    std::size_t homecounter = 0;
                    const node_id_t realhome = homenode(ptr);
                    for (addr -= granularity; ; addr -= granularity) {
                        lessaddr = addr - granularity;
                        pagenum = lessaddr / granularity;
                        currhome = homenode(static_cast<char*>(start_address) + addr);
                        homecounter += (currhome == realhome) ? 1 : 0;
                        if (((addr <= (nodes * granularity)) && (currhome == realhome)) ||
                            (((pagenum % prime) >= nodes) && (currhome == realhome))) {
                            offset = (pagenum / nodes) * granularity + !realhome * granularity;
                            offset += homecounter * granularity;
                            break;
                        }
                    }
                }
                if(offset >=size_per_node){
                    exit(EXIT_FAILURE);
                }
                return offset;
            #elif ARGO_MEM_ALLOC_POLICY == 6
                static const std::size_t pageblock = PAGE_BLOCK * granularity;
                static const std::size_t prime = (3 * nodes) / 2;
                std::size_t addr = ptr - start_address;
                std::size_t offset, lessaddr = (addr >= granularity) ? addr - granularity : 0;
                std::size_t pagenum = lessaddr / pageblock;
                if ((addr <= (nodes * pageblock)) || ((pagenum % prime) >= nodes))
                    offset = (pagenum / nodes) * pageblock + lessaddr % pageblock + (addr > 0 && !homenode(ptr)) * granularity;
                else {
                    node_id_t currhome;
                    std::size_t homecounter = 0;
                    const node_id_t realhome = homenode(ptr);
                    for (addr -= pageblock; ; addr -= pageblock) {
                        lessaddr = addr - granularity;
                        pagenum = lessaddr / pageblock;
                        currhome = homenode(static_cast<char*>(start_address) + addr);
                        homecounter += (currhome == realhome) ? 1 : 0;
                        if (((addr <= (nodes * pageblock)) && (currhome == realhome)) || 
                            (((pagenum % prime) >= nodes) && (currhome == realhome))) {
                            offset = (pagenum / nodes) * pageblock + lessaddr % pageblock + !realhome * granularity;
                            offset += homecounter * pageblock;
                            break;
                        }
                    }
                }
                if(offset >=size_per_node){
                    exit(EXIT_FAILURE);
                }
                return offset;
            #elif ARGO_MEM_ALLOC_POLICY == 7
                const std::size_t addr = ptr - start_address;
                const std::size_t index = 2 * (addr / granularity);
                const std::size_t offset = globalOwners[index + 1];
                if(offset >=size_per_node){
                    exit(EXIT_FAILURE);
                }
                return offset;
            #endif
        }
    }
}