/**
 * @file
 * @brief This file implements the virtual memory and virtual address handling
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>
#include "virtual_memory.hpp"

namespace {
	/* file constants */
	/** @todo hardcoded start address */
	const char* ARGO_START = (char*) 0x200000000000l;
	/** @todo hardcoded end address */
	const char* ARGO_END   = (char*) 0x600000000000l;
	/** @todo hardcoded size */
	const ptrdiff_t ARGO_SIZE = ARGO_END - ARGO_START;
	/** @todo hardcoded maximum size */
	const ptrdiff_t ARGO_SIZE_LIMIT = 0x80000000000l;

	/** @brief error message string */
	const std::string msg_alloc_fail = "ArgoDSM could not allocate mappable memory";
	/** @brief error message string */
	const std::string msg_mmap_fail = "ArgoDSM failed to map in virtual address space.";
	/** @brief error message string */
	const std::string msg_main_mmap_fail = "ArgoDSM failed to set up virtual memory. Please report a bug.";

	/* file variables */
	/** @brief a file descriptor for backing the virtual address space used by ArgoDSM */
	int fd;
	/** @brief the address at which the virtual address space used by ArgoDSM starts */
	void* start_addr;
	/** @brief the size of the ArgoDSM virtual address space */
	std::size_t avail;
}

namespace argo {
	namespace virtual_memory {
		void init() {
			/* find maximum filesize */
			struct statvfs b;
			/*statvfs*/
			/**
			 * @brief Returns information about the mounted file system.
			 * @param path the pathname of any file within the mounted file system.
			 * @param buf is a pointer to a statvfs structure.
			 * @note /dev/shm is a file system, which keeps all files in virtual memory. Everything is temporary
			 * in the sense that no files will be created on the hard drive.
			 */
			statvfs("/dev/shm", &b);
			// unsigned long f_bsize - Filesystem block size.
			// fsblkcnt_t f_bavail   - Number of free blocks for unpriviledged users.
			avail = b.f_bavail * b.f_bsize;
			if(avail > static_cast<unsigned long>(ARGO_SIZE_LIMIT)) {
				avail = ARGO_SIZE_LIMIT;
			}
			std::string filename = "/argocache" + std::to_string(getpid());
			/*shm_open*/
			/**
			 * @brief POSIX shared memory is organized using memory-mapped files, which associate the region of 
			 * shared memory with a file. A process must first create a shared-memory object.
			 * @param name the first parameter specifies the name of the shared-memory object. Processes that 
			 * wish to access this shared memory must refer to the object by this name.
			 * @param oflag the subsequent parameters specify that the shared-memory object is to be created if
			 * it does not yet exist (O_CREAT) and that the object is open for reading and writing (O_RDWR).
			 * @param mode the last parameter establishes the file-access permissions of the shared-memory object.
			 */
			fd = shm_open(filename.c_str(), O_RDWR|O_CREAT, 0644);

			// The file descriptor (fd), i.e. 3 in this case, is the index into the process-specific file descriptor
			// table, not the open file table. The file descriptor entry itself contains an index to an entry in the 
			// kernel's global open file table, as well as file descriptor flags.
			//printf("Process getpid(): %i, filename: %s, fd: %i\n", getpid(), filename.c_str(), fd);

			/*shm_unlink*/
			/**
			 * @brief Removes the shared-memory segment previously created by shm_open().
			 * @returns returns 0 on success, or -1 on error.
			 * @note Maybe we unlink here because of the anonymous mapping below. We don't need to assosiate the
			 * region of shared memory with a file. Take note, that this doesn't affect the memory mapping.
			 * shm_unlink() just removes a POSIX shared memory segment from the shm filesystem and only if the
			 * last mapping is removed, the actual memory is destroyed.
			 * shm_unlink() man page: "it removes a shared memory object name, and, once all processes have unmapped
			 * the object, de-allocates and destroys the contents of the associated memory region."
			 */
			if(shm_unlink(filename.c_str())) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
			/*ftruncate*/
			/**
			 * @brief Used to configure the size of the object in bytes.
			 * @returns 0 on success, or -1 on error.
			 */
			if(ftruncate(fd, avail)) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
			/** @todo check desired range is free */
			/**
			 * @param MAP_ANONYMOUS The mapping in not backed by any file; its contents are initialized to zero.
			 * @param MAP_SHARED Share this mapping. Updates to the mapping are visible to other processes that
			 * map this file, and are carried through to the underlying file.
			 * @param MAP_FIXED Don't interpret addr as a hint: place the mapping at exactly that address. addr
			 * must be a multiple of the page size.
			 */
			constexpr int flags = MAP_ANONYMOUS|MAP_SHARED|MAP_FIXED;
			/*mmap*/
			/**
			 * @brief Establishes a memory-mapped file containing the shared-memory object.
			 * @param addr the address to map to.
			 * @param size the size of the mapping.
			 * @param offset the offset into the backing memory.
			 * @param prot protection flags for the mapping (PROT_NONE Pages may not be accessed).
			 * @returns a pointer to the memory-mapped file that is used for accessing the shared-memory object.
			 * @note Anonymous mappings are not backed by a file, so the "fd" and "offset" are not even used
			 * when the MAP_ANONYMOUS flag is specified in mmap().
			 */
			start_addr = ::mmap((void*)ARGO_START, avail, PROT_NONE, flags, -1, 0);
			if(start_addr == MAP_FAILED) {
				std::cerr << msg_main_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_main_mmap_fail);
				exit(EXIT_FAILURE);
			}
		}

		void* start_address() {
			return start_addr;
		}

		std::size_t size() {
			return avail;
		}

		/*allocate_mappable*/
		/**
		 * @brief Allocates memory that can be mapped into the ArgoDSM virtual address space.
		 * @param alignment the alignment of the allocation.
		 * @param size size of the allocation.
		 * @return a pointer to the new memory allocation.
		 * @details this will allocate memory that is guaranteed to work with map_memory(). Any memory allocated
		 * through other means may not be possible to map into the visible ArgoDSM virtual memory space later.
		 */
		void* allocate_mappable(std::size_t alignment, std::size_t size) {
			void* p;
			auto r = posix_memalign(&p, alignment, size);
			if(r || p == nullptr) {
				std::cerr << msg_alloc_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(r)), msg_alloc_fail);
				return nullptr;
			}
			return p;
		}

		/*map_memory*/
		/**
		 * @brief Maps memory into ArgoDSM virtual address space.
		 * @param addr the address to map to.
		 * @param size the size of the mapping.
		 * @param offset the offset into the backing memory.
		 * @param prot protection flags for the mapping.
		 */
		void map_memory(void* addr, std::size_t size, std::size_t offset, int prot) {
			auto p = ::mmap(addr, size, prot, MAP_SHARED|MAP_FIXED, fd, offset);
			if(p == MAP_FAILED) {
				std::cerr << msg_mmap_fail << std::endl;
				throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)), msg_mmap_fail);
				exit(EXIT_FAILURE);
			}
		}
	} // namespace virtual_memory
} // namespace argo
