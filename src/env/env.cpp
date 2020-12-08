/**
 * @file
 * @brief This file implements the handling of environment variables
 * @copyright Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "env.hpp"

namespace {
	/* file constants */
	/**
	 * @brief default requested memory size (if environment variable is unset)
	 * @see @ref ARGO_MEMORY_SIZE
	 */
	const std::size_t default_memory_size = 8ul*(1ul<<30); // default: 8GB

	/**
	 * @brief default requested cache size (if environment variable is unset)
	 * @see @ref ARGO_CACHE_SIZE
	 */
	const std::size_t default_cache_size = 1ul<<30; // default: 1GB

	/**
	 * @brief default requested memory policy (if environment variable is unset)
	 * @see @ref ARGO_MEMORY_POLICY
	 */
	const std::size_t default_memory_policy = 0ul; // default: Bind-All

	/**
	 * @brief default requested page size (if environment variable is unset)
	 * @see @ref ARGO_BLOCK_SIZE
	 */
	const std::size_t default_page_size = 16ul; // default: 64KB

	/**
	 * @brief environment variable used for requesting memory size
	 * @see @ref ARGO_MEMORY_SIZE
	 */
	const std::string env_memory_size = "ARGO_MEMORY_SIZE";

	/**
	 * @brief environment variable used for requesting cache size
	 * @see @ref ARGO_CACHE_SIZE
	 */
	const std::string env_cache_size = "ARGO_CACHE_SIZE";

	/**
	 * @brief environment variable used for choosing memory policy
	 * @see @ref ARGO_MEMORY_POLICY
	 */
	const std::string env_memory_policy = "ARGO_MEMORY_POLICY";

	/**
	 * @brief environment variable used for choosing page size
	 * @see @ref ARGO_BLOCK_SIZE
	 */
	const std::string env_page_size = "ARGO_BLOCK_SIZE";

	/** @brief error message string */
	const std::string msg_uninitialized = "argo::env::init() must be called before accessing environment values";
	/** @brief error message string */
	const std::string msg_illegal_format = "An environment variable could not be converted to a number: ";
	/** @brief error message string */
	const std::string msg_out_of_range = "An environment variable contains a number outside the possible range: ";

	/* file variables */
	/**
	 * @brief memory size requested through the environment variable @ref ARGO_MEMORY_SIZE
	 */
	std::size_t value_memory_size;

	/**
	 * @brief cache size requested through the environment variable @ref ARGO_CACHE_SIZE
	 */
	std::size_t value_cache_size;

	/**
	 * @brief memory policy requested through the environment variable @ref ARGO_MEMORY_POLICY
	 */
	std::size_t value_memory_policy;

	/**
	 * @brief page size requested through the environment variable @ref ARGO_BLOCK_SIZE
	 */
	std::size_t value_page_size;

	/** @brief flag to allow checking that environment variables have been read before accessing their values */
	bool is_initialized = false;

	/* helper functions */
	/** @brief throw an exception if argo::env::init() has not yet been called */
	void assert_initialized() {
		if(!is_initialized) {
			throw std::logic_error(msg_uninitialized);
		}
	}

	/**
	 * @brief parse an environment variable
	 * @param name the environment variable to parse
	 * @param fallback the default value to use if the environment variable is undefined
	 * @return a pair <env_used, value>, where env_used is true iff the environment variable is set,
	 *         and value is either the value of the environment variable or the fallback value.
	 */
	std::pair<bool, std::size_t> parse_env(std::string name, std::size_t fallback) {
		auto env_value = std::getenv(name.c_str());
		try {
			if(env_value != nullptr) {
				return std::make_pair(true, std::stoul(env_value));
			} else {
				return std::make_pair(false, fallback);
			}
		} catch (const std::invalid_argument& e) {
			// environment variable exists, but value is not convertable to an unsigned long
			std::cerr << msg_illegal_format << name << std::endl;
			throw;
		} catch (const std::out_of_range& e) {
			// environment variable exists, but value is out of range
			std::cerr << msg_out_of_range << name << std::endl;
			throw;
		}
	}

} // unnamed namespace

namespace argo {
	namespace env {
		void init() {
			value_memory_size = parse_env(env_memory_size, default_memory_size).second;
			value_cache_size = parse_env(env_cache_size, default_cache_size).second;
			
			value_memory_policy = parse_env(env_memory_policy, default_memory_policy).second;
			value_page_size = parse_env(env_page_size, default_page_size).second;

			is_initialized = true;
		}

		std::size_t memory_size() {
			assert_initialized();
			return value_memory_size;
		}

		std::size_t cache_size() {
			assert_initialized();
			return value_cache_size;
		}

		std::size_t memory_policy() {
			assert_initialized();
			return value_memory_policy;
		}

		std::size_t page_size() {
			assert_initialized();
			return value_page_size;
		}

	} // namespace env
} // namespace argo
