# Copyright (C) Eta Scale AB. Licensed under the Eta Scale Open Source License. See the LICENSE file for details.

add_library(argobackend-singlenode SHARED singlenode.cpp)

install(TARGETS argobackend-singlenode
	COMPONENT "Runtime"
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
