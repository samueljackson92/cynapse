set(INC_FILES
    BatchImageLoader.h
)

set(SRC_FILES
    BatchImageLoader.cpp
)

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem )
include_directories( SYSTEM ${Boost_INCLUDE_DIR} )

# Build the Cynapse library
add_library(${CYNAPSE_TOOLS_LIBRARY_NAME} STATIC ${INCLUDE_FILES} ${SRC_FILES})
target_link_libraries(${CYNAPSE_TOOLS_LIBRARY_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CYNAPSE_LIBRARY_NAME} ${EIGEN3_LIBRARIES})
set_property(TARGET ${CYNAPSE_TOOLS_LIBRARY_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${CYNAPSE_TOOLS_LIBRARY_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
