cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(RoADPro)

#set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREADED ON)
#ADD_DEFINITIONS(-DBoost_USE_STATIC_LIBS=ON)
#set(Boost_INCLUDE_DIR /usr/include/)
#set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


#find_package(Boost REQUIRED)

find_package(PCL)
find_package(OpenCV)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#add_executable(FeatureExtraction hyperspectral_reader.cpp)


add_executable(merge src/merge_pcd.cpp)
add_executable(measure src/measure_3d_holistic.cpp)
add_executable(leaf src/leaf_segment.cpp)


target_link_libraries(merge ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(measure ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(leaf ${PCL_LIBRARIES} ${OpenCV_LIBS})




