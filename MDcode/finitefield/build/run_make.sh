cmake -D CMAKE_C_COMPILER=$(which icx) -D CMAKE_CXX_COMPILER=$(which icpx) ..
cmake --build . --config Release 
#cmake --build . --config Debug
