clc; clear;
mex -v CXXFLAGS="\$CXXFLAGS -O3 -DNOBLAS" solve_assign.cc csa.cc kofn.cc match.cc Exception.cc Matrix.cc Random.cc String.cc Timer.cc