clc; clear;
mex -v CXXFLAGS="\$CXXFLAGS -O3 -DNOBLAS" solve_assign_pair.cc csa.cc kofn.cc match_mrf.cc Exception.cc Matrix.cc Random.cc String.cc Timer.cc