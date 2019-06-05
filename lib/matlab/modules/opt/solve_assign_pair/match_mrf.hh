
#ifndef __match_mrf_hh__
#define __match_mrf_hh__

class Matrix;

// solves the assignment problem with pairwise costs
double solveAssignMRF(const Matrix& maskMap, const Matrix& gtMap, const Matrix& edgeCost, const Matrix& bdryTan, 
                      const Matrix& preMatch, double sigmaX, double sigmaY, double maxSpatialCost, double outlierCost, 
                      double neighSize, double wPairwise, Matrix& match1, Matrix& match2);

#endif // __match_mrf_hh__
