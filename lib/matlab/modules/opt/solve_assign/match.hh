
#ifndef __match_hh__
#define __match_hh__

class Matrix;

// solves the assignment problem
double solveAssign(const Matrix& maskMap, const Matrix& gtMap, const Matrix& edgeCost, const Matrix& pBdryTan, 
                   double sigmaX, double sigmaY, double maxSpatialCost, double outlierCost, Matrix& match1, Matrix& match2);

#endif // __match_hh__
