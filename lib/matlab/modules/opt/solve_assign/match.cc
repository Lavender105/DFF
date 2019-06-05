#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include "csa.hh"
#include "kofn.hh"
#include "Point.hh"
#include "Matrix.hh"
#include "Array.hh"
#include "match.hh"
#include "Timer.hh"
#include "mex.h"
#define assert( isOK ) ( (isOK) ? (void)0 : (void) mexErrMsgTxt("assert failed\n") )
#define assertMsg( isOK, msg ) ( (isOK) ? (void)0 : (void) mexErrMsgTxt( msg "\n") )

struct Edge {
    int i,j;	// node ids, 0-based
    double w;	// distance between pixels
};

// CSA code needs integer weights.  Use this multiplier to convert
// floating-point weights to integers.
static const int multiplier = 100;

// The degree of outlier connections.
static const int degree = 6;

double 
solveAssign(const Matrix& maskMap, const Matrix& gtMap, const Matrix& edgeCost, const Matrix& bdryTan, 
            double sigmaX, double sigmaY, double maxSpatialCost, double outlierCost, Matrix& m1, Matrix& m2)// Modified by Zhiding
{
    // Check global constants.
    assertMsg (degree > 0, "degree must be larger than 0!");
    assertMsg (multiplier > 0, "multiplier must be larger than 0!");

    // Check arguments.
    assertMsg(gtMap.nrows() == maskMap.nrows() && 
              gtMap.nrows() == edgeCost.nrows() && 
              gtMap.nrows() == bdryTan.nrows(), "Row numbers of input matrices must be the same!");
    assertMsg(gtMap.ncols() == maskMap.ncols() &&
              gtMap.ncols() == edgeCost.ncols() && 
              gtMap.ncols() == bdryTan.ncols(), "Column numbers of input matrices must be the same!");
    
    const int height = gtMap.nrows();
    const int width = gtMap.ncols();
    const double maxSigma = std::max(sigmaX, sigmaY);

    // Initialize candidate edge map to zeros
    Matrix candMap = Matrix(height,width);
    Matrix precA = Matrix(height,width);
    Matrix precB = Matrix(height,width);
    Matrix precC = Matrix(height,width);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            precA(y, x) = pow(cos(bdryTan(y, x)), 2.0)/(2*pow(sigmaX, 2.0)) + pow(sin(bdryTan(y, x)), 2.0)/(2*pow(sigmaY, 2.0));
            precB(y, x) = sin(2*bdryTan(y, x))/(4*pow(sigmaX, 2.0)) - sin(2*bdryTan(y, x))/(4*pow(sigmaY, 2.0));
            precC(y, x) = pow(sin(bdryTan(y, x)), 2.0)/(2*pow(sigmaX, 2.0)) + pow(cos(bdryTan(y, x)), 2.0)/(2*pow(sigmaY, 2.0));
        }
    }

    // Define radius of search window.
    const int r = (int) ceil(maxSigma*sqrt(2*maxSpatialCost));

    // Figure out which pixels are matchable, i.e. within maxSpatialCost of gt edge
    // Count the number of nodes on each side of the match.
    // Construct nodeID->pixel and pixel->nodeID maps.
    // Node IDs range from [0,n1) and [0,n2).
    // Construct the list of edges between pixels within maxSpatialCost.
    int n1=0, n2=0;
    Array2D<bool> matchable1 (width,height);
    Array2D<bool> matchable2 (width,height);
    matchable1.init(false);
    matchable2.init(false);
    std::vector<Pixel> nodeToPix1;
    std::vector<Pixel> nodeToPix2;
    Array2D<int> pixToNode1 (width,height);
    Array2D<int> pixToNode2 (width,height);
    std::vector<Edge> edges;
    for (int x2 = 0; x2 < width; x2++) {
        for (int y2 = 0; y2 < height; y2++) {
            if (!gtMap(y2,x2)) { continue; }
            matchable2(x2,y2) = true;
            pixToNode2(x2,y2) = n2;
            Pixel pix2 (x2,y2);
            nodeToPix2.push_back(pix2);
            n2++;
            for (int u = -r; u <= r; u++) {
                const int x1 = x2 + u;
                if (x1 < 0 || x1 >= width) { continue; }
                for (int v = -r; v <= r; v++) {
                    const int y1 = y2 - v; // Flipped y direction
                    if (y1 < 0 || y1 >= height) { continue; }
                    if (!maskMap(y1,x1)) { continue; }
                    const double spatialCost = precA(y2,x2)*u*u + 2*precB(y2,x2)*u*v + precC(y2,x2)*v*v;
                    if (spatialCost > maxSpatialCost) { continue; }
                    if (!matchable1(x1,y1)) {
                        candMap(y1,x1) = true;
                        matchable1(x1,y1) = true;
                        pixToNode1(x1,y1) = n1;
                        Pixel pix1 (x1,y1);
                        nodeToPix1.push_back(pix1);
                        n1++;
                    }
                    Edge e;
                    e.i = pixToNode1(x1,y1);
                    e.j = pixToNode2(x2,y2);
                    e.w = spatialCost + edgeCost(y1,x1);// Modified by Zhiding
                    assertMsg (e.i >= 0 && e.i < n1, "Node1 index out of range!");
                    assertMsg (e.j >= 0 && e.j < n2, "Node2 index out of range!");
                    assertMsg (e.w < outlierCost, "Edge weight must be smaller than outlier cost!");
                    edges.push_back(e);
                }
            }
        }
    }

    // The cardinality of the match is n.
    const int n = n1 + n2;
    const int nmin = std::min(n1,n2);
    const int nmax = std::max(n1,n2);

    // Compute the degree of various outlier connections.
    const int d1 = std::max(0,std::min(degree,n1-1)); // from map1
    const int d2 = std::max(0,std::min(degree,n2-1)); // from map2
    const int d3 = std::min(degree,std::min(n1,n2)); // between outliers
    const int dmax = std::max(d1,std::max(d2,d3));

    assert (n1 == 0 || (d1 >= 0 && d1 < n1));
    assert (n2 == 0 || (d2 >= 0 && d2 < n2));
    assert (d3 >= 0 && d3 <= nmin);

    // Count the number of edges.
    int m = 0;
    m += edges.size(); 	// real connections
    m += d1 * n1;	// outlier connections
    m += d2 * n2;	// outlier connections
    m += d3 * nmax;	// outlier-outlier connections
    m += n; 		// high-cost perfect match overlay

    // If the graph is empty, then there's nothing to do.
    if (m == 0) {
        return 0;
    }

    // Weight of outlier connections.
    const int ow = (int) ceil (outlierCost * multiplier);

    // Scratch array for outlier edges.
    Array1D<int> outliers (dmax);

    // Construct the input graph for the assignment problem.
    Array2D<int> igraph (m,3);
    int count = 0;
    // real edges
    for (int a = 0; a < (int)edges.size(); a++) {
        int i = edges[a].i;
        int j = edges[a].j;
        assert (i >= 0 && i < n1);
        assert (j >= 0 && j < n2);
        igraph(count,0) = i;
        igraph(count,1) = j;
        igraph(count,2) = (int) rint (edges[a].w * multiplier);
        count++;
    }
    // outliers edges for map1, exclude diagonal
    for (int i = 0; i < n1; i++) {
        kOfN(d1,n1-1,outliers.data());
        for (int a = 0; a < d1; a++) {
            int j = outliers(a);
            if (j >= i) { j++; }
            assert (i != j);
            assert (j >= 0 && j < n1);
            igraph(count,0) = i;
            igraph(count,1) = n2 + j;
            igraph(count,2) = ow;
            count++;
        }
    }
    // outliers edges for map2, exclude diagonal
    for (int j = 0; j < n2; j++) {
        kOfN(d2,n2-1,outliers.data());
        for (int a = 0; a < d2; a++) {
            int i = outliers(a);
            if (i >= j) { i++; }
            assert (i != j);
            assert (i >= 0 && i < n2);
            igraph(count,0) = n1 + i;
            igraph(count,1) = j;
            igraph(count,2) = ow;
            count++;
        }
    }
    // outlier-to-outlier edges
    for (int i = 0; i < nmax; i++) {
        kOfN(d3,nmin,outliers.data());
        for (int a = 0; a < d3; a++) {
            const int j = outliers(a);
            assert (j >= 0 && j < nmin);
            if (n1 < n2) {
                assert (i >= 0 && i < n2);
                assert (j >= 0 && j < n1);
                igraph(count,0) = n1 + i;
                igraph(count,1) = n2 + j;
            } else {
                assert (i >= 0 && i < n1);
                assert (j >= 0 && j < n2);
                igraph(count,0) = n1 + j;
                igraph(count,1) = n2 + i;
            }
            igraph(count,2) = ow;
            count++;
        }
    }
    // perfect match overlay (diagonal)
    for (int i = 0; i < n1; i++) {
        igraph(count,0) = i;
        igraph(count,1) = n2 + i;
        igraph(count,2) = ow * multiplier;
        count++;
    }
    for (int i = 0; i < n2; i++) {
        igraph(count,0) = n1 + i;
        igraph(count,1) = i;
        igraph(count,2) = ow * multiplier;
        count++;
    }
    assert (count == m);

    // Check all the edges, and set the values up for CSA.
    for (int i = 0; i < m; i++) {
        assert(igraph(i,0) >= 0 && igraph(i,0) < n);
        assert(igraph(i,1) >= 0 && igraph(i,1) < n);
        igraph(i,0) += 1;
        igraph(i,1) += 1+n;
    }

    // Solve the assignment problem.
    CSA csa(2*n,m,igraph.data());
    assert(csa.edges()==n);

    Array2D<int> ograph (n,3);
    for (int i = 0; i < n; i++) {
        int a,b,c;
        csa.edge(i,a,b,c);
        ograph(i,0)=a-1; ograph(i,1)=b-1-n; ograph(i,2)=c;
    }

    // Check the solution.
    // Count the number of high-cost edges from the perfect match
    // overlay that were used in the match.
    int overlayCount = 0;
    for (int a = 0; a < n; a++) {
        const int i = ograph(a,0);
        const int j = ograph(a,1);
        const int c = ograph(a,2);
        assert (i >= 0 && i < n);
        assert (j >= 0 && j < n);
        assert (c >= 0);
        // edge from high-cost perfect match overlay
        if (c == ow * multiplier) { overlayCount++; }
        // skip outlier edges
        if (i >= n1) { continue; }
        if (j >= n2) { continue; }
        // for edges between real nodes, check the edge weight
        const Pixel pix1 = nodeToPix1[i];
        const Pixel pix2 = nodeToPix2[j];
        const int dx = pix1.x - pix2.x;
        const int dy = pix2.y - pix1.y; // Flipped y direction
		const double spatialCost = precA(pix2.y,pix2.x)*dx*dx + 2*precB(pix2.y,pix2.x)*dx*dy + precC(pix2.y,pix2.x)*dy*dy;
		const int w = (int)rint((spatialCost + edgeCost(pix1.y,pix1.x))*multiplier);// Modified by Zhiding
        assert (w == c);
    }

    // Print a warning if any of the edges from the perfect match overlay
    // were used.  This should happen rarely.  If it happens frequently,
    // then the outlier connectivity should be increased.
    if (overlayCount > 5) {
        fprintf (stderr, "%s:%d: WARNING: The match includes %d "
                 "outlier(s) from the perfect match overlay.\n",
                 __FILE__, __LINE__, overlayCount);
    }

    // Initialize to zeros.
    m1 = Matrix(height,width);
    m2 = Matrix(height,width);

    // Initialize match[12] arrays to (-1,-1).
    Array2D<Pixel> match1 (width,height);
    Array2D<Pixel> match2 (width,height);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            match1(x,y) = Pixel(-1,-1);
            match2(x,y) = Pixel(-1,-1);
        }
    }

    // Compute match arrays.
    for (int a = 0; a < n; a++) {
        // node ids
        const int i = ograph(a,0);
        const int j = ograph(a,1);
        // skip outlier edges
        if (i >= n1) { continue; }
        if (j >= n2) { continue; }
        // map node ids to pixels
        const Pixel pix1 = nodeToPix1[i];
        const Pixel pix2 = nodeToPix2[j];
        // record edges
        match1(pix1.x,pix1.y) = pix2;
        match2(pix2.x,pix2.y) = pix1;
    }
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (candMap(y,x)) {
                if (match1(x,y) != Pixel(-1,-1)) {
                    m1(y,x) = match1(x,y).x*height + match1(x,y).y + 1;
                }
            }
            if (gtMap(y,x)) {
                if (match2(x,y) != Pixel(-1,-1)) {
                    m2(y,x) = match2(x,y).x*height + match2(x,y).y + 1;
                }
            }
        }
    }

    // Compute the match cost.
    double cost = 0;
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (candMap(y,x)) {
                if (match1(x,y) == Pixel(-1,-1)) {
                    cost += outlierCost;
                } else {
                    const int x2 = match1(x,y).x;
                    const int y2 = match1(x,y).y;
                    const int dx = x - x2;
                    const int dy = y2 - y; // Flipped y direction
                    const double spatialCost = precA(y2,x2)*dx*dx + 2*precB(y2,x2)*dx*dy + precC(y2,x2)*dy*dy;
                    cost += spatialCost+edgeCost(y,x);
                }
            }
            if (gtMap(y,x)) {
                if (match2(x,y) == Pixel(-1,-1)) {
                    cost += outlierCost;
                } else {
                    const int x1 = match2(x,y).x;
                    const int y1 = match2(x,y).y;
                    const int dx = x1 - x;
                    const int dy = y - y1; // Flipped y direction
                    const double spatialCost = precA(y,x)*dx*dx + 2*precB(y,x)*dx*dy + precC(y,x)*dy*dy;
                    cost += spatialCost+edgeCost(y1,x1);
                }
            }
        }
    }    

    // Return the match cost.
    return cost;
}
