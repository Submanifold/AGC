#pragma once
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

namespace Eigen {
    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    class Matrix;
    using MatrixXd = Matrix<double, -1, -1, 0, -1, -1>;
    using MatrixXi = Matrix<int, -1, -1, 0, -1, -1>;
    using Vector4d = Matrix<double, 4, 1, 0, 4, 1>;
    using Vector3d = Matrix<double, 3, 1, 0, 3, 1>;
    using Matrix3D = Matrix<double, 3, 3, 0, 3, 3>;
}

#include <vector>


class VariationalShapeDeformation3D {
public:
    Eigen::MatrixXd C_;  // Cage vertices (n x 3)
    Eigen::MatrixXi CF_; // Cage faces (m x 3)  
    Eigen::MatrixXd A_;  // Anisotropic tensor (3x3)
    Eigen::MatrixXd B_;  // A^{1/2}
    Eigen::MatrixXd Binv_; // A^{-1/2}
    double epsilon_ = 1e-6;
    Eigen::Vector3d unit(const Eigen::Vector3d& x, double eps = 1e-15) const;
    Eigen::Matrix3d skew(const Eigen::Vector3d& u) const;
    double omega_triangle(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2,
        const Eigen::Vector3d& v3, const Eigen::Vector3d& eta) const;

    struct CiGradResult;
    CiGradResult compute_Ci_gradCi(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2,
        const Eigen::Vector3d& v3, const Eigen::Vector3d& eta) const;


    struct TriangleResult;
    TriangleResult compute_triangle_derivatives_closed_form(const Eigen::Vector3d& v1,
        const Eigen::Vector3d& v2,
        const Eigen::Vector3d& v3,
        const Eigen::Vector3d& eta,
        int compute_phi_for_vertex = 0) const;
public:
    VariationalShapeDeformation3D(const Eigen::MatrixXd& C, const Eigen::MatrixXi& CF, const Eigen::MatrixXd& A);
    void debugCompareNumericalAndClosedForm(const std::vector<Eigen::Vector3d>& test_points);
    void computeGreenCoordinatesAtPoint(const Eigen::Vector3d& p, Eigen::VectorXd& phi, Eigen::VectorXd& psi);
    void computeGreenCoordinateGradientsAtPointNumerical(const Eigen::Vector3d& p, Eigen::MatrixXd& grad_phi, Eigen::MatrixXd& grad_psi);
    void computeGreenCoordinateGradientsAtPointClose(const Eigen::Vector3d& p, Eigen::MatrixXd& grad_phi, Eigen::MatrixXd& grad_psi);
    void computeGreenCoordinateHessiansAtPointNumerical(const Eigen::Vector3d& p, std::vector<Eigen::Matrix3d>& hess_phi, std::vector<Eigen::Matrix3d>& hess_psi);
    void computeGreenCoordinateHessiansAtPointClose(const Eigen::Vector3d& p, std::vector<Eigen::Matrix3d>& hess_phi, std::vector<Eigen::Matrix3d>& hess_psi);
    Eigen::Vector3d deformPoint(const Eigen::Vector3d& point, const Eigen::MatrixXd& deformed_cage_vertices, const Eigen::MatrixXd& deformed_face_coeffs);

   
    const Eigen::MatrixXd& getCageVertices() const { return C_; }
    const Eigen::MatrixXi& getCageFaces() const { return CF_; }

    
    void computePhiAndPsiForOneTriangle(const Eigen::Vector3d& eta, Eigen::Vector3d* tri_vertices, double* phi, double& psi);
    void AnisotropicIntegralFundamentalSolution(Eigen::Vector3d* tri_vertices, const Eigen::Vector3d& eta, double& psi, Eigen::Vector3d& side_integral, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Binv);
    void IntegralFundamentalSolution(Eigen::Vector3d* tri_vertices, const Eigen::Vector3d& eta, double& psi, Eigen::Vector3d& side_integral);
    double get_signed_solid_angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c);
};

class VariationalDeformationSolver3D {
public:
    VariationalShapeDeformation3D& deformation_;

    double lambda_ = 0.05;
    double lambda_user_ = 100.0;
    double lambda_change_ = 0.001;
    double lambda_rigid_ = 1.0;

    int max_iter_ = 100;
    double tol_ = 1e-4;

    Eigen::MatrixXd final_a_;
    Eigen::MatrixXd final_b_;

public:
    VariationalDeformationSolver3D(
        VariationalShapeDeformation3D& deformation);

    void solve(
        const Eigen::MatrixXd& user_constraints_origin,
        const Eigen::MatrixXd& user_constraints_deformed,
        const Eigen::MatrixXd& medial_points,
        const Eigen::MatrixXd& boundary_samples,
        Eigen::MatrixXd& A);

    Eigen::MatrixXd getDeformedCageVertices() const
    {
        return final_a_.transpose();
    }

    Eigen::MatrixXd getDeformedFaceCoeffs() const
    {
        return final_b_.transpose();
    }


    std::vector<Eigen::Matrix3d> solveLocalStepClose(
        const Eigen::MatrixXd& a,
        const Eigen::MatrixXd& b,
        const Eigen::MatrixXd& anchors);

    Eigen::Matrix3d computeJacobianAtPointNumerical(
        const Eigen::Vector3d& p,
        const Eigen::MatrixXd& a,
        const Eigen::MatrixXd& b);

    Eigen::Matrix3d computeJacobianAtPointClose(
        const Eigen::Vector3d& p,
        const Eigen::MatrixXd& a,
        const Eigen::MatrixXd& b);

    void solveGlobalStepClose(
        Eigen::MatrixXd& a,
        Eigen::MatrixXd& b,
        const std::vector<Eigen::Matrix3d>& R,
        const Eigen::MatrixXd& user_constraints_origin,
        const Eigen::MatrixXd& user_constraints_deformed,
        const Eigen::MatrixXd& anchors,
        const Eigen::MatrixXd& boundary_samples,
        const Eigen::MatrixXd& a0,
        const Eigen::MatrixXd& b0);

    

    void buildLinearSystemForDimensionClose(
        int dim,
        const Eigen::MatrixXd& a,
        const Eigen::MatrixXd& b,
        const std::vector<Eigen::Matrix3d>& R,
        const Eigen::MatrixXd& user_constraints_origin,
        const Eigen::MatrixXd& user_constraints_deformed,
        const Eigen::MatrixXd& anchors,
        const Eigen::MatrixXd& boundary_samples,
        Eigen::MatrixXd& A_ls,
        Eigen::VectorXd& B_ls,
        Eigen::MatrixXd& C_constraints,
        Eigen::VectorXd& d_constraints);

    Eigen::VectorXd solveConstrainedLeastSquares(
        const Eigen::MatrixXd& A_ls,
        const Eigen::VectorXd& B_ls,
        const Eigen::MatrixXd& C_constraints,
        const Eigen::VectorXd& d_constraints,
        const Eigen::VectorXd& x0);

    Eigen::VectorXd solveLeastSquares(
        const Eigen::MatrixXd& A_ls,
        const Eigen::VectorXd& B_ls,
        const Eigen::VectorXd& x0);
};

struct VariationalShapeDeformation3D::CiGradResult {
    Eigen::Vector3d C;
    Eigen::Matrix3d gradC;
    Eigen::Vector3d Cbar;
    Eigen::Matrix3d d;
    Eigen::Matrix3d e;
    Eigen::Matrix3d e_hat;
    Eigen::Vector3d R;
};

struct VariationalShapeDeformation3D::TriangleResult {
    double psi;
    double phi[3];
    Eigen::Vector3d grad_psi;
    Eigen::Vector3d grad_phi[3];
    Eigen::Matrix3d hess_psi;
    Eigen::Matrix3d hess_phi[3];
};


void calcNormals(const Eigen::MatrixXd& C, const Eigen::MatrixXi& CF, Eigen::MatrixXd& normals);


void calculateAGC(const Eigen::MatrixXd& C, const Eigen::MatrixXi& CF, const Eigen::MatrixXd& normals, const Eigen::MatrixXd& eta_m,
    Eigen::MatrixXd& phi, Eigen::MatrixXd& psi, Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& Binv);

void calcScalingFactors(const Eigen::MatrixXd& C, const Eigen::MatrixXd& C_deformed, const Eigen::MatrixXi& CF, Eigen::MatrixXd& normals);

