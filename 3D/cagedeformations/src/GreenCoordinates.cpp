#define _USE_MATH_DEFINES
#include <cmath>

#include <cagedeformations/GreenCoordinates.h>
#include <array>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>


template< class point_t >
double get_signed_solid_angle(point_t const& a, point_t const& b, point_t const& c) 
{
	typedef double    T;
	T det = a.dot(b.cross(c));
	if (fabs(det) < 0.0000000001)
		return 2.0 * M_PI;

	T al = a.norm(), bl = b.norm(), cl = c.norm();

	T div = al * bl * cl + a.dot(b) * cl + a.dot(c) * bl + b.dot(c) * al;
	T at = std::atan2(std::abs(det), div);
	if (at < 0) at += M_PI;
	T omega = 2.0 * at;

	if (det > 0.0) return omega;
	return -omega;
}


VariationalShapeDeformation3D::VariationalShapeDeformation3D(const Eigen::MatrixXd& C, const Eigen::MatrixXi& CF,
	const Eigen::MatrixXd& A)
	: C_(C), CF_(CF), A_(A) {

	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A);
	Eigen::MatrixXd D_sqrt = eigensolver.eigenvalues().cwiseSqrt().asDiagonal();
	B_ = eigensolver.eigenvectors() * D_sqrt * eigensolver.eigenvectors().transpose();
	Binv_ = B_.inverse();
}


void VariationalShapeDeformation3D::computeGreenCoordinatesAtPoint(const Eigen::Vector3d& p,
	Eigen::VectorXd& phi,  
	Eigen::VectorXd& psi) { 
	int n_vertices = C_.rows();
	int n_faces = CF_.rows();
	phi.resize(n_vertices);
	psi.resize(n_faces);
	phi.setZero();
	psi.setZero();
	for (int face_idx = 0; face_idx < n_faces; ++face_idx) {
		Eigen::Vector3i tri_indices = CF_.row(face_idx);
		Eigen::Vector3d tri_verts[3];
		double phi_values[3];
		double psi_value;
		for (int v = 0; v < 3; ++v) {
			tri_verts[v] = C_.row(tri_indices[v]);
		}
		computePhiAndPsiForOneTriangle(p, tri_verts, phi_values, psi_value);
		psi(face_idx) = psi_value;
		for (int v = 0; v < 3; ++v) {
			phi(tri_indices[v]) += phi_values[v];
		}
	}
}

Eigen::Vector3d VariationalShapeDeformation3D::unit(const Eigen::Vector3d& x, double eps) const {
	double n = x.norm();
	if (n < eps) throw std::runtime_error("Zero-length vector encountered.");
	return x / n;
}

Eigen::Matrix3d VariationalShapeDeformation3D::skew(const Eigen::Vector3d& u) const {
	Eigen::Matrix3d S;
	S << 0.0, -u.z(), u.y(),
		u.z(), 0.0, -u.x(),
		-u.y(), u.x(), 0.0;
	return S;
}

double VariationalShapeDeformation3D::omega_triangle(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2,
	const Eigen::Vector3d& v3, const Eigen::Vector3d& eta) const {
	Eigen::Vector3d e1 = v1 - eta;
	Eigen::Vector3d e2 = v2 - eta;
	Eigen::Vector3d e3 = v3 - eta;
	double r1 = e1.norm();
	double r2 = e2.norm();
	double r3 = e3.norm();
	double det = e1.dot(e2.cross(e3));
	double denom = r1 * r2 * r3 + e1.dot(e2) * r3 + e2.dot(e3) * r1 + e3.dot(e1) * r2;
	Eigen::Vector3d normal = (v2 - v1).cross(v3 - v1);	
	double signed_dist = (eta - v1).dot(normal);
	double Omega = 2.0 * std::atan2(det, denom);
	if (signed_dist > 0) {
		Omega = -std::abs(Omega);
	}
	else {
		Omega = std::abs(Omega);
	}

	return Omega / (4.0 * M_PI);
}

VariationalShapeDeformation3D::CiGradResult VariationalShapeDeformation3D::compute_Ci_gradCi(const Eigen::Vector3d& v1,
	const Eigen::Vector3d& v2,
	const Eigen::Vector3d& v3,
	const Eigen::Vector3d& eta) const {
	CiGradResult result;
	Eigen::Vector3d e1 = v1 - eta;
	Eigen::Vector3d e2 = v2 - eta;
	Eigen::Vector3d e3 = v3 - eta;
	result.e.row(0) = e1;
	result.e.row(1) = e2;
	result.e.row(2) = e3;
	double r1 = e1.norm();
	double r2 = e2.norm();
	double r3 = e3.norm();
	result.e_hat.row(0) = e1 / r1;
	result.e_hat.row(1) = e2 / r2;
	result.e_hat.row(2) = e3 / r3;	
	result.d.row(0) = v2 - v3;
	result.d.row(1) = v3 - v1;
	result.d.row(2) = v1 - v2;
	double L1 = result.d.row(0).norm();
	double L2 = result.d.row(1).norm();
	double L3 = result.d.row(2).norm();	
	result.R(0) = r2 + r3;
	result.R(1) = r3 + r1;
	result.R(2) = r1 + r2;	
	auto compute_C_Cbar = [](double R, double L) -> std::pair<double, double> {
		if (R <= L) throw std::runtime_error("Invalid geometry for log");
		double C = (1.0 / (4.0 * M_PI * L)) * std::log((R + L) / (R - L));
		double Cbar = 1.0 / (2.0 * M_PI * (R + L) * (R - L));
		return std::make_pair(C, Cbar);
		};

	std::pair<double, double> c_pair1 = compute_C_Cbar(result.R(0), L1);
	std::pair<double, double> c_pair2 = compute_C_Cbar(result.R(1), L2);
	std::pair<double, double> c_pair3 = compute_C_Cbar(result.R(2), L3);

	double C1 = c_pair1.first, Cbar1 = c_pair1.second;
	double C2 = c_pair2.first, Cbar2 = c_pair2.second;
	double C3 = c_pair3.first, Cbar3 = c_pair3.second;

	result.C << C1, C2, C3;
	result.Cbar << Cbar1, Cbar2, Cbar3;
	result.gradC.row(0) = Cbar1 * (result.e_hat.row(1) + result.e_hat.row(2));
	result.gradC.row(1) = Cbar2 * (result.e_hat.row(2) + result.e_hat.row(0));
	result.gradC.row(2) = Cbar3 * (result.e_hat.row(0) + result.e_hat.row(1));

	return result;
}


VariationalShapeDeformation3D::TriangleResult VariationalShapeDeformation3D::compute_triangle_derivatives_closed_form(const Eigen::Vector3d& v1,
	const Eigen::Vector3d& v2,
	const Eigen::Vector3d& v3,
	const Eigen::Vector3d& eta,
	int compute_phi_for_vertex) const {
	TriangleResult result;

	Eigen::Vector3d n_raw = (v2 - v1).cross(v3 - v1);
	double A = 0.5 * n_raw.norm();
	Eigen::Vector3d n = unit(n_raw);
	Eigen::Vector3d e1 = v1 - eta;
	Eigen::Vector3d e2 = v2 - eta;
	Eigen::Vector3d e3 = v3 - eta;
	double vol = std::abs((1.0 / 6.0) * e1.dot(e2.cross(e3)));
	Eigen::Vector3d Z1 = -e2.cross(e3);
	Eigen::Vector3d Z2 = -e3.cross(e1);
	Eigen::Vector3d Z3 = -e1.cross(e2);
	double om = omega_triangle(v1, v2, v3, eta);	
	CiGradResult ci_result = compute_Ci_gradCi(v1, v2, v3, eta);
	double psi = 0.0;
	psi -= ci_result.C(0) * Z1.dot(n);
	psi -= ci_result.C(1) * Z2.dot(n);
	psi -= ci_result.C(2) * Z3.dot(n);
	psi -= (3.0 / A) * om * vol;
	result.psi = psi;
	Eigen::Vector3d S = ci_result.C(0) * ci_result.d.row(0) +
		ci_result.C(1) * ci_result.d.row(1) +
		ci_result.C(2) * ci_result.d.row(2);
	Eigen::Vector3d P = n.cross(S) - om * n;
	result.grad_psi = -P;	
	double H = (v1 - eta).dot(n);
	Eigen::Vector3d s;
	s(0) = Z1.dot(n);
	s(1) = Z2.dot(n);
	s(2) = Z3.dot(n);
	Eigen::Vector3d grad_omega = -(1.0 / H) * (s(0) * ci_result.gradC.row(0) +
		s(1) * ci_result.gradC.row(1) +
		s(2) * ci_result.gradC.row(2));
	Eigen::Matrix3d J_S = ci_result.d.row(0).transpose() * ci_result.gradC.row(0) +
		ci_result.d.row(1).transpose() * ci_result.gradC.row(1) +
		ci_result.d.row(2).transpose() * ci_result.gradC.row(2);
	Eigen::Matrix3d J_P = skew(n) * J_S - n * grad_omega.transpose();
	result.hess_psi = -J_P;
	Eigen::Vector3d Z_phi, d_phi;
	Eigen::Matrix3d J_Z_phi;
	if (compute_phi_for_vertex == 0) {
		Z_phi = Z1;
		d_phi = ci_result.d.row(0);
		J_Z_phi = skew(e2) - skew(e3);
	}
	else if (compute_phi_for_vertex == 1) {
		Z_phi = Z2;
		d_phi = ci_result.d.row(1);
		J_Z_phi = skew(e3) - skew(e1);
	}
	else if (compute_phi_for_vertex == 2) {
		Z_phi = Z3;
		d_phi = ci_result.d.row(2);
		J_Z_phi = skew(e1) - skew(e2);
	}
	else {
		throw std::runtime_error("compute_phi_for_vertex must be 0, 1, or 2");
	}	
	double phi = (1.0 / (2.0 * A)) * P.dot(Z_phi);
	result.phi[compute_phi_for_vertex] = phi;
	result.grad_phi[compute_phi_for_vertex] = (1.0 / (2.0 * A)) *
		(J_Z_phi.transpose() * P);
	result.hess_phi[compute_phi_for_vertex] = (1.0 / (2.0 * A)) *
		(-skew(d_phi) * J_P);
	return result;
}



void VariationalShapeDeformation3D::computeGreenCoordinateGradientsAtPointNumerical(const Eigen::Vector3d& p,
	Eigen::MatrixXd& grad_phi, 
	Eigen::MatrixXd& grad_psi) { 
	int n_vertices = C_.rows();
	int n_faces = CF_.rows();

	grad_phi.resize(3, n_vertices);
	grad_psi.resize(3, n_faces);
	grad_phi.setZero();
	grad_psi.setZero();
	Eigen::VectorXd phi_base, psi_base;
	computeGreenCoordinatesAtPoint(p, phi_base, psi_base);
	for (int dim = 0; dim < 3; ++dim) {
		Eigen::Vector3d delta = Eigen::Vector3d::Zero();
		delta[dim] = epsilon_;
		Eigen::VectorXd phi_plus, psi_plus;
		computeGreenCoordinatesAtPoint(p + delta, phi_plus, psi_plus);
		Eigen::VectorXd phi_minus, psi_minus;
		computeGreenCoordinatesAtPoint(p - delta, phi_minus, psi_minus);
		for (int i = 0; i < n_vertices; ++i) {
			grad_phi(dim, i) = (phi_plus(i) - phi_minus(i)) / (2 * epsilon_);
		}
		for (int i = 0; i < n_faces; ++i) {
			grad_psi(dim, i) = (psi_plus(i) - psi_minus(i)) / (2 * epsilon_);
		}
	}
}


void VariationalShapeDeformation3D::computeGreenCoordinateGradientsAtPointClose(
	const Eigen::Vector3d& p,
	Eigen::MatrixXd& grad_phi, 
	Eigen::MatrixXd& grad_psi) { 
	int n_vertices = C_.rows();
	int n_faces = CF_.rows();
	grad_phi.resize(3, n_vertices);
	grad_psi.resize(3, n_faces);
	grad_phi.setZero();
	grad_psi.setZero();
	Eigen::Vector3d p_transformed = Binv_ * p;
	for (int face_idx = 0; face_idx < n_faces; ++face_idx) {
		Eigen::Vector3i tri_indices = CF_.row(face_idx);
		Eigen::Vector3d ov0 = C_.row(tri_indices[0]).transpose();
		Eigen::Vector3d ov1 = C_.row(tri_indices[1]).transpose();
		Eigen::Vector3d ov2 = C_.row(tri_indices[2]).transpose();
		Eigen::Vector3d original_normal = ((ov1 - ov0).cross(ov2 - ov0)).normalized();
		double nAn = original_normal.transpose() * A_ * original_normal;
		double anisotropy_factor = 1.0 / std::sqrt(nAn);
		Eigen::Vector3d v0 = Binv_ * C_.row(tri_indices[0]).transpose();
		Eigen::Vector3d v1 = Binv_ * C_.row(tri_indices[1]).transpose();
		Eigen::Vector3d v2 = Binv_ * C_.row(tri_indices[2]).transpose();
		TriangleResult result0 = compute_triangle_derivatives_closed_form(
			v0, v1, v2, p_transformed, 0);	
		Eigen::Vector3d grad_psi_transformed = result0.grad_psi * anisotropy_factor;	
		grad_psi.col(face_idx) = Binv_.transpose() * grad_psi_transformed;
		TriangleResult result1 = compute_triangle_derivatives_closed_form(
			v0, v1, v2, p_transformed, 1);
		TriangleResult result2 = compute_triangle_derivatives_closed_form(
			v0, v1, v2, p_transformed, 2);
		grad_phi.col(tri_indices[0]) += Binv_.transpose() * result0.grad_phi[0];
		grad_phi.col(tri_indices[1]) += Binv_.transpose() * result1.grad_phi[1];
		grad_phi.col(tri_indices[2]) += Binv_.transpose() * result2.grad_phi[2];
	}
}


void VariationalShapeDeformation3D::computeGreenCoordinateHessiansAtPointNumerical(const Eigen::Vector3d& p,
	std::vector<Eigen::Matrix3d>& hess_phi,
	std::vector<Eigen::Matrix3d>& hess_psi) { 
	int n_vertices = C_.rows();
	int n_faces = CF_.rows();
	hess_phi.resize(n_vertices);
	hess_psi.resize(n_faces);

	for (int i = 0; i < n_vertices; ++i) hess_phi[i].setZero();
	for (int i = 0; i < n_faces; ++i) hess_psi[i].setZero();
	
	Eigen::MatrixXd grad_phi_base, grad_psi_base;
	computeGreenCoordinateGradientsAtPointNumerical(p, grad_phi_base, grad_psi_base);

	for (int dim1 = 0; dim1 < 3; ++dim1) {
		Eigen::Vector3d delta = Eigen::Vector3d::Zero();
		delta[dim1] = epsilon_;	
		Eigen::MatrixXd grad_phi_plus, grad_psi_plus;
		computeGreenCoordinateGradientsAtPointNumerical(p + delta, grad_phi_plus, grad_psi_plus);
		Eigen::MatrixXd grad_phi_minus, grad_psi_minus;
		computeGreenCoordinateGradientsAtPointNumerical(p - delta, grad_phi_minus, grad_psi_minus);
		for (int i = 0; i < n_vertices; ++i) {
			for (int dim2 = 0; dim2 < 3; ++dim2) {
				hess_phi[i](dim1, dim2) = (grad_phi_plus(dim2, i) - grad_phi_minus(dim2, i)) / (2 * epsilon_);
			}
		}

		for (int i = 0; i < n_faces; ++i) {
			for (int dim2 = 0; dim2 < 3; ++dim2) {
				hess_psi[i](dim1, dim2) = (grad_psi_plus(dim2, i) - grad_psi_minus(dim2, i)) / (2 * epsilon_);
			}
		}
	}
}


void VariationalShapeDeformation3D::computeGreenCoordinateHessiansAtPointClose(
	const Eigen::Vector3d& p,
	std::vector<Eigen::Matrix3d>& hess_phi, 
	std::vector<Eigen::Matrix3d>& hess_psi) {

	int n_vertices = C_.rows();
	int n_faces = CF_.rows();

	hess_phi.assign(n_vertices, Eigen::Matrix3d::Zero());
	hess_psi.assign(n_faces, Eigen::Matrix3d::Zero());

	
	Eigen::Vector3d p_transformed = Binv_ * p;

	for (int face_idx = 0; face_idx < n_faces; ++face_idx) 
	{
		Eigen::Vector3i tri_indices = CF_.row(face_idx);
		Eigen::Vector3d ov0 = C_.row(tri_indices[0]).transpose();
		Eigen::Vector3d ov1 = C_.row(tri_indices[1]).transpose();
		Eigen::Vector3d ov2 = C_.row(tri_indices[2]).transpose();
		Eigen::Vector3d original_normal = ((ov1 - ov0).cross(ov2 - ov0)).normalized();

		double nAn = original_normal.transpose() * A_ * original_normal;
		double anisotropy_factor = 1.0 / std::sqrt(nAn);
		
		Eigen::Vector3d v0 = Binv_ * C_.row(tri_indices[0]).transpose();
		Eigen::Vector3d v1 = Binv_ * C_.row(tri_indices[1]).transpose();
		Eigen::Vector3d v2 = Binv_ * C_.row(tri_indices[2]).transpose();

		TriangleResult result0 = compute_triangle_derivatives_closed_form(
			v0, v1, v2, p_transformed, 0);

		Eigen::Matrix3d hess_psi_transformed = result0.hess_psi * anisotropy_factor;
		hess_psi[face_idx] = Binv_.transpose() * hess_psi_transformed * Binv_;
		TriangleResult result1 = compute_triangle_derivatives_closed_form(
			v0, v1, v2, p_transformed, 1);
		TriangleResult result2 = compute_triangle_derivatives_closed_form(
			v0, v1, v2, p_transformed, 2);
		hess_phi[tri_indices[0]] += Binv_.transpose() * result0.hess_phi[0] * Binv_;
		hess_phi[tri_indices[1]] += Binv_.transpose() * result1.hess_phi[1] * Binv_;
		hess_phi[tri_indices[2]] += Binv_.transpose() * result2.hess_phi[2] * Binv_;
	}
}


Eigen::Vector3d VariationalShapeDeformation3D::deformPoint(const Eigen::Vector3d& point,
	const Eigen::MatrixXd& deformed_cage_vertices,
	const Eigen::MatrixXd& deformed_face_coeffs) {
	Eigen::VectorXd phi, psi;
	computeGreenCoordinatesAtPoint(point, phi, psi);

	Eigen::Vector3d deformed_point = Eigen::Vector3d::Zero();

	
	for (int i = 0; i < phi.size(); ++i) {
		deformed_point += phi(i) * deformed_cage_vertices.row(i).transpose();
	}

	
	for (int i = 0; i < psi.size(); ++i) {
		deformed_point += psi(i) * deformed_face_coeffs.row(i).transpose();
	}

	return deformed_point;
}


void VariationalShapeDeformation3D::computePhiAndPsiForOneTriangle(const Eigen::Vector3d& eta,
	Eigen::Vector3d* tri_vertices,
	double* phi,
	double& psi) {
	
	Eigen::Vector3d Nt = (tri_vertices[1] - tri_vertices[0]).cross(tri_vertices[2] - tri_vertices[0]);
	double NtNorm = Nt.norm();
	double At = NtNorm / 2.0;
	Nt /= NtNorm;

	psi = 0.0;
	for (unsigned int v = 0; v < 3; ++v) phi[v] = 0.0;

	Eigen::Vector3d side_integral;
	AnisotropicIntegralFundamentalSolution(tri_vertices, eta, psi, side_integral, A_, B_, Binv_);

	Eigen::Vector3d e[3];
	Eigen::Vector3d J[3];
	for (unsigned int v = 0; v < 3; ++v) e[v] = tri_vertices[v] - eta;
	for (unsigned int v = 0; v < 3; ++v) J[v] = e[(v + 2) % 3].cross(e[(v + 1) % 3]);

	Eigen::Vector3d side_normal[3];
	for (unsigned int v = 0; v < 3; ++v) {
		side_normal[v] = J[v] / (J[v].norm());
	}

	Eigen::Vector3d psi_pgpn = (-psi * (A_ * Nt));
	Eigen::Vector3d AJv[3];
	for (unsigned int v = 0; v < 3; ++v) {
		AJv[v] = A_ * side_normal[v];
		psi_pgpn -= (AJv[v] * side_integral[v]);
	}

	for (unsigned int v = 0; v < 3; ++v) {
		double uu = side_normal[v].dot(e[v]);
		if (std::abs(uu) > 1e-10) {
			phi[v] += ((side_normal[v].dot(psi_pgpn)) / (side_normal[v].dot(e[v])));
		}
	}
}

void VariationalShapeDeformation3D::AnisotropicIntegralFundamentalSolution(Eigen::Vector3d* tri_vertices,
	const Eigen::Vector3d& eta,
	double& psi,
	Eigen::Vector3d& side_integral,
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& B,
	const Eigen::MatrixXd& Binv) {
	
	Eigen::Vector3d Nt = (tri_vertices[1] - tri_vertices[0]).cross(tri_vertices[2] - tri_vertices[0]);
	double NtNorm = Nt.norm();
	double At = NtNorm / 2.0;
	Nt /= NtNorm;
	double factor = sqrt(Nt.transpose() * A * Nt);
	factor = 1.0 / factor;

	Eigen::Vector3d newTrianglePoints[3];
	for (unsigned int v = 0; v < 3; ++v) {
		newTrianglePoints[v] = Binv * (tri_vertices[v] - eta);
	}

	Eigen::Vector3d zero_vector = Eigen::Vector3d::Zero();
	IntegralFundamentalSolution(newTrianglePoints, zero_vector, psi, side_integral);
	psi *= factor;

	Eigen::Vector3d e[3];
	Eigen::Vector3d J[3];
	for (unsigned int v = 0; v < 3; ++v) e[v] = tri_vertices[v] - eta;
	for (unsigned int v = 0; v < 3; ++v) J[v] = e[(v + 2) % 3].cross(e[(v + 1) % 3]);

	Eigen::Vector3d side_normal[3];
	for (unsigned int v = 0; v < 3; ++v) {
		side_normal[v] = J[v] / (J[v].norm());
	}

	double side_factor[3];
	for (unsigned int v = 0; v < 3; ++v) {
		side_factor[v] = sqrt(side_normal[v].transpose() * A * side_normal[v]);
		side_factor[v] = 1.0 / side_factor[v];
	}

	for (unsigned int v = 0; v < 3; ++v) side_integral[v] *= side_factor[v];
}

void VariationalShapeDeformation3D::IntegralFundamentalSolution(Eigen::Vector3d* tri_vertices,
	const Eigen::Vector3d& eta,
	double& psi,
	Eigen::Vector3d& side_integral) {
	
	Eigen::Vector3d Nt = (tri_vertices[1] - tri_vertices[0]).cross(tri_vertices[2] - tri_vertices[0]);
	double NtNorm = Nt.norm();
	double At = NtNorm / 2.0;
	Nt /= NtNorm;

	Eigen::Vector3d e[3];
	double e_norm[3];
	Eigen::Vector3d e_normalized[3];
	double R[3];
	Eigen::Vector3d d[3];
	double d_norm[3];
	double C[3];

	for (unsigned int v = 0; v < 3; ++v) e[v] = tri_vertices[v] - eta;
	for (unsigned int v = 0; v < 3; ++v) e_norm[v] = e[v].norm();
	for (unsigned int v = 0; v < 3; ++v) e_normalized[v] = e[v] / e_norm[v];

	double signed_solid_angle = get_signed_solid_angle(e_normalized[0], e_normalized[1], e_normalized[2]) / (4.0 * M_PI);
	double signed_volume = (e[0].cross(e[1])).dot(e[2]) / 6.0;

	for (unsigned int v = 0; v < 3; ++v) R[v] = e_norm[(v + 1) % 3] + e_norm[(v + 2) % 3];
	for (unsigned int v = 0; v < 3; ++v) d[v] = tri_vertices[(v + 1) % 3] - tri_vertices[(v + 2) % 3];
	for (unsigned int v = 0; v < 3; ++v) d_norm[v] = d[v].norm();
	for (unsigned int v = 0; v < 3; ++v) C[v] = std::log((R[v] + d_norm[v]) / (R[v] - d_norm[v])) / (4.0 * M_PI * d_norm[v]);

	psi = -3.0 * signed_solid_angle * signed_volume / At;

	Eigen::Vector3d J[3];
	for (unsigned int v = 0; v < 3; ++v) J[v] = e[(v + 2) % 3].cross(e[(v + 1) % 3]);
	for (unsigned int v = 0; v < 3; ++v) psi -= C[v] * J[v].dot(Nt);
	for (unsigned int v = 0; v < 3; ++v) side_integral[v] = J[v].norm() * C[v];
}

double VariationalShapeDeformation3D::get_signed_solid_angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
	double numerator = a.dot(b.cross(c));
	double denominator = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
	return 2.0 * atan2(numerator, denominator);
}



VariationalDeformationSolver3D::VariationalDeformationSolver3D(VariationalShapeDeformation3D& deformation)
	: deformation_(deformation) {}

void VariationalDeformationSolver3D::solve(const Eigen::MatrixXd& user_constraints_origin,
	const Eigen::MatrixXd& user_constraints_deformed,
	const Eigen::MatrixXd& medial_points,
	const Eigen::MatrixXd& boundary_samples, Eigen::MatrixXd& A) {
	/*
	std::vector<Eigen::Vector3d> test_points;
	test_points.push_back(Eigen::Vector3d(0.1, 0.1, 0.1));
	test_points.push_back(Eigen::Vector3d(0.2, 0.3, 0.15));
	test_points.push_back(Eigen::Vector3d(0.35, 0.2, 0.25));
	deformation_.debugCompareNumericalAndClosedForm(test_points);
	*/


	int n_vertices = deformation_.getCageVertices().rows();
	int n_faces = deformation_.getCageFaces().rows();
	Eigen::MatrixXd a = deformation_.getCageVertices().transpose(); // 3 x n
	Eigen::MatrixXd b = Eigen::MatrixXd::Zero(3, n_faces); // 3 x m

	for (int i = 0; i < n_faces; ++i) {
		Eigen::Vector3i face = deformation_.getCageFaces().row(i);
		Eigen::Vector3d v0 = deformation_.getCageVertices().row(face[0]);
		Eigen::Vector3d v1 = deformation_.getCageVertices().row(face[1]);
		Eigen::Vector3d v2 = deformation_.getCageVertices().row(face[2]);

		Eigen::Vector3d edge1 = v1 - v0;
		Eigen::Vector3d edge2 = v2 - v0;
		Eigen::Vector3d face_normal = edge1.cross(edge2);

		double norm = face_normal.norm();
		if (norm > 1e-10) {
			face_normal /= norm;
		}
		b.col(i) = A * face_normal;
	}
	Eigen::MatrixXd a0 = a;
	Eigen::MatrixXd b0 = b;
	
	std::vector<Eigen::Matrix3d> R(medial_points.rows(), Eigen::Matrix3d::Identity());

	Eigen::MatrixXd a_prev = a;
	Eigen::MatrixXd b_prev = b;
	std::cout << "Starting variational deformation with " << max_iter_ << " iterations" << std::endl;

	for (int iter = 0; iter < max_iter_; ++iter) {
		std::cout << "Iteration " << iter << std::endl;
		Eigen::MatrixXd a_prev = a;
		Eigen::MatrixXd b_prev = b;
		
		R = solveLocalStepClose(a, b, medial_points);
		solveGlobalStepClose(a, b, R, user_constraints_origin, user_constraints_deformed, medial_points, boundary_samples, a_prev, b_prev);
		double delta_a = (a - a_prev).norm();
		double delta_b = (b - b_prev).norm();

		std::cout << "  Delta a: " << delta_a << ", Delta b: " << delta_b << std::endl;

		if (delta_a < tol_ && delta_b < tol_) {
			std::cout << "Converged after " << iter << " iterations!" << std::endl;
			break;
		}

		a_prev = a;
		b_prev = b;
	}
	final_a_ = a;
	final_b_ = b;
	
	std::cout << "Deformation completed." << std::endl;
	
}


std::vector<Eigen::Matrix3d> VariationalDeformationSolver3D::solveLocalStepClose(const Eigen::MatrixXd& a,
	const Eigen::MatrixXd& b,
	const Eigen::MatrixXd& anchors) {
	std::vector<Eigen::Matrix3d> R_new(anchors.rows());

	for (int i = 0; i < anchors.rows(); ++i) {
		Eigen::Vector3d p = anchors.row(i);
		Eigen::Matrix3d J = computeJacobianAtPointClose(p, a, b);		
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
		
		if (R.determinant() < 0) {
			Eigen::Matrix3d V = svd.matrixV();
			V.col(2) = -V.col(2);
			R = svd.matrixU() * V.transpose();
		}

		R_new[i] = R;
	}

	return R_new;
}


Eigen::Matrix3d VariationalDeformationSolver3D::computeJacobianAtPointNumerical(const Eigen::Vector3d& p,
	const Eigen::MatrixXd& a,
	const Eigen::MatrixXd& b) {
	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();

	
	Eigen::MatrixXd grad_phi, grad_psi;
	deformation_.computeGreenCoordinateGradientsAtPointNumerical(p, grad_phi, grad_psi);

	int n_vertices = a.cols();
	int n_faces = b.cols();

	
	for (int v = 0; v < n_vertices; ++v) {
		for (int dim = 0; dim < 3; ++dim) {
			J.row(dim) += a(dim, v) * grad_phi.col(v).transpose();
		}
	}

	
	for (int f = 0; f < n_faces; ++f) {
		for (int dim = 0; dim < 3; ++dim) {
			J.row(dim) += b(dim, f) * grad_psi.col(f).transpose();
		}
	}

	return J.transpose();
}

Eigen::Matrix3d VariationalDeformationSolver3D::computeJacobianAtPointClose(const Eigen::Vector3d& p,
	const Eigen::MatrixXd& a,
	const Eigen::MatrixXd& b) {
	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	Eigen::MatrixXd grad_phi, grad_psi;
	deformation_.computeGreenCoordinateGradientsAtPointClose(p, grad_phi, grad_psi);
	int n_vertices = a.cols();
	int n_faces = b.cols();
	for (int v = 0; v < n_vertices; ++v) {
		for (int dim = 0; dim < 3; ++dim) {
			J.row(dim) += a(dim, v) * grad_phi.col(v).transpose();
		}
	}
	for (int f = 0; f < n_faces; ++f) {
		for (int dim = 0; dim < 3; ++dim) {
			J.row(dim) += b(dim, f) * grad_psi.col(f).transpose();
		}
	}
	return J.transpose();
}


void VariationalShapeDeformation3D::debugCompareNumericalAndClosedForm(
	const std::vector<Eigen::Vector3d>& test_points)
{
	
	auto maxAbsDiffMatXd = [](const Eigen::MatrixXd& a,
		const Eigen::MatrixXd& b) {
			return (a - b).array().abs().maxCoeff();
		};

	
	auto maxAbsDiffMat3d = [](const std::vector<Eigen::Matrix3d>& a,
		const std::vector<Eigen::Matrix3d>& b) {
			double m = 0.0;
			for (size_t i = 0; i < a.size(); ++i) 
			{
				
				m = std::max(m, (a[i] - b[i]).cwiseAbs().maxCoeff());
			}
			return m;
		};

	const double eps = 1e-12;

	for (size_t pi = 0; pi < test_points.size(); ++pi) {
		const Eigen::Vector3d& p = test_points[pi];
		Eigen::MatrixXd grad_phi_num, grad_psi_num;
		std::vector<Eigen::Matrix3d> hess_phi_num, hess_psi_num;
		computeGreenCoordinateGradientsAtPointNumerical(p, grad_phi_num, grad_psi_num);
		computeGreenCoordinateHessiansAtPointNumerical(p, hess_phi_num, hess_psi_num);
		Eigen::MatrixXd grad_phi_cf, grad_psi_cf;
		std::vector<Eigen::Matrix3d> hess_phi_cf, hess_psi_cf;
		computeGreenCoordinateGradientsAtPointClose(p, grad_phi_cf, grad_psi_cf);
		computeGreenCoordinateHessiansAtPointClose(p, hess_phi_cf, hess_psi_cf);

		double grad_phi_err = maxAbsDiffMatXd(grad_phi_num, grad_phi_cf);
		double grad_psi_err = maxAbsDiffMatXd(grad_psi_num, grad_psi_cf);
		double hess_phi_err = maxAbsDiffMat3d(hess_phi_num, hess_phi_cf);
		double hess_psi_err = maxAbsDiffMat3d(hess_psi_num, hess_psi_cf);
		double grad_phi_ref = std::max(1.0, grad_phi_num.array().abs().maxCoeff());
		double grad_psi_ref = std::max(1.0, grad_psi_num.array().abs().maxCoeff());

		double hess_phi_ref = 1.0;
		for (const auto& m : hess_phi_num) {
			hess_phi_ref = std::max(hess_phi_ref, m.cwiseAbs().maxCoeff());
		}
		double hess_psi_ref = 1.0;
		for (const auto& m : hess_psi_num) {
			hess_psi_ref = std::max(hess_psi_ref, m.cwiseAbs().maxCoeff());
		}

		std::cout << "[Check point " << pi << "] p = "
			<< p.transpose() << "\n";
		std::cout << "  grad_phi max abs diff = " << grad_phi_err
			<< ", rel = " << grad_phi_err / grad_phi_ref << "\n";
		std::cout << "  grad_psi max abs diff = " << grad_psi_err
			<< ", rel = " << grad_psi_err / grad_psi_ref << "\n";
		std::cout << "  hess_phi max abs diff = " << hess_phi_err
			<< ", rel = " << hess_phi_err / hess_phi_ref << "\n";
		std::cout << "  hess_psi max abs diff = " << hess_psi_err
			<< ", rel = " << hess_psi_err / hess_psi_ref << "\n";

		if (grad_phi_err > 1e-3 || grad_psi_err > 1e-3 ||
			hess_phi_err > 1e-3 || hess_psi_err > 1e-3) {
			std::cout << "  [WARN] mismatch detected\n";
		}
	}
}



Eigen::VectorXd VariationalDeformationSolver3D::solveLeastSquares(const Eigen::MatrixXd& A_ls,
	const Eigen::VectorXd& B_ls,
	const Eigen::VectorXd& x0) {

	int n = x0.size();
	int m = A_ls.rows();
	double lambda_fidelity = 1.0;      
	double lambda_change = lambda_change_;        
	Eigen::MatrixXd A_aug(m + n, n);
	Eigen::VectorXd B_aug(m + n);
	A_aug.topRows(m) = sqrt(lambda_fidelity) * A_ls;
	B_aug.head(m) = sqrt(lambda_fidelity) * B_ls;
	A_aug.bottomRows(n) = sqrt(lambda_change) * Eigen::MatrixXd::Identity(n, n);
	B_aug.tail(n) = sqrt(lambda_change) * x0;

	
	return A_aug.colPivHouseholderQr().solve(B_aug);
}


void VariationalDeformationSolver3D::solveGlobalStepClose(Eigen::MatrixXd& a, Eigen::MatrixXd& b,
	const std::vector<Eigen::Matrix3d>& R,
	const Eigen::MatrixXd& user_constraints_origin,
	const Eigen::MatrixXd& user_constraints_deformed,
	const Eigen::MatrixXd& anchors,
	const Eigen::MatrixXd& boundary_samples,
	const Eigen::MatrixXd& a0, const Eigen::MatrixXd& b0) {

	int n_vertices = a.cols();
	int n_faces = b.cols();
	int total_vars = n_vertices + n_faces;
	for (int dim = 0; dim < 3; ++dim) {
		
		Eigen::MatrixXd A_ls;
		Eigen::VectorXd B_ls;
		Eigen::MatrixXd C_constraints;  
		Eigen::VectorXd d_constraints;  

		buildLinearSystemForDimensionClose(dim, a, b, R, user_constraints_origin,
			user_constraints_deformed, anchors, boundary_samples,
			A_ls, B_ls, C_constraints, d_constraints);
		Eigen::VectorXd x0(total_vars);
		for (int i = 0; i < n_vertices; ++i) {
			x0[i] = a0(dim, i);
		}
		for (int i = 0; i < n_faces; ++i) {
			x0[n_vertices + i] = b0(dim, i);
		}

		Eigen::VectorXd solution = solveLeastSquares(A_ls, B_ls, x0);
		a.row(dim) = solution.head(n_vertices).transpose();
		b.row(dim) = solution.tail(n_faces).transpose();
	}
}


void VariationalDeformationSolver3D::buildLinearSystemForDimensionClose(int dim,
	const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
	const std::vector<Eigen::Matrix3d>& R,
	const Eigen::MatrixXd& user_constraints_origin,
	const Eigen::MatrixXd& user_constraints_deformed,
	const Eigen::MatrixXd& anchors,
	const Eigen::MatrixXd& boundary_samples,
	Eigen::MatrixXd& A_ls, Eigen::VectorXd& B_ls,
	Eigen::MatrixXd& C_constraints, Eigen::VectorXd& d_constraints) {

	int n_vertices = a.cols();
	int n_faces = b.cols();
	int total_vars = n_vertices + n_faces;

	std::vector<Eigen::Triplet<double>> A_triplets;
	std::vector<double> B_values;

	int row_counter = 0;
	const double lambda_rigid = lambda_rigid_;
	
	for (int i = 0; i < anchors.rows(); ++i) {
		Eigen::Vector3d p = anchors.row(i);
		Eigen::MatrixXd grad_phi, grad_psi;
		deformation_.computeGreenCoordinateGradientsAtPointClose(p, grad_phi, grad_psi);
		for (int j = 0; j < 3; ++j) {
			for (int v = 0; v < n_vertices; ++v) {
				A_triplets.emplace_back(row_counter + j, v, sqrt(lambda_rigid) * grad_phi(j, v));
			}
			for (int f = 0; f < n_faces; ++f) {
				A_triplets.emplace_back(row_counter + j, n_vertices + f, sqrt(lambda_rigid) * grad_psi(j, f));
			}

			B_values.push_back(sqrt(lambda_rigid) * R[i](j, dim));
		}
		row_counter += 3;
	}


	for (int i = 0; i < boundary_samples.rows(); ++i) {
		Eigen::Vector3d p = boundary_samples.row(i);
		std::vector<Eigen::Matrix3d> hess_phi, hess_psi;
		deformation_.computeGreenCoordinateHessiansAtPointClose(p, hess_phi, hess_psi);
		int row_this = 0;
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k <= j; k++) {
				for (int v = 0; v < n_vertices; ++v) {
					A_triplets.emplace_back(row_counter + row_this, v, sqrt(lambda_) * hess_phi[v](j, k));
				}
				for (int f = 0; f < n_faces; ++f) {
					A_triplets.emplace_back(row_counter + row_this, n_vertices + f, sqrt(lambda_) * hess_psi[f](j, k));
				}
				B_values.push_back(0.0); 
				row_this++;
			}
		}
		row_counter += 6;
	}

	for (int i = 0; i < user_constraints_origin.rows(); ++i) {
		Eigen::Vector3d p = user_constraints_origin.row(i);
		Eigen::VectorXd phi, psi;
		deformation_.computeGreenCoordinatesAtPoint(p, phi, psi);
		double weight = sqrt(lambda_user_);

		for (int v = 0; v < n_vertices; ++v) {
			A_triplets.emplace_back(row_counter, v, weight * phi(v));
		}
		for (int f = 0; f < n_faces; ++f) {
			A_triplets.emplace_back(row_counter, n_vertices + f, weight * psi(f));
		}

		B_values.push_back(weight * user_constraints_deformed(i, dim));
		row_counter++;
	}

	
	A_ls = Eigen::MatrixXd::Zero(row_counter, total_vars);
	for (const auto& triplet : A_triplets) {
		A_ls(triplet.row(), triplet.col()) += triplet.value();
	}

	B_ls = Eigen::Map<Eigen::VectorXd>(B_values.data(), B_values.size());
	C_constraints = Eigen::MatrixXd::Zero(0, total_vars);
	d_constraints = Eigen::VectorXd::Zero(0);
}

Eigen::VectorXd VariationalDeformationSolver3D::solveConstrainedLeastSquares(const Eigen::MatrixXd& A_ls,
	const Eigen::VectorXd& B_ls,
	const Eigen::MatrixXd& C_constraints,
	const Eigen::VectorXd& d_constraints,
	const Eigen::VectorXd& x0) {
	int n = x0.size();
	int m = A_ls.rows();
	int p = C_constraints.rows();
	double lambda_fidelity = 1.0;
	double lambda_change = lambda_change_;
	if (p == 0) {
		
		Eigen::MatrixXd A_aug(m + n, n);
		Eigen::VectorXd B_aug(m + n);
		A_aug << sqrt(lambda_fidelity) * A_ls,
			sqrt(lambda_change)* Eigen::MatrixXd::Identity(n, n);

		B_aug << sqrt(lambda_fidelity) * B_ls,
			sqrt(lambda_change)* x0;
		return A_aug.colPivHouseholderQr().solve(B_aug);
	}
	else {
		
		Eigen::MatrixXd H = lambda_change * Eigen::MatrixXd::Identity(n, n) +
			lambda_fidelity * (A_ls.transpose() * A_ls);
		Eigen::VectorXd f = -lambda_change * x0 - lambda_fidelity * A_ls.transpose() * B_ls;
		Eigen::MatrixXd KKT(n + p, n + p);
		Eigen::VectorXd rhs(n + p);

		KKT << H, C_constraints.transpose(),
			C_constraints, Eigen::MatrixXd::Zero(p, p);

		rhs << -f, d_constraints;
		Eigen::VectorXd full_solution = KKT.colPivHouseholderQr().solve(rhs);
		return full_solution.head(n);
	}
}


void IntegralFundamentalSolution(Eigen::Vector3d* tri_vertices, Eigen::Vector3d eta, double& psi, double *phi)
{
	Eigen::Vector3d Nt = (tri_vertices[1] - tri_vertices[0]).cross(tri_vertices[2] - tri_vertices[0]);
	double NtNorm = Nt.norm();
	double At = NtNorm / 2.0;
	Nt /= NtNorm;
	Eigen::Vector3d e[3];  double e_norm[3];   Eigen::Vector3d e_normalized[3];    double R[3];    Eigen::Vector3d d[3];    double d_norm[3];     double C[3];
	for (unsigned int v = 0; v < 3; ++v) e[v] = tri_vertices[v] - eta;
	for (unsigned int v = 0; v < 3; ++v) e_norm[v] = e[v].norm();
	for (unsigned int v = 0; v < 3; ++v) e_normalized[v] = e[v] / e_norm[v];

	double signed_solid_angle = get_signed_solid_angle(e_normalized[0], e_normalized[1], e_normalized[2]) / (4.f * M_PI);
	double signed_volume = (e[0].cross(e[1])).dot(e[2]) / 6.0;

	for (unsigned int v = 0; v < 3; ++v) R[v] = e_norm[(v + 1) % 3] + e_norm[(v + 2) % 3];
	for (unsigned int v = 0; v < 3; ++v) d[v] = tri_vertices[(v + 1) % 3] - tri_vertices[(v + 2) % 3];
	for (unsigned int v = 0; v < 3; ++v) d_norm[v] = d[v].norm();
	for (unsigned int v = 0; v < 3; ++v) C[v] = std::log((R[v] + d_norm[v]) / (R[v] - d_norm[v])) / (4.0 * M_PI * d_norm[v]);

	Eigen::Vector3d J[3];
	Eigen::Vector3d Pt(-signed_solid_angle * Nt);
	for (unsigned int v = 0; v < 3; ++v) Pt += Nt.cross(C[v] * d[v]);
	for (unsigned int v = 0; v < 3; ++v) J[v] = e[(v + 2) % 3].cross(e[(v + 1) % 3]);

	psi = -3.0 * signed_solid_angle * signed_volume / At;
	for (unsigned int v = 0; v < 3; ++v) psi -= C[v] * J[v].dot(Nt);
	for (unsigned int v = 0; v < 3; ++v) phi[v] += Pt.dot(J[v]) / (2.0 * At);
}


void AnisotropicIntegralFundamentalSolution(Eigen::Vector3d* tri_vertices, Eigen::Vector3d eta, double& psi, double* phi, Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& Binv)
{
	Eigen::Vector3d Nt = (tri_vertices[1] - tri_vertices[0]).cross(tri_vertices[2] - tri_vertices[0]);
	double NtNorm = Nt.norm();
	double At = NtNorm / 2.0;
	Nt /= NtNorm;
	double factor = sqrt(Nt.transpose() * A * Nt);
	factor = 1.0 / factor;
	Eigen::Vector3d newTrianglePoints[3];
	for (unsigned int v = 0; v < 3; ++v)
	{
		newTrianglePoints[v] = Binv * (tri_vertices[v] - eta);
	}
	Eigen::Vector3d zero_vector;
	zero_vector.setZero();
	IntegralFundamentalSolution(newTrianglePoints, zero_vector, psi, phi);
	psi *= factor;
}


template< class float_t, class point_t >
void computePhiAndPsiForOneTriangle(point_t const& eta,
	point_t* tri_vertices, // an array of 3 points
	double* phi, // an array of 3 floats
	float_t& psi, Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& Binv) {
	typedef double    T;
	point_t Nt = (tri_vertices[1] - tri_vertices[0]).cross(tri_vertices[2] - tri_vertices[0]);
	T NtNorm = Nt.norm();
	T At = NtNorm / 2.0;
	Nt /= NtNorm;
	point_t ANt;
	ANt = A * Nt;
	psi = 0.0;
	for (unsigned int v = 0; v < 3; ++v) phi[v] = 0.0;
	AnisotropicIntegralFundamentalSolution(tri_vertices, eta,  psi, phi, A, B, Binv);
	
}


void calculateAGC(const Eigen::MatrixXd& C, const Eigen::MatrixXi& CF, const Eigen::MatrixXd& normals, const Eigen::MatrixXd& eta_m,
	Eigen::MatrixXd& phi, Eigen::MatrixXd& psi, Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& Binv)
{
	phi.resize(C.rows(), eta_m.rows());
	psi.resize(CF.rows(), eta_m.rows());
	phi.fill(0); psi.fill(0);

	for (int i = 0; i < eta_m.rows(); ++i)
	{
		const Eigen::Vector3d eta = eta_m.row(i);

		for (unsigned int face_idx = 0; face_idx < CF.rows(); ++face_idx)
		{
			Eigen::Vector3d tri_verts[3];
			const Eigen::Vector3i tri_indices = CF.row(face_idx);
			double phi_[3];
			for (unsigned int l = 0; l < 3u; ++l)
			{
				tri_verts[l] = C.row(tri_indices[l]);
			}
			computePhiAndPsiForOneTriangle(eta, tri_verts, phi_, psi(face_idx, i), A, B, Binv);

			for (unsigned int v = 0; v < 3u; ++v)
			{
				phi(tri_indices[v], i) += phi_[v];
			}
		}
	}
	/*
	for (unsigned int i = 0; i < eta_m.rows(); ++i)
	{
		double res = 0;
		for (unsigned int j = 0; j < C.rows(); ++j)
		{
			res += phi(j, i);
		}
		assert(std::abs(1. - res) < 1e-3);
	}
	*/
}



void calcNormals(const Eigen::MatrixXd& C, const Eigen::MatrixXi& CF, Eigen::MatrixXd& normals)
{
	normals.resize(CF.rows(), 3);

	for (int i = 0; i < CF.rows(); ++i)
	{
		Eigen::Vector3i index_vector = CF.row(i);

		const Eigen::Vector3d t_0 = C.row(index_vector[0]);
		const Eigen::Vector3d t_1 = C.row(index_vector[1]);
		const Eigen::Vector3d t_2 = C.row(index_vector[2]);

		auto const normal = ((t_1 - t_0).cross(t_2 - t_0)).normalized();

		normals.row(i) = normal;
	}
}

double calc_scaling_factor_tri(Eigen::Vector3d old_tri[3], Eigen::Vector3d new_tri[3])
{
	auto const old_u = old_tri[0] - old_tri[1];
	auto const old_v = old_tri[0] - old_tri[2];

	auto const area = .5 * (old_u.cross(old_v)).stableNorm();

	auto const new_u = new_tri[0] - new_tri[1];
	auto const new_v = new_tri[0] - new_tri[2];

	return std::sqrt(new_u.squaredNorm() * old_v.squaredNorm() - 2. * new_u.dot(new_v) * old_u.dot(old_v) + new_v.squaredNorm() * old_u.squaredNorm()) /
		(2.8284271247461903 * area);
}

void calcScalingFactors(const Eigen::MatrixXd& C, const Eigen::MatrixXd& C_deformed, const Eigen::MatrixXi& CF, Eigen::MatrixXd& normals)
{
	for (int i = 0; i < CF.rows(); ++i)
	{
		const Eigen::Vector3i index_vec = CF.row(i);

		Eigen::Vector3d old_tri[3], new_tri[3];
		for (int k = 0; k < 3; ++k)
		{
			old_tri[k] = C.row(index_vec(k));
			new_tri[k] = C_deformed.row(index_vec(k));
		}

		auto const scaling_factor = calc_scaling_factor_tri(old_tri, new_tri);
		normals.row(i) *= scaling_factor;
	}
}

