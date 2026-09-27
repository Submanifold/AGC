#pragma once
#include <random>       
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Dense>
bool loadConstraintsFromFile(const std::string& filepath, Eigen::MatrixXd& user_constraints_origin,
    Eigen::MatrixXd& user_constraints_deformed, Eigen::MatrixXd& medial_points);
bool isPointInsideCage(
    const Eigen::Vector3d& point,
    const Eigen::MatrixXd& C,  
    const Eigen::MatrixXi& CF);
double solidAngleTriangle(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& v0,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2);
Eigen::MatrixXd generateBoundarySamples(const Eigen::MatrixXd& C,
    const Eigen::MatrixXi& CF,
    int samples_per_face,
    double offset);
void saveBoundarySamplesToXYZ(const Eigen::MatrixXd& boundary_samples,
    const std::string& filename);