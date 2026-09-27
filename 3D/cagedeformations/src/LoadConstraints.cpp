#include <cagedeformations/LoadConstraints.h>
#include <cagedeformations/LoadMesh.h>
#include <cagedeformations/GreenCoordinates.h>
#include <Eigen/Dense>


bool loadConstraintsFromFile(const std::string& filepath,
    Eigen::MatrixXd& user_constraints_origin,
    Eigen::MatrixXd& user_constraints_deformed,
    Eigen::MatrixXd& medial_points) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << filepath << std::endl;
        return false;
    }

    
    std::vector<std::vector<double>> origin_data;
    std::vector<std::vector<double>> deformed_data;
    std::vector<std::vector<double>> medial_data;

    std::string line;
    int section = 0; 

    while (std::getline(file, line)) {
        
        if (line.empty() || line[0] == '#') continue;

       
        if (line.find("user_constraints_origin") != std::string::npos) {
            section = 1;
            continue;
        }
        else if (line.find("user_constraints_deformed") != std::string::npos) {
            section = 2;
            continue;
        }
        else if (line.find("medial_points") != std::string::npos) {
            section = 3;
            continue;
        }

       
        std::istringstream iss(line);
        std::vector<double> row;
        double value;

        while (iss >> value) {
            row.push_back(value);
        }

        if (row.empty()) continue;

       
        switch (section) {
        case 1:
            origin_data.push_back(row);
            break;
        case 2:
            deformed_data.push_back(row);
            break;
        case 3:
            medial_data.push_back(row);
            break;
        default:
            break;
        }
    }

    file.close();

   
    user_constraints_origin.resize(origin_data.size(), 3);
    for (size_t i = 0; i < origin_data.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            user_constraints_origin(i, j) = origin_data[i][j];
        }
    }

    user_constraints_deformed.resize(deformed_data.size(), 3);
    for (size_t i = 0; i < deformed_data.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            user_constraints_deformed(i, j) = deformed_data[i][j];
        }
    }

    medial_points.resize(medial_data.size(), 3);
    for (size_t i = 0; i < medial_data.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            medial_points(i, j) = medial_data[i][j];
        }
    }

    return true;
}


Eigen::Vector3d samplePointInTriangle(const Eigen::Vector3d& v0,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    std::mt19937& rng) 
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r1 = dist(rng);
    double r2 = dist(rng);

    
    if (r1 + r2 > 1.0) {
        r1 = 1.0 - r1;
        r2 = 1.0 - r2;
    }

    return v0 + r1 * (v1 - v0) + r2 * (v2 - v0);
}

double solidAngleTriangle(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& v0,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2)
{
    const Eigen::Vector3d a = v0 - p;
    const Eigen::Vector3d b = v1 - p;
    const Eigen::Vector3d c = v2 - p;
    const double al = a.norm();
    const double bl = b.norm();
    const double cl = c.norm();
    const double eps = 1e-14;
    if (al < eps || bl < eps || cl < eps) {
       
        return 4.0 * 3.14159265358979323846;
    }


    const double numerator = a.dot(b.cross(c)); 
    const double denom =
        al * bl * cl
        + a.dot(b) * cl
        + b.dot(c) * al
        + c.dot(a) * bl;

    
    return 2.0 * std::atan2(numerator, denom);
}

bool isPointInsideCage(
    const Eigen::Vector3d& point,
    const Eigen::MatrixXd& C,     // #V x 3 vertices
    const Eigen::MatrixXi& CF)    // #F x 3 faces (indices into C)
{
    constexpr double PI = 3.14159265358979323846;
    double omega_sum = 0.0;

    for (int i = 0; i < CF.rows(); ++i) {
        const Eigen::Vector3i f = CF.row(i);
        const Eigen::Vector3d v0 = C.row(f[0]);
        const Eigen::Vector3d v1 = C.row(f[1]);
        const Eigen::Vector3d v2 = C.row(f[2]);

        const double omega = solidAngleTriangle(point, v0, v1, v2);

        // Optional: if point is on/very near surface, early return true
        if (std::abs(omega) > 3.0 * PI) { 
            return true;
        }
        omega_sum += omega;
    }
    const double winding = omega_sum / (4.0 * PI);
    return winding > 0.5;
}

void saveBoundarySamplesToXYZ(const Eigen::MatrixXd& boundary_samples,
    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot create: " << filename << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(6);  
    for (int i = 0; i < boundary_samples.rows(); ++i) {
        file << boundary_samples(i, 0) << " "
            << boundary_samples(i, 1) << " "
            << boundary_samples(i, 2) << "\n";
    }
    file.close();
    std::cout << "Save in: " << filename
        << " ( " << boundary_samples.rows() << " boundary points)" << std::endl;
}


Eigen::MatrixXd generateBoundarySamples(const Eigen::MatrixXd& C,
    const Eigen::MatrixXi& CF,
    int samples_per_face = 3,
    double offset = 0.02) {
    Eigen::MatrixXd normals;
    calcNormals(C, CF, normals);
    std::vector<Eigen::Vector3d> samples;
    for (int i = 0; i < CF.rows(); ++i) {
        Eigen::Vector3i index_vector = CF.row(i);
        const Eigen::Vector3d v0 = C.row(index_vector[0]);
        const Eigen::Vector3d v1 = C.row(index_vector[1]);
        const Eigen::Vector3d v2 = C.row(index_vector[2]);
        const Eigen::Vector3d normal = normals.row(i);
        Eigen::Vector3d edge1 = v1 - v0;
        Eigen::Vector3d edge2 = v2 - v0;
        int grid_size = static_cast<int>(std::sqrt(samples_per_face)) + 1;
        for (int u_idx = 1; u_idx < grid_size; ++u_idx) 
        {
            for (int v_idx = 1; v_idx < grid_size - u_idx; ++v_idx) 
            {              
                double u = static_cast<double>(u_idx) / grid_size;
                double v = static_cast<double>(v_idx) / grid_size;
                if (u + v > 1.0) continue;
                double w = 1.0 - u - v;
                Eigen::Vector3d sampled_point = v0 * u + v1 * v + v2 * w;
                Eigen::Vector3d offset_point = sampled_point - normal * offset;
                if (isPointInsideCage(offset_point, C, CF)) {
                    samples.push_back(offset_point);
                }
            }
        }
   
        if (samples.size() < (i + 1) * samples_per_face) {
            
            Eigen::Vector3d centroid = (v0 + v1 + v2) / 3.0;
            Eigen::Vector3d offset_centroid = centroid - normal * offset;

            if (isPointInsideCage(offset_centroid, C, CF)) {
                samples.push_back(offset_centroid);
            }
        }
    }

    if (samples.size() > CF.rows() * samples_per_face) {
        std::vector<Eigen::Vector3d> downsampled;
        double step = static_cast<double>(samples.size()) / (CF.rows() * samples_per_face);
        for (double idx = 0; idx < samples.size(); idx += step) {
            downsampled.push_back(samples[static_cast<int>(idx)]);
        }
        samples = downsampled;
    }

    Eigen::MatrixXd boundary_samples(samples.size(), 3);
    for (size_t i = 0; i < samples.size(); ++i) {
        boundary_samples.row(i) = samples[i];
       
    }
    return boundary_samples;
}



bool isPointOnSameSide(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    Eigen::Vector3d cp1 = (b - a).cross(p1 - a);
    Eigen::Vector3d cp2 = (b - a).cross(p2 - a);
    return cp1.dot(cp2) >= 0;
}


bool isPointInTetrahedron(const Eigen::Vector3d& point,
    const Eigen::Vector3d& a, const Eigen::Vector3d& b,
    const Eigen::Vector3d& c, const Eigen::Vector3d& d) {
    auto signedVolume = [](const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3, const Eigen::Vector3d& p4) -> double {
            Eigen::Matrix4d M;
            M << p1[0], p1[1], p1[2], 1.0,
                p2[0], p2[1], p2[2], 1.0,
                p3[0], p3[1], p3[2], 1.0,
                p4[0], p4[1], p4[2], 1.0;
            return M.determinant() / 6.0;
        };
    double v0 = signedVolume(a, b, c, d);
    double v1 = signedVolume(point, b, c, d);
    double v2 = signedVolume(a, point, c, d);
    double v3 = signedVolume(a, b, point, d);
    double v4 = signedVolume(a, b, c, point);
    return (v0 * v1 >= 0) && (v0 * v2 >= 0) && (v0 * v3 >= 0) && (v0 * v4 >= 0);
}