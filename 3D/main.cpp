#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include <iostream>

#include <string>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>
#include <initializer_list>
#include <cstddef>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <cagedeformations/GreenCoordinates.h>
#include <cagedeformations/LoadMesh.h>
#include <cagedeformations/LoadConstraints.h>

#include <igl/writeOBJ.h>
#include <igl/Timer.h>

namespace fs
{
	class path
	{
	public:
		std::string value;

		path() {}
		path(const std::string& v) : value(v) {}
		path(const char* v) : value(v) {}

		std::string string() const { return value; }
		bool empty() const { return value.empty(); }

		std::string extension() const
		{
			const std::size_t dot = value.find_last_of('.');
			const std::size_t sep = value.find_last_of("\\/");
			if (dot == std::string::npos)
			{
				return std::string();
			}
			if (sep != std::string::npos && dot < sep)
			{
				return std::string();
			}
			return value.substr(dot);
		}

		std::string filename() const
		{
			const std::size_t sep = value.find_last_of("\\/");
			if (sep == std::string::npos)
			{
				return value;
			}
			return value.substr(sep + 1);
		}

		path operator/(const std::string& child) const
		{
			if (value.empty())
			{
				return path(child);
			}
			const char last = value[value.size() - 1];
			if (last == '\\' || last == '/')
			{
				return path(value + child);
			}
			return path(value + "/" + child);
		}

		path operator/(const char* child) const
		{
			return operator/(std::string(child));
		}

		bool operator<(const path& other) const
		{
			return value < other.value;
		}
	};

	inline std::ostream& operator<<(std::ostream& stream, const path& p)
	{
		stream << p.value;
		return stream;
	}

#ifdef _WIN32

	inline bool is_regular_file(const path& p)
	{
		const DWORD attributes = GetFileAttributesA(p.value.c_str());
		return attributes != INVALID_FILE_ATTRIBUTES &&
			(attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
	}

	inline bool is_directory(const path& p)
	{
		const DWORD attributes = GetFileAttributesA(p.value.c_str());
		return attributes != INVALID_FILE_ATTRIBUTES &&
			(attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
	}

	inline std::vector<path> list_directory_files(const path& directory)
	{
		std::vector<path> result;

		std::string pattern = directory.value;
		if (!pattern.empty() && pattern.back() != '\\' && pattern.back() != '/')
		{
			pattern += "\\";
		}
		pattern += "*";

		WIN32_FIND_DATAA data;
		HANDLE handle = FindFirstFileA(pattern.c_str(), &data);
		if (handle == INVALID_HANDLE_VALUE)
		{
			return result;
		}

		do
		{
			const std::string name = data.cFileName;
			if (name == "." || name == "..") continue;
			if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
			result.push_back(directory / name);
		} while (FindNextFileA(handle, &data));

		FindClose(handle);
		return result;
	}

#else // ---------- POSIX ----------

	inline bool is_regular_file(const path& p)
	{
		struct stat st;
		return ::stat(p.value.c_str(), &st) == 0 && S_ISREG(st.st_mode);
	}

	inline bool is_directory(const path& p)
	{
		struct stat st;
		return ::stat(p.value.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
	}

	inline std::vector<path> list_directory_files(const path& directory)
	{
		std::vector<path> result;

		DIR* dir = ::opendir(directory.value.c_str());
		if (!dir) return result;

		struct dirent* entry = nullptr;
		while ((entry = ::readdir(dir)) != nullptr)
		{
			const std::string name = entry->d_name;
			if (name == "." || name == "..") continue;

			const path full = directory / name;
			if (is_regular_file(full))
			{
				result.push_back(full);
			}
		}
		::closedir(dir);
		return result;
	}

#endif
}

struct CommandLineOptions
{
	std::string mode;
	fs::path input_directory;
	fs::path output_file;

	float scaling_factor = 1.0f;

	double lambda_smooth = 0.05;
	double lambda_user = 100.0;
	double lambda_change = 0.001;
	double lambda_rigid = 1.0;

	int boundary_samples_per_face = 2;
	double boundary_offset = 0.05;

	int max_iterations = 100;
	double tolerance = 1e-4;
	std::vector<double> anisotropy_values;
	bool help = false;
	bool measure_time = false;
};

static bool parse_double_value(const std::string& text, double& value)
{
	try
	{
		std::size_t position = 0;
		const double parsed = std::stod(text, &position);

		if (position != text.size())
		{
			return false;
		}

		value = parsed;
		return true;
	}
	catch (...)
	{
		return false;
	}
}

static bool parse_float_value(const std::string& text, float& value)
{
	double parsed = 0.0;

	if (!parse_double_value(text, parsed))
	{
		return false;
	}

	value = static_cast<float>(parsed);
	return true;
}

static bool parse_int_value(const std::string& text, int& value)
{
	try
	{
		std::size_t position = 0;
		const int parsed = std::stoi(text, &position);

		if (position != text.size())
		{
			return false;
		}

		value = parsed;
		return true;
	}
	catch (...)
	{
		return false;
	}
}

static std::string to_lower(std::string value)
{
	std::transform(
		value.begin(),
		value.end(),
		value.begin(),
		[](unsigned char character)
		{
			return static_cast<char>(std::tolower(character));
		});

	return value;
}

static bool has_extension(
	const fs::path& path,
	const std::initializer_list<std::string>& extensions)
{
	const std::string extension = to_lower(path.extension());

	for (const std::string& candidate : extensions)
	{
		if (extension == candidate)
		{
			return true;
		}
	}

	return false;
}

static fs::path find_file_by_name(
	const fs::path& directory,
	const std::vector<std::string>& names)
{
	for (const std::string& name : names)
	{
		const fs::path candidate = directory / name;

		if (fs::is_regular_file(candidate))
		{
			return candidate;
		}
	}

	return fs::path();
}

static fs::path find_model_file(const fs::path& directory)
{
	const std::vector<std::string> preferred_names = {
		"Model.obj",
		"model.obj",
		"Input.obj",
		"input.obj",
		"Object.obj",
		"object.obj"
	};

	const fs::path preferred = find_file_by_name(directory, preferred_names);
	if (!preferred.empty())
	{
		return preferred;
	}

	std::vector<fs::path> candidates;

	for (const fs::path& entry : fs::list_directory_files(directory))
	{
		if (!fs::is_regular_file(entry))
		{
			continue;
		}

		if (!has_extension(entry, { ".obj", ".msh" }))
		{
			continue;
		}

		const std::string filename = to_lower(entry.filename());

		if (filename == "cage.obj" ||
			filename == "cage_deformed.obj" ||
			filename == "cage-deformed.obj")
		{
			continue;
		}

		candidates.push_back(entry);
	}

	std::sort(candidates.begin(), candidates.end());

	if (candidates.empty())
	{
		return fs::path();
	}

	return candidates.front();
}

static fs::path find_constraints_file(const fs::path& directory)
{
	const std::vector<std::string> preferred_names = {
		"Constraints.txt",
		"constraints.txt",
		"VariationalConstraints.txt",
		"variational_constraints.txt",
		"constraints.dat",
		"constraints.constraint"
	};

	const fs::path preferred = find_file_by_name(directory, preferred_names);
	if (!preferred.empty())
	{
		return preferred;
	}

	std::vector<fs::path> candidates;

	for (const fs::path& entry : fs::list_directory_files(directory))
	{
		if (!fs::is_regular_file(entry))
		{
			continue;
		}

		if (has_extension(entry, { ".txt", ".dat", ".constraint" }))
		{
			candidates.push_back(entry);
		}
	}

	std::sort(candidates.begin(), candidates.end());

	if (candidates.empty())
	{
		return fs::path();
	}

	return candidates.front();
}

static void print_help(const char* executable)
{
	std::cout
		<< "Usage:\n"
		<< " " << executable << " cage-based <input_directory> [options]\n"
		<< " " << executable << " variational <input_directory> [options]\n\n"
		<< "Common input files:\n"
		<< " Cage.obj\n"
		<< " Model.obj, Input.obj, Object.obj, or the first remaining OBJ file\n\n"
		<< "Cage-based additional input files:\n"
		<< " Cage_Deformed.obj\n\n"
		<< "Variational input files:\n"
		<< " Constraints.txt, or the first TXT/DAT/CONSTRAINT file\n\n"
		<< "Options:\n"
		<< " -h, --help Show this help message\n"
		<< " -o, --output Output OBJ path\n"
		<< " --scale Mesh scale factor, default: 1.0\n"
		<< " --lambda-smooth Hessian smoothness weight, default: 0.05\n"
		<< " --lambda-user User constraint weight, default: 100.0\n"
		<< " --lambda-change Iteration change weight, default: 0.001\n"
		<< " --lambda-rigid ARAP Jacobian weight, default: 1.0\n"
		<< " --boundary-samples Samples per cage face, default: 2\n"
		<< " --boundary-offset Interior sample offset, default: 0.05\n"
		<< " --iterations Maximum iterations, default: 100\n"
		<< " --tolerance Convergence tolerance, default: 1e-4\n"
		<< " --time Measure coordinate computation time\n"
		<< " -A, --matrix 9 values in row-major order, default: identity 3x3\n";
}

static std::string require_value(
	int& index,
	int argc,
	char** argv,
	const std::string& option)
{
	if (index + 1 >= argc)
	{
		throw std::runtime_error("Missing value for option: " + option);
	}

	++index;
	return argv[index];
}

static CommandLineOptions parse_command_line(int argc, char** argv)
{
	CommandLineOptions options;

	if (argc <= 1)
	{
		options.help = true;
		return options;
	}

	for (int index = 1; index < argc; ++index)
	{
		const std::string argument = argv[index];

		if (argument == "-h" || argument == "--help")
		{
			options.help = true;
		}
		else if (argument == "cage-based" ||
			argument == "cage_based" ||
			argument == "a_green")
		{
			options.mode = "cage-based";
		}
		else if (argument == "variational" ||
			argument == "va_green")
		{
			options.mode = "variational";
		}
		else if (argument == "-o" || argument == "--output")
		{
			options.output_file = require_value(
				index,
				argc,
				argv,
				argument);
		}
		else if (argument == "--scale")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_float_value(value, options.scaling_factor))
			{
				throw std::runtime_error(
					"Invalid value for --scale: " + value);
			}
		}
		else if (argument == "--lambda-smooth")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_double_value(value, options.lambda_smooth))
			{
				throw std::runtime_error(
					"Invalid value for --lambda-smooth: " + value);
			}
		}
		else if (argument == "--lambda-user")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_double_value(value, options.lambda_user))
			{
				throw std::runtime_error(
					"Invalid value for --lambda-user: " + value);
			}
		}
		else if (argument == "--lambda-change")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_double_value(value, options.lambda_change))
			{
				throw std::runtime_error(
					"Invalid value for --lambda-change: " + value);
			}
		}
		else if (argument == "--lambda-rigid")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_double_value(value, options.lambda_rigid))
			{
				throw std::runtime_error(
					"Invalid value for --lambda-rigid: " + value);
			}
		}
		else if (argument == "--boundary-samples")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_int_value(value, options.boundary_samples_per_face))
			{
				throw std::runtime_error(
					"Invalid value for --boundary-samples: " + value);
			}
		}
		else if (argument == "--boundary-offset")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_double_value(value, options.boundary_offset))
			{
				throw std::runtime_error(
					"Invalid value for --boundary-offset: " + value);
			}
		}
		else if (argument == "--iterations")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_int_value(value, options.max_iterations))
			{
				throw std::runtime_error(
					"Invalid value for --iterations: " + value);
			}
		}
		else if (argument == "--tolerance")
		{
			const std::string value = require_value(
				index,
				argc,
				argv,
				argument);

			if (!parse_double_value(value, options.tolerance))
			{
				throw std::runtime_error(
					"Invalid value for --tolerance: " + value);
			}
		}
		else if (argument == "--time")
		{
			options.measure_time = true;
		}
		else if (argument == "-A" || argument == "--matrix")
		{
			if (index + 9 >= argc)
			{
				throw std::runtime_error(
					"Option " + argument + " requires 9 numeric values");
			}

			options.anisotropy_values.clear();
			options.anisotropy_values.reserve(9);

			for (int k = 0; k < 9; ++k)
			{
				++index;
				double v = 0.0;
				if (!parse_double_value(argv[index], v))
				{
					throw std::runtime_error(
						std::string("Invalid value for ") + argument +
						" (element " + std::to_string(k) + "): " + argv[index]);
				}
				options.anisotropy_values.push_back(v);
			}
			}
		else if (options.input_directory.empty())
		{
			options.input_directory = argument;
		}
		else
		{
			throw std::runtime_error(
				"Unknown option or unexpected argument: " + argument);
		}
	}

	return options;
}

static bool validate_options(const CommandLineOptions& options)
{
	if (options.mode.empty())
	{
		std::cerr << "Missing deformation mode. Use cage-based or variational.\n";
		return false;
	}

	if (options.input_directory.empty())
	{
		std::cerr << "Missing input directory.\n";
		return false;
	}

	if (!fs::is_directory(options.input_directory))
	{
		std::cerr
			<< "Input path is not a directory: "
			<< options.input_directory << "\n";
		return false;
	}

	if (options.scaling_factor <= 0.0f)
	{
		std::cerr << "--scale must be positive.\n";
		return false;
	}

	if (options.lambda_smooth < 0.0 ||
		options.lambda_user < 0.0 ||
		options.lambda_change < 0.0 ||
		options.lambda_rigid < 0.0)
	{
		std::cerr << "Lambda values must be non-negative.\n";
		return false;
	}

	if (options.boundary_samples_per_face <= 0)
	{
		std::cerr << "--boundary-samples must be positive.\n";
		return false;
	}

	if (options.boundary_offset < 0.0)
	{
		std::cerr << "--boundary-offset must be non-negative.\n";
		return false;
	}

	if (options.max_iterations <= 0)
	{
		std::cerr << "--iterations must be positive.\n";
		return false;
	}

	if (options.tolerance <= 0.0)
	{
		std::cerr << "--tolerance must be positive.\n";
		return false;
	}

	return true;
}

int main(int argc, char** argv)
{
	CommandLineOptions options;

	try
	{
		options = parse_command_line(argc, argv);
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << "\n\n";
		print_help(argv[0]);
		return 1;
	}

	if (options.help)
	{
		print_help(argv[0]);
		return 0;
	}

	if (!validate_options(options))
	{
		print_help(argv[0]);
		return 1;
	}

	const bool cage_based = options.mode == "cage-based";
	const bool variational = options.mode == "variational";

	const fs::path input_directory = options.input_directory;

	const fs::path cage_file = find_file_by_name(
		input_directory,
		{ "Cage.obj", "cage.obj" });

	const fs::path model_file = find_model_file(input_directory);

	if (cage_file.empty())
	{
		std::cerr
			<< "Could not find Cage.obj in "
			<< input_directory << "\n";
		return 1;
	}

	if (model_file.empty())
	{
		std::cerr
			<< "Could not find an input model OBJ/MSH file in "
			<< input_directory << "\n";
		return 1;
	}

	// Cage_Deformed.obj is only needed for cage-based deformation.
	fs::path deformed_cage_file;

	if (cage_based)
	{
		deformed_cage_file = find_file_by_name(
			input_directory,
			{
				"Cage_Deformed.obj",
				"cage_deformed.obj",
				"Cage-Deformed.obj",
				"cage-deformed.obj"
			});

		if (deformed_cage_file.empty())
		{
			std::cerr
				<< "Could not find Cage_Deformed.obj in "
				<< input_directory << "\n";
			return 1;
		}
	}

	// Variational mode requires a constraints file instead.
	const fs::path constraints_file = variational
		? find_constraints_file(input_directory)
		: fs::path();

	if (variational && constraints_file.empty())
	{
		std::cerr
			<< "Could not find a variational constraints file in "
			<< input_directory << "\n";
		return 1;
	}

	std::cout
		<< "Input directory: " << input_directory << "\n"
		<< "Model: " << model_file << "\n"
		<< "Source cage: " << cage_file << "\n";

	if (cage_based)
	{
		std::cout
			<< "Deformed cage: "
			<< deformed_cage_file
			<< "\n";
	}

	if (variational)
	{
		std::cout
			<< "Constraints: "
			<< constraints_file
			<< "\n";
	}

	std::unique_ptr<igl::Timer> timer;

	if (options.measure_time)
	{
		timer = std::make_unique<igl::Timer>();
	}

	auto start_timer = [&]()
		{
			if (timer)
			{
				timer->start();
			}
		};

	auto stop_timer = [&]()
		{
			if (timer)
			{
				timer->stop();
			}
		};

	Eigen::MatrixXd model_vertices;
	Eigen::MatrixXi model_faces;
	Eigen::MatrixXd cage_vertices;
	Eigen::MatrixXd deformed_cage_vertices;
	Eigen::MatrixXi cage_faces;
	Eigen::MatrixXi deformed_cage_faces;
	Eigen::VectorXi cage_boundary;
	Eigen::VectorXi deformed_cage_boundary;

	if (!load_mesh(
		model_file.string(),
		model_vertices,
		model_faces,
		options.scaling_factor))
	{
		std::cerr
			<< "Failed to load model: "
			<< model_file << "\n";
		return 1;
	}

	if (!load_cage(
		cage_file.string(),
		cage_vertices,
		cage_boundary,
		cage_faces,
		options.scaling_factor,
		false))
	{
		std::cerr
			<< "Failed to load source cage: "
			<< cage_file << "\n";
		return 1;
	}

	if (cage_based)
	{
		if (!load_cage(
			deformed_cage_file.string(),
			deformed_cage_vertices,
			deformed_cage_boundary,
			deformed_cage_faces,
			options.scaling_factor,
			false))
		{
			std::cerr
				<< "Failed to load deformed cage: "
				<< deformed_cage_file << "\n";
			return 1;
		}

		if (cage_vertices.rows() != deformed_cage_vertices.rows())
		{
			std::cerr
				<< "Source and deformed cages have different vertex counts.\n";
			return 1;
		}

		if (cage_faces.rows() != deformed_cage_faces.rows())
		{
			std::cerr
				<< "Source and deformed cages have different face counts.\n";
			return 1;
		}
	}

	Eigen::MatrixXd anisotropy_matrix(3, 3);

	if (options.anisotropy_values.size() == 9)
	{
		anisotropy_matrix <<
			options.anisotropy_values[0], options.anisotropy_values[1], options.anisotropy_values[2],
			options.anisotropy_values[3], options.anisotropy_values[4], options.anisotropy_values[5],
			options.anisotropy_values[6], options.anisotropy_values[7], options.anisotropy_values[8];
	}
	else
	{
		anisotropy_matrix <<
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(
		anisotropy_matrix);

	if (eigen_solver.info() != Eigen::Success)
	{
		std::cerr << "Failed to decompose anisotropy matrix.\n";
		return 1;
	}

	Eigen::MatrixXd anisotropy_square_root =
		eigen_solver.operatorSqrt();

	Eigen::MatrixXd anisotropy_inverse_square_root =
		anisotropy_square_root.inverse();

	Eigen::MatrixXd source_normals;
	calcNormals(cage_vertices, cage_faces, source_normals);

	Eigen::MatrixXd coordinates_phi;
	Eigen::MatrixXd coordinates_psi;

	start_timer();

	calculateAGC(
		cage_vertices,
		cage_faces,
		source_normals,
		model_vertices,
		coordinates_phi,
		coordinates_psi,
		anisotropy_matrix,
		anisotropy_square_root,
		anisotropy_inverse_square_root);

	stop_timer();

	if (timer)
	{
		std::cout
			<< "Coordinate computation time: "
			<< timer->getElapsedTime()
			<< " seconds\n";
	}

	Eigen::MatrixXd output_vertices;

	if (cage_based)
	{
		Eigen::MatrixXd deformed_normals;
		calcNormals(
			deformed_cage_vertices,
			deformed_cage_faces,
			deformed_normals);

		for (int face_index = 0;
			face_index < deformed_normals.rows();
			++face_index)
		{
			const Eigen::Vector3d normal =
				deformed_normals.row(face_index);

			deformed_normals.row(face_index) =
				(anisotropy_matrix * normal).transpose();
		}

		calcScalingFactors(
			cage_vertices,
			deformed_cage_vertices,
			cage_faces,
			deformed_normals);

		output_vertices =
			coordinates_phi.transpose() * deformed_cage_vertices +
			coordinates_psi.transpose() * deformed_normals;
	}
	else if (variational)
	{
		Eigen::MatrixXd user_constraints_origin;
		Eigen::MatrixXd user_constraints_deformed;
		Eigen::MatrixXd medial_points;

		if (!loadConstraintsFromFile(
			constraints_file.string(),
			user_constraints_origin,
			user_constraints_deformed,
			medial_points))
		{
			std::cerr
				<< "Failed to load constraints file: "
				<< constraints_file << "\n";
			return 1;
		}

		Eigen::MatrixXd boundary_samples =
			generateBoundarySamples(
				cage_vertices,
				cage_faces,
				options.boundary_samples_per_face,
				options.boundary_offset);

		VariationalShapeDeformation3D deformation(
			cage_vertices,
			cage_faces,
			anisotropy_matrix);

		VariationalDeformationSolver3D solver(deformation);

		solver.lambda_ = options.lambda_smooth;
		solver.lambda_user_ = options.lambda_user;
		solver.lambda_change_ = options.lambda_change;
		solver.lambda_rigid_ = options.lambda_rigid;
		solver.max_iter_ = options.max_iterations;
		solver.tol_ = options.tolerance;

		solver.solve(
			user_constraints_origin,
			user_constraints_deformed,
			medial_points,
			boundary_samples,
			anisotropy_matrix);

		const Eigen::MatrixXd final_cage_vertices =
			solver.getDeformedCageVertices();

		const Eigen::MatrixXd final_face_coefficients =
			solver.getDeformedFaceCoeffs();

		output_vertices =
			coordinates_phi.transpose() * final_cage_vertices +
			coordinates_psi.transpose() * final_face_coefficients;
	}

	fs::path output_file = options.output_file;

	if (output_file.empty())
	{
		output_file =
			input_directory /
			(cage_based
				? "Result_CageBased.obj"
				: "Result_Variational.obj");
	}

	if (!igl::writeOBJ(
		output_file.string(),
		output_vertices,
		model_faces))
	{
		std::cerr
			<< "Failed to write output mesh: "
			<< output_file << "\n";
		return 1;
	}

	std::cout
		<< "Output mesh: "
		<< output_file << "\n";

	return 0;
}