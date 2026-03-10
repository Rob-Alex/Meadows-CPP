/*
  Meadows CPP
  Robbie Alexander 
  exporter.hpp
  HDF5 + XDMF exporter for VisIt / ParaView visualisation.

  Writes cell-centred data on uniform grids. Each export produces:
    - an HDF5 file  (binary data: component arrays)
    - an XDMF file  (XML metadata: grid geometry, references into HDF5)

  VisIt/ParaView open the .xmf file and find everything they need.

  Usage pattern (from solver or main):
    GridExporter<double, 2> exporter("build/outputs", "poisson");
    exporter.write(hierarchy, bc_registry, {"phi", "rhs", "res"}, step);

  This writes:
    build/outputs/poisson_000000.h5
    build/outputs/poisson_000000.xmf
*/
#pragma once
#include "grid.hpp"
#include "operators.hpp"
#include <H5Cpp.h>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <sstream>
#include <iomanip>

template<typename T, int Dims, typename Alloc = HostAllocator<T, default_tracking<T>>>
class GridExporter {
private:
  std::string _output_dir;
  std::string _prefix;

public:
  GridExporter(const std::string& output_dir, const std::string& prefix)
    : _output_dir(output_dir), _prefix(prefix) {
    std::filesystem::create_directories(_output_dir);
  }

  // Write a snapshot of specified components from the finest level.
  // comp_names: list of registered component names to export.
  // step: iteration number (used for filename numbering).
  void write(GridHierarchy<T, Dims, Alloc>& hierarchy,
             const std::vector<std::string>& comp_names,
             int step) {
    // build filenames
    std::ostringstream ss;
    ss << _prefix << "_" << std::setw(6) << std::setfill('0') << step;
    std::string base = ss.str();
    std::string h5_path = _output_dir + "/" + base + ".h5";
    std::string xmf_path = _output_dir + "/" + base + ".xmf";
    std::string h5_filename = base + ".h5";

    auto& level = hierarchy.finest();
    const auto& geom = level.geometry();
    
    write_hdf5(hierarchy, level, geom, comp_names, h5_path);
    write_xdmf(geom, comp_names, h5_filename, xmf_path);

    std::printf("  Exported step %d → %s\n", step, xmf_path.c_str());
  }

  // Write a time-series master XDMF that groups all steps.
  // VisIt can open this single file and scrub through time.
  void write_time_series(int n_steps,
                         const GridGeometry<T, Dims>& geom,
                         const std::vector<std::string>& comp_names) {
    std::string xmf_path = _output_dir + "/" + _prefix + ".xmf";
    std::ofstream f(xmf_path);

    f << "<?xml version=\"1.0\" ?>\n";
    f << "<Xdmf Version=\"3.0\">\n";
    f << "  <Domain>\n";
    f << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

    for (int step = 0; step < n_steps; ++step) {
      std::ostringstream ss;
      ss << _prefix << "_" << std::setw(6) << std::setfill('0') << step;
      std::string base = ss.str();
      std::string h5_filename = base + ".h5";

      f << "      <Grid Name=\"step_" << step << "\" GridType=\"Uniform\">\n";
      f << "        <Time Value=\"" << step << "\" />\n";
      write_xdmf_grid_body(f, geom, comp_names, h5_filename);
      f << "      </Grid>\n";
    }

    f << "    </Grid>\n";
    f << "  </Domain>\n";
    f << "</Xdmf>\n";
    f.close();
    std::printf("  Time series → %s\n", xmf_path.c_str());
  }

private:
  // Map C++ type to HDF5 type
  static const H5::PredType& h5_type() {
    if constexpr (std::is_same_v<T, double>) {
      return H5::PredType::NATIVE_DOUBLE;
    } else if constexpr (std::is_same_v<T, float>) {
      return H5::PredType::NATIVE_FLOAT;
    } else {
      static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                    "Unsupported type for HDF5 export");
      return H5::PredType::NATIVE_DOUBLE;
    }
  }

  // Map C++ type to XDMF type name
  static const char* xdmf_number_type() {
    if constexpr (std::is_same_v<T, double>) {
      return "Float";
    } else {
      return "Float";
    }
  }

  static int xdmf_precision() {
    return static_cast<int>(sizeof(T));
  }

  void write_hdf5(GridHierarchy<T, Dims, Alloc>& hierarchy,
                   Level<T, Dims, Alloc>& level,
                   const GridGeometry<T, Dims>& geom,
                   const std::vector<std::string>& comp_names,
                   const std::string& h5_path) {
    H5::H5File file(h5_path, H5F_ACC_TRUNC);

    // write geometry attributes
    {
      hsize_t dims_attr[1] = {static_cast<hsize_t>(Dims)};
      H5::DataSpace attr_space(1, dims_attr);

      // origin
      std::vector<T> origin(geom._origin.begin(), geom._origin.end());
      auto origin_attr = file.createAttribute("origin", h5_type(), attr_space);
      origin_attr.write(h5_type(), origin.data());

      // spacing
      std::vector<T> spacing(geom._spacing.begin(), geom._spacing.end());
      auto spacing_attr = file.createAttribute("spacing", h5_type(), attr_space);
      spacing_attr.write(h5_type(), spacing.data());

      // n_interior
      std::vector<int> ni(geom._n_interior.begin(), geom._n_interior.end());
      auto ni_attr = file.createAttribute("n_interior", H5::PredType::NATIVE_INT, attr_space);
      ni_attr.write(H5::PredType::NATIVE_INT, ni.data());
    }

    // write each component as a dataset
    // data is stored in row-major order (C order), interior cells only
    int n_total = geom.total_interior_cells();

    for (const auto& name : comp_names) {
      int comp_idx = hierarchy.get_component_index(name);
      auto acc = level.accessor(comp_idx);

      // collect interior data into a flat buffer (row-major)
      std::vector<T> buffer(n_total);
      int flat = 0;
      std::array<int, Dims> idx{};
      InteriorLoop<Dims, 0>::run(geom._n_interior, idx,
        [&](const std::array<int, Dims>& cell) {
          buffer[flat++] = acc[cell];
        });

      // write as a Dims-dimensional dataset
      std::vector<hsize_t> h5_dims(Dims);
      for (int d = 0; d < Dims; ++d) {
        h5_dims[d] = static_cast<hsize_t>(geom._n_interior[d]);
      }
      H5::DataSpace dataspace(Dims, h5_dims.data());
      auto dataset = file.createDataSet(name, h5_type(), dataspace);
      dataset.write(buffer.data(), h5_type());
    }

    file.close();
  }

  void write_xdmf(const GridGeometry<T, Dims>& geom,
                   const std::vector<std::string>& comp_names,
                   const std::string& h5_filename,
                   const std::string& xmf_path) {
    std::ofstream f(xmf_path);
    f << "<?xml version=\"1.0\" ?>\n";
    f << "<Xdmf Version=\"3.0\">\n";
    f << "  <Domain>\n";
    f << "    <Grid Name=\"mesh\" GridType=\"Uniform\">\n";
    write_xdmf_grid_body(f, geom, comp_names, h5_filename);
    f << "    </Grid>\n";
    f << "  </Domain>\n";
    f << "</Xdmf>\n";
    f.close();
  }

  void write_xdmf_grid_body(std::ofstream& f,
                             const GridGeometry<T, Dims>& geom,
                             const std::vector<std::string>& comp_names,
                             const std::string& h5_filename) {
    // topology: structured grid with N+1 nodes per dimension
    // (cell-centred data has N cells, N+1 nodes)
    f << "        <Topology TopologyType=\"";
    if constexpr (Dims == 2) {
      f << "2DCoRectMesh";
    } else if constexpr (Dims == 3) {
      f << "3DCoRectMesh";
    } else {
      f << "1DCoRectMesh";  // VisIt may not support this, but it's correct
    }
    f << "\" Dimensions=\"";
    // XDMF wants node counts in slowest-to-fastest order
    for (int d = 0; d < Dims; ++d) {
      if (d > 0) f << " ";
      f << (geom._n_interior[d] + 1);
    }
    f << "\"/>\n";

    // geometry: origin + spacing (co-rectangular mesh)
    f << "        <Geometry GeometryType=\"";
    if constexpr (Dims == 2) {
      f << "ORIGIN_DXDY";
    } else if constexpr (Dims == 3) {
      f << "ORIGIN_DXDYDZ";
    } else {
      f << "ORIGIN_DX";
    }
    f << "\">\n";

    // origin (physical coordinates of the first node, not cell centre)
    f << "          <DataItem Dimensions=\"" << Dims << "\" NumberType=\""
      << xdmf_number_type() << "\" Precision=\"" << xdmf_precision()
      << "\" Format=\"XML\">";
    for (int d = 0; d < Dims; ++d) {
      if (d > 0) f << " ";
      // node origin = cell-centre origin - dx/2
      f << (geom._origin[d] - geom._spacing[d] / T{2});
    }
    f << "</DataItem>\n";

    // spacing
    f << "          <DataItem Dimensions=\"" << Dims << "\" NumberType=\""
      << xdmf_number_type() << "\" Precision=\"" << xdmf_precision()
      << "\" Format=\"XML\">";
    for (int d = 0; d < Dims; ++d) {
      if (d > 0) f << " ";
      f << geom._spacing[d];
    }
    f << "</DataItem>\n";

    f << "        </Geometry>\n";

    // attributes (cell-centred data)
    for (const auto& name : comp_names) {
      f << "        <Attribute Name=\"" << name
        << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
      f << "          <DataItem Dimensions=\"";
      for (int d = 0; d < Dims; ++d) {
        if (d > 0) f << " ";
        f << geom._n_interior[d];
      }
      f << "\" NumberType=\"" << xdmf_number_type()
        << "\" Precision=\"" << xdmf_precision()
        << "\" Format=\"HDF\">" << h5_filename << ":/" << name
        << "</DataItem>\n";
      f << "        </Attribute>\n";
    }
  }
};
