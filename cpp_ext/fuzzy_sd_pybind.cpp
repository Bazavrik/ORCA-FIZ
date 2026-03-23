#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "fuzzy_sd.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fuzzy_sd_cpp, m) {
    m.doc() = "Fuzzy_SD Mamdani FIS (C++) with batch evaluation";

    py::class_<FuzzySD>(m, "FuzzySD")
        .def(py::init<std::size_t>(), py::arg("grid_n") = 301)
        .def("eval", &FuzzySD::eval,
             py::arg("V"), py::arg("d"), py::arg("fi"), py::arg("C"))
        .def("eval_batch", [](const FuzzySD& self, py::array_t<double, py::array::c_style | py::array::forcecast> x) {
            auto buf = x.request();
            if (buf.ndim != 2 || buf.shape[1] != 4) {
                throw std::runtime_error("eval_batch expects array of shape (N,4)");
            }
            const auto rows = static_cast<std::size_t>(buf.shape[0]);
            const auto cols = static_cast<std::size_t>(buf.shape[1]);
            const auto* ptr = static_cast<const double*>(buf.ptr);
            auto out = self.eval_batch_flat(ptr, rows, cols);
            py::array_t<double> y(rows);
            auto ybuf = y.request();
            auto* yptr = static_cast<double*>(ybuf.ptr);
            for (std::size_t i = 0; i < rows; ++i) yptr[i] = out[i];
            return y;
        }, py::arg("X"));
}
