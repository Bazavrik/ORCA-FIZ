#include <pybind11/pybind11.h>
#include "fuzzy_sd.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fuzzy_sd_cpp, m) {
    m.doc() = "Fuzzy_SD Mamdani FIS (C++)";

    py::class_<FuzzySD>(m, "FuzzySD")
        .def(py::init<std::size_t>(), py::arg("grid_n") = 301)
        .def("eval", &FuzzySD::eval,
             py::arg("V"), py::arg("d"), py::arg("fi"), py::arg("C"),
             "Evaluate FIS, returns 0..1");
}