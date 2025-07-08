#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <greedy-joining.hxx>
#include "greedy-joining-lookahead.hxx"
#include "greedy-joining-cohesion.hxx"
#include "greedy-joining-uniform.hxx"


// wrap as Python module
PYBIND11_MODULE(greedy_joining, m)
{
    m.def("greedy_joining", &greedy_joining<double>); 
    m.def("greedy_joining_lookahead", &lookahead::greedy_joining_lookahead<double>);
    m.def("greedy_joining_cohesion", &cohesion::greedy_joining_cohesion<double>);
    m.def("greedy_joining_uniform", &uniform::greedy_joining_uniform<double>);
}