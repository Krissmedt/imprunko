#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

#include "../ffe/tile.h"

#include "../ffe/currents/rffe.h"
#include "../ffe/skinny_yee.h"


namespace ffe {

//--------------------------------------------------
template<size_t D>
auto declare_tile(
    py::module& m,
    const std::string& pyclass_name) 
{

  return 
  py::class_<ffe::Tile<D>, 
             fields::Tile<D>,
             corgi::Tile<D>, 
             std::shared_ptr<ffe::Tile<D>>
             >(m, 
               pyclass_name.c_str(),
               py::multiple_inheritance()
               )
    .def(py::init<int, int, int>())
    .def_readwrite("cfl",       &ffe::Tile<D>::cfl);


}

//--------------------------------------------------
//
/// trampoline class for Current solver
//template<size_t D>
//class PyCurrent : public Current<D>
//{
//  using Current<D>::Current;
//
//  void comp_drift_cur( Tile<D>& tile ) override {
//  PYBIND11_OVERLOAD_PURE(
//      void,
//      Current<D>,
//      comp_drift_cur,
//      tile
//      );
//  }
//
//  void comp_parallel_cur( Tile<D>& tile ) override {
//  PYBIND11_OVERLOAD_PURE(
//      void,
//      Current<D>,
//      comp_parallel_cur,
//      tile
//      );
//  }
//
//  void limiter( Tile<D>& tile ) override {
//  PYBIND11_OVERLOAD_PURE(
//      void,
//      Current<D>,
//      limiter,
//      tile
//      );
//  }
//
//};
//


// python bindings for plasma classes & functions
void bind_ffe(py::module& m_sub)
{

  // skinny version of the Yee lattice with only (e and b meshes)
  py::class_<ffe::SkinnyYeeLattice>(m_sub, "SkinnyYeeLattice")
    .def(py::init<int, int, int>())
    .def_readwrite("ex",   &ffe::SkinnyYeeLattice::ex)
    .def_readwrite("ey",   &ffe::SkinnyYeeLattice::ey)
    .def_readwrite("ez",   &ffe::SkinnyYeeLattice::ez)
    .def_readwrite("bx",   &ffe::SkinnyYeeLattice::bx)
    .def_readwrite("by",   &ffe::SkinnyYeeLattice::by)
    .def_readwrite("bz",   &ffe::SkinnyYeeLattice::bz)
    .def("set_yee",        &ffe::SkinnyYeeLattice::set_yee)
    .def(py::self += py::self)
    .def(py::self -= py::self)
    .def(py::self *= float())
    .def(py::self /= float())
    .def(py::self +  py::self)
    .def(py::self -  py::self)
    .def(py::self *  float())
    .def(py::self /  float());


  m_sub.def("set_step", [](fields::YeeLattice& yee, ffe::SkinnyYeeLattice skyee)
      -> void 
      {
        yee.ex = skyee.ex;
        yee.ey = skyee.ey;
        yee.ez = skyee.ez;
        
        yee.bx = skyee.bx;
        yee.by = skyee.by;
        yee.bz = skyee.bz;
     }
  );


  //--------------------------------------------------
  // 1D bindings
  //py::module m_1d = m_sub.def_submodule("oneD", "1D specializations");

  //--------------------------------------------------
  // 2D bindings
  py::module m_2d = m_sub.def_submodule("twoD", "2D specializations");
  auto t2 = ffe::declare_tile<2>(m_2d, "Tile");


  //--------------------------------------------------
  // 3D bindings
  py::module m_3d = m_sub.def_submodule("threeD", "3D specializations");
  auto t3 = ffe::declare_tile<3>(m_3d, "Tile");


  //--------------------------------------------------
  // 2D Current solver bindings


  //--------------------------------------------------
  // 3D Current solver bindings
  py::class_< ffe::rFFE2<3> > currentcalc3d(m_3d, "rFFE2");
  currentcalc3d
    .def(py::init<int, int, int>())
    .def("copy_eb",      &ffe::rFFE2<3>::copy_eb)
    .def("comp_rho",     &ffe::rFFE2<3>::comp_rho)
    .def("push_eb",      &ffe::rFFE2<3>::push_eb)
    .def("add_jperp",    &ffe::rFFE2<3>::add_jperp)
    .def("update_eb",    &ffe::rFFE2<3>::update_eb)
    .def("remove_jpar",  &ffe::rFFE2<3>::remove_jpar)
    .def("limit_e",      &ffe::rFFE2<3>::limit_e);



}

} // end of ns ffe
