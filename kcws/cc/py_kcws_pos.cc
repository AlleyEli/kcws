/*
 * Copyright 2016- 2018 Koth. All Rights Reserved.
 * =====================================================================================
 * Filename:  py_word2vec_vob.cc
 * Author:  Koth Chen
 * Create Time: 2016-07-25 18:46:27
 * Description:
 *
 */
#include "third_party/pybind11/pybind11.h"
#include "third_party/pybind11/stl.h"
#include "kcws_pos_use.h"
namespace py = pybind11;

PYBIND11_PLUGIN(py_kcws_pos) {
  py::module m("py_kcws_pos", "python binding for py_sba");
  py::class_<kcwsPosProcess>(m, "kcwsPosProcess", "python class seg_backend_api_hy")
  .def(py::init())
  .def("kcwsSetEnvfilePars", &kcwsPosProcess::kcws_set_envfile_pars, "load env file and set parmerters")
  .def("kcwsPosProcessSentence", &kcwsPosProcess::kcws_pos_process, "process sentence");
  return m.ptr();
}