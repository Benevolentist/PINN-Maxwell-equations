#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>

int main() {
    Py_Initialize();
    PyRun_SimpleString("import sys; sys.path.append('.')");

    PyObject* pName = PyUnicode_DecodeFSDefault("PINN_Maxwell");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "predict_cpp");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Set input coordinates (x = 2.5, t = 1.0)
            double x_in = 2.5;
            double t_in = 1.0;

            PyObject* pArgs = PyTuple_Pack(2, PyFloat_FromDouble(x_in), PyFloat_FromDouble(t_in));

            PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != nullptr) {
                // Extract output tuple (E_pred, B_pred)
                double E_pred = PyFloat_AsDouble(PyTuple_GetItem(pValue, 0));
                double B_pred = PyFloat_AsDouble(PyTuple_GetItem(pValue, 1));

                std::cout << "--- Inference from C++ ---" << std::endl;
                std::cout << "Inputs: x = " << x_in << ", t = " << t_in << std::endl;
                std::cout << "Predicted E field: " << E_pred << std::endl;
                std::cout << "Predicted B field: " << B_pred << std::endl;

                Py_DECREF(pValue);
            } else {
                PyErr_Print();
                std::cerr << "Error: Function execution failed." << std::endl;
            }
            Py_XDECREF(pFunc);
        } else {
            PyErr_Print();
            std::cerr << "Error: Cannot find function 'predict_cpp'" << std::endl;
        }
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        std::cerr << "Error: Failed to load module 'PINN_Maxwell'" << std::endl;
    }

    Py_Finalize();
    return 0;
}
