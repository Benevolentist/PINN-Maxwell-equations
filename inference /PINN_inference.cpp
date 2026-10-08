#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <string>

class PINNInferenceEngine {

private:
    PyObject* pModule = nullptr;
    PyObject* pFunc = nullptr;

public:
    PINNInferenceEngine(const std::string& module_name, const std::string& func_name) {
        Py_Initialize();
        PyObject* sysPath = PySys_GetObject("path");
        PyObject* pDir = PyUnicode_FromString(".");
        if (sysPath && pDir) {
            PyList_Append(sysPath, pDir);
        }
        Py_XDECREF(pDir);

        PyObject* pName = PyUnicode_DecodeFSDefault(module_name.c_str());
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);

        if (pModule) {
            pFunc = PyObject_GetAttrString(pModule, func_name.c_str());
            if (!pFunc || !PyCallable_Check(pFunc)) {
                PyErr_Print();
                std::cerr << "Error: Function '" << func_name << "' not found" << std::endl;
            }
        } else {
            PyErr_Print();
            std::cerr << "Error: Failed to load module '" << module_name << "'." << std::endl;
        }
    }

    ~PINNInferenceEngine() {
        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
        Py_Finalize();
    }

    // memory safe inference
    bool predict(double x, double t, double& E_out, double& B_out) {
        if (!pFunc) return false;
        PyObject* pArgs = PyTuple_New(2);
        PyObject* pX = PyFloat_FromDouble(x);
        PyObject* pT = PyFloat_FromDouble(t);

        if (!pArgs || !pX || !pT) {
            Py_XDECREF(pArgs);
            Py_XDECREF(pX);
            Py_XDECREF(pT);
            return false;
        }
        PyTuple_SetItem(pArgs, 0, pX);
        PyTuple_SetItem(pArgs, 1, pT);

        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs); // deallocates tuple with pX and pT

        if (pValue != nullptr) {
            if (PyTuple_Check(pValue) && PyTuple_Size(pValue) == 2) {
                E_out = PyFloat_AsDouble(PyTuple_GetItem(pValue, 0));
                B_out = PyFloat_AsDouble(PyTuple_GetItem(pValue, 1));
                Py_DECREF(pValue);
                return true;
            }
            Py_DECREF(pValue);
        }
        PyErr_Print();
        return false;
    }
};

int main() {
    // single initialization for Python engine
    PINNInferenceEngine engine("PINN_Maxwell", "predict_cpp");

    double x_in = 2.5;
    double t_in = 1.0;
    double E_pred = 0.0, B_pred = 0.0;

    if (engine.predict(x_in, t_in, E_pred, B_pred)) {
        std::cout << "--- Optimized C++ Inference ---" << std::endl;
        std::cout << "Inputs: x = " << x_in << ", t = " << t_in << std::endl;
        std::cout << "Predicted E field: " << E_pred << std::endl;
        std::cout << "Predicted B field: " << B_pred << std::endl;
    } else {
        std::cerr << "Inference failed." << std::endl;
    }

    return 0;
}
