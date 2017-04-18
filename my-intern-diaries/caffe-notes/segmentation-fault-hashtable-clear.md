- **problem info**:
    ```
    I0414 10:32:02.777673  7696 layer_factory.hpp:77] Creating layer data
    *** Aborted at 1492137123 (unix time) try "date -d @1492137123" if you are using GNU date ***
    PC: @     0x7fc524976873 std::_Hashtable<>::clear()
    *** SIGSEGV (@0x9) received by PID 7696 (TID 0x7fc55fb90740) from PID 9; stack trace: ***
        @     0x7fc55cd944b0 (unknown)
        @     0x7fc524976873 std::_Hashtable<>::clear()
        @     0x7fc524968346 google::protobuf::DescriptorPool::FindFileByName()
        @     0x7fc524946ac8 google::protobuf::python::cdescriptor_pool::AddSerializedFile()
        @     0x7fc55d3fd7d0 PyEval_EvalFrameEx
        @     0x7fc55d52601c PyEval_EvalCodeEx
        @     0x7fc55d47c3dd (unknown)
        @     0x7fc55d44f1e3 PyObject_Call
        @     0x7fc55d46fae5 (unknown)
        @     0x7fc55d406123 (unknown)
        @     0x7fc55d44f1e3 PyObject_Call
        @     0x7fc55d3fa13c PyEval_EvalFrameEx
        @     0x7fc55d52601c PyEval_EvalCodeEx
        @     0x7fc55d3f4b89 PyEval_EvalCode
        @     0x7fc55d4891b4 PyImport_ExecCodeModuleEx
        @     0x7fc55d489b8f (unknown)
        @     0x7fc55d48b300 (unknown)
        @     0x7fc55d48b5c8 (unknown)
        @     0x7fc55d48c6db PyImport_ImportModuleLevel
        @     0x7fc55d403698 (unknown)
        @     0x7fc55d44f1e3 PyObject_Call
        @     0x7fc55d525447 PyEval_CallObjectWithKeywords
        @     0x7fc55d3f85c6 PyEval_EvalFrameEx
        @     0x7fc55d52601c PyEval_EvalCodeEx
        @     0x7fc55d3f4b89 PyEval_EvalCode
        @     0x7fc55d4891b4 PyImport_ExecCodeModuleEx
        @     0x7fc55d489b8f (unknown)
        @     0x7fc55d48b300 (unknown)
        @     0x7fc55d48b5c8 (unknown)
        @     0x7fc55d48c6db PyImport_ImportModuleLevel
        @     0x7fc55d403698 (unknown)
        @     0x7fc55d44f1e3 PyObject_Call
    ./examples/end2end_convnet_selfdriving/run_end2end_convnet.sh: line 12:  7696 Segmentation fault      (core dumped) ./build/tools/caffe train -solver ./examples/end2end_convnet_selfdriving/end2end_convnet_solver.prototxt
       ```
- **solution**:
     downgrading protobuf python package to fixed it:
        >$ pip install --user --upgrade protobuf==3.1.0.post1

     please refer to [this link](https://github.com/BVLC/caffe/issues/5357)
