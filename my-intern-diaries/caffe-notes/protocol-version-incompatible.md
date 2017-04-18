- **problem info:**

    ```
    [libprotobuf FATAL google/protobuf/stubs/common.cc:61] This program requires version 3.2.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "google/protobuf/descriptor.pb.cc".)
    terminate called after throwing an instance of 'google::protobuf::FatalException'
      what():  This program requires version 3.2.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "google/protobuf/descriptor.pb.cc".)
    *** Aborted at 1491544094 (unix time) try "date -d @1491544094" if you are using GNU date ***
    PC: @     0x7f839d75e428 gsignal
    *** SIGABRT (@0x3e800001dfa) received by PID 7674 (TID 0x7f83a0409740) from PID 7674; stack trace: ***
        @     0x7f839d75e4b0 (unknown)
        @     0x7f839d75e428 gsignal
        @     0x7f839d76002a abort
        @     0x7f839e57184d __gnu_cxx::__verbose_terminate_handler()
        @     0x7f839e56f6b6 (unknown)
        @     0x7f839e56f701 std::terminate()
        @     0x7f839e56f919 __cxa_throw
        @     0x7f839eab5647 google::protobuf::internal::LogMessage::Finish()
        @     0x7f839eab587d google::protobuf::internal::VerifyVersion()
        @     0x7f8373b0a0d4 google::protobuf::protobuf_google_2fprotobuf_2fdescriptor_2eproto::TableStruct::InitDefaultsImpl()
        @     0x7f839eab5f75 google::protobuf::GoogleOnceInitImpl()
        @     0x7f8373b04d85 google::protobuf::protobuf_google_2fprotobuf_2fdescriptor_2eproto::InitDefaults()
        @     0x7f8373b04db9 google::protobuf::protobuf_google_2fprotobuf_2fdescriptor_2eproto::AddDescriptorsImpl()
        @     0x7f839eab5f75 google::protobuf::GoogleOnceInitImpl()
        @     0x7f8373b04e35 google::protobuf::protobuf_google_2fprotobuf_2fdescriptor_2eproto::AddDescriptors()
        @     0x7f83a023a4ea (unknown)
        @     0x7f83a023a5fb (unknown)
        @     0x7f83a023f712 (unknown)
        @     0x7f83a023a394 (unknown)
        @     0x7f83a023ebd9 (unknown)
        @     0x7f838cfaef09 (unknown)
        @     0x7f83a023a394 (unknown)
        @     0x7f838cfaf571 (unknown)
        @     0x7f838cfaefa1 dlopen
        @     0x7f839dde588d _PyImport_GetDynLoadFunc
        @     0x7f839de544be _PyImport_LoadDynamicModule
        @     0x7f839de55300 (unknown)
        @     0x7f839de555c8 (unknown)
        @     0x7f839de566db PyImport_ImportModuleLevel
        @     0x7f839ddcd698 (unknown)
        @     0x7f839de191e3 PyObject_Call
        @     0x7f839deef447 PyEval_CallObjectWithKeywords
    Aborted (core dumped)
    ```
- solution:

    rebuild caffe (make clean; make all; make test; make runtest)
    don't know why.
