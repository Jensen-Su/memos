## problem description
```matlab
libprotobuf ERROR google/protobuf/descriptor_database.cc:57] File already exists in database: foo/foo.proto libprotobuf FATAL google/protobuf/descriptor.cc:862] CHECK failed: generated_database_->Add(encoded_file_descriptor, size):  terminate called after throwing an instance of 'google::protobuf::FatalException'   what():  CHECK failed: generated_database_->Add(encoded_file_descriptor, size):
```

## 动静态库问题

在`Linux`上编译`google protobuff`时，`configure` 默认选项是生成动态库，即`libprotobuf.so`文件。如果同时在多个动态库(动态库以`dlopen`方式动态加载)中使用同一`buff`结构，则运行时会报错误：
```
ibprotobuf ERROR google/protobuf/descriptor_database.cc:57] File already exists in database: foo/foo.proto libprotobuf FATAL google/protobuf/descriptor.cc:862] CHECK failed: generated_database_->Add(encoded_file_descriptor, size):  terminate called after throwing an instance of 'google::protobuf::FatalException'   what():  CHECK failed: generated_database_->Add(encoded_file_descriptor, size): 
```

为了解决这个问题，`google protobuff`，则不能以动态库的形式调用，改用静态库的形式在编译时加载。

编译`google protobuff`时，在`configure` 时加上选项：
```shell
configrue --disable-shared
```
即可编译成静态库：`libprotobuf.a` 但是默认的`configure`文件中，在编译时未加`-fPIC` ，导致在引用静态库的工程中编译链接时报错误：
```shell
libs/assert.o: relocation R_X86_64_32 against `a local symbol' can not be used when making a shared object; recompile with -fPIC .libs/assert.o: could not read symbols: Bad value
```

解决该问题，需要重新编译`google protobuff`库，并添加编译选项:`-fPIC`

以文本形式打开`google buff`代码目录下的`configure`文件，在把第2575至2578行修改为如下：
```configure
if test "x${ac_cv_env_CFLAGS_set}" = "x"; then :   CFLAGS="-fPIC" fi if test "x${ac_cv_env_CXXFLAGS_set}" = "x"; then :   CXXFLAGS="-fPIC"
```
需要注意的是不同版本的`configure`文件不同，所以源代码的行数也不同，2.3.0是1962行开始,贴出被替换代码，以便于替换

```configure
if test "x${ac_cv_env_CFLAGS_set}" = "x"; then
 CFLAGS=""
fi

if test "x${ac_cv_env_CXXFLAGS_set}" = "x"; then
  CXXFLAGS=""
fi
```

替换时注意if 和fi 的配对使用，否则执行不了，会出现语法错误，文件无法正常结束。

在修改文件后编译要重新编译，首先进行make clean ，否则不会重新执行:
```shell
./configure --disable-shared
make clean
make
make check
make install

最后修改环境变量，建议修改本用户的环境变量，`~/.bashrc`，不修改`etc`下环境变量
```shell
# append protobuf to PATH/lys
export PROTOBUF_HOME=/usr/local/protobuf/protobuf-2.3.0
export PATH=$PROTOBUF_HOME/bin:$PATH
```
`HOME`目录由安装目录而定，各不相同。

测试是否安装成功，`protoc --version`,显示出版本则说明安装成功
