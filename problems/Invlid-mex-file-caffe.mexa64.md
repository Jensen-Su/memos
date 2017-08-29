## problem description
```matlab
    Invalid MEX-file ‘**/caffe.mexa64’ 
    /usr/local/MATLAB/R2015b/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version GLIBCXX_3.4.21 not found (required by /usr/lib/x86_64-Linux-gnu/libgflags.so.2)
```

## diagnose

```shell
$ strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3
$ locate libstdc++.so.6 | grep /usr/lib/
/usr/lib/x86_64-linux-gnu/libstdc++.so.6
/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21
/usr/share/gdb/auto-load/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21-gdb.py
```

## solution

```shell
$ ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/local/MATLAB/R2015b/bin/glnxa64/libstdc++.so.6
```

Restart you matlab.
