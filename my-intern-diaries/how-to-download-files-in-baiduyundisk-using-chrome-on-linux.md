## An instruction for how to download big files stored in BaiduyunDisk using chrome plugin on Ubuntu 14.04

1. Switch to somewhere you like, then clone the following repository:

    $ git clone https://github.com/acgotaku/BaiduExporter.git

There will be a `.crx` file under directory `BaiduExporter`, drag it to chrome extension page to install.

2. Install `aria2`:

    $ sudo add-apt-repository ppa:t-tujikawa/ppa
    $ sudo apt-get update
    $ sudo apt-get install aria2

Ok, Now we have sucessfully installed chrome extension `BaiduExporter.crx` and `aria2`. To start `aria2c` in terminal:

    $ aria2c --enable-rpc --rpc-listen-all --rpc-allow-origin-all  --file-allocation=none --max-connection-per-server=3 --max-concurrent-downloads=5 --continue -d ~/

Some explaination for the options:

    --enable-rpc[=true|false]
        Enable JSON-RPC/XML-RPC sever. It is strongly recommended to set username and password using `--rpc-user` and `--rpc-passwd` option. Default: false

    --rpc-allow-origin-all[=true|fale]
        Add Access-Control-Allow-Origin header field with value \* to the RPC response. Default: false

    -x, --max-connection-per-server=<NUM>
        The maximun number of connections to one server for each download. Default: 1

    -j, --max-concurrent-downloads=<N>
        Set maximun number of parallel downloads for every static (HTTP/FTP) URI, torrent and metalink. Default: 5

    -c, --continue[=true|false]
        Continue downloading a partially downloaded file. Use this option to resume a download started by a web browser or another program which downloads files sequentially from the beginning. Currently this option is only applicable to HTTP(S)/FTP downloads.

    -d, --dir=<DIR>
        The directory to store the downloaded file.


However, it is a bit boring to type a such long command. We can make it an executable script:

    $ sudo vim /usr/local/bin/Aria2c

        vim: aria2c --enable-rpc --rpc-listen-all --rpc-allow-origin-all  --file-allocation=none --max-connection-per-server=3 --max-concurrent-downloads=5 --continue -d ~/
        vim: shift+:x

    $ sudo chmod +x /usr/local/bin/Aria2c


Ok, Let's download a file from BaiduyunDisk for a test. 

...

It is a bit annoying to see the downloading progress in a terminal, however.

Install the chrome extension YAAW then you can see the progress bar in a web page.





We can, still, combine the `uget` and `aria2` tools for a more powerful downloading on Ubuntu.

1. install `uget`:

    $ sudo add-apt-repository ppa:plushuang-tw/uget-stable
    $ sudo apt-get update
    $ sudo apt-get install uget

One also can install it by downloading a `.deb` package from http://ugetdm.com/

2. open uget->edit->settings->plugin:

...Just see the file...
