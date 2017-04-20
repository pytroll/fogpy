===========================
 Installation instructions
===========================

Getting the files and installing them
=====================================

First you need to get the files from github::

  cd /path/to/my/source/directory/
  git clone https://github.com/m4sth0/fogpy

You can also retreive a tarball from there if you prefer, then run::
  
  tar zxvf tarball.tar.gz

Then you need to install fogpy on you computer::

  cd fogpy
  python setup.py install [--prefix=/my/custom/installation/directory]

You can also install it in develop mode to make it easier to hack::

  python setup.py develop [--prefix=/my/custom/installation/directory]

