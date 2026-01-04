Installation
=============

Generally speaking, installing `forest_gen` is simpler than installing
`stripe_kit`. We are going to assume you have `stripe_kit` installed in your
current environment, and - by association, have IsaacLab installed and working.

Installing the package itself is simple enough. Installing it from PyPI is as
simple as running:

.. code-block:: bash

    pip install forest_gen

We are as shocked as you that a name this generic wasn't taken. Alternatively,
of course installing it from git is also possible:

.. code-block:: bash

    pip install git+https://github.com/GrafCzterech/forest_gen.git

Here's where the tricky part begins though. Due to PyPI package size
constraints, the models we build the module around aren't included in the
package, much to our dismay. You will need to download them yourself, and place
them in a directory of your choice. :py:class:`forest_gen.ForestGenSpec`
accepts a path to the directory containing the models.

You can always just clone the repository and obtain the models that way:

.. code-block:: bash

    git clone https://github.com/GrafCzterech/forest_gen.git
    cd forest_gen
    cp -r models /path/to/models

But that sort of defeats the purpose of using PyPI to download the package.
So we have this here bash script that will download the models without
cloning the repository:

.. code-block:: bash
    mkdir -p models
    cd models

    for url in $(curl -s https://api.github.com/repos/GrafCzterech/forest_gen/contents/models | jq -r '.[] | select(.type=="file") | .download_url'); do
        wget $url
    done
