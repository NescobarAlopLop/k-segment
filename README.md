# k-segment-2d
The goal of this project is to extend the ability of k-segment coreset algorithm to be able to provide compression
for 2 dimensional data.

## How to:
To start working with this project please have a look at tests first.
I feel like my predecessors did quite a good job and they are pretty much self explanatory.

The original work was done here [https://github.com/vkhakham/k-segment](https://github.com/vkhakham/k-segment)  


## Spark Jupiter Install:

Follow this 
[medium article](https://medium.com/@josemarcialportilla/getting-spark-python-and-jupyter-notebook-running-on-amazon-ec2-dec599e1c297),
 for more recent conda install instructions, this [digitalocen tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart) is also useful.
 
 TIP: 8GB root volume machine wont suffice.
 ```conda install -c anaconda msgpack-python```
 `conda create -n py35 python=3.5 anaconda
activate py35`
