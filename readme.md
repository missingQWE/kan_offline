# Offline Model-Based Optimization by Learning to Rank

#### 环境安装

```
# Create conda environment
conda create -n mojoco python=3.8 -y
conda activate mojoco

#To install mujoco-py on Ubuntu, make sure you have the following libraries
installed:
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

# Download MuJoCo package
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-
linux-x86_64.tar.gz -O mujoco210_linux.tar.gz
mkdir ~/.mujoco
tar -zxvf mujoco210_linux.tar.gz -C ~/.mujoco

pip install Cython==0.29.36 numpy==1.22.0 mujoco_py==2.1.2.14
# Set up the environment variable

conda env config vars set
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
# Reactivate the conda environment to make the variable take effect
conda activate mojoco
# Mujoco Compile
python -c "import mujoco_py"
```

![fd4852c1-73da-4fc8-a688-d46830a03732](C:\Users\chukuo\xwechat_files\wxid_zvd50143pjuj22_ecb3\temp\InputTemp\fd4852c1-73da-4fc8-a688-d46830a03732.png)

```
# Torch Installation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --
extra-index-url https://download.pytorch.org/whl/cu117
# Design-Bench Installation
pip install design-bench==2.0.12
pip install pip==24.0
pip install robel==0.1.2 morphing_agents==1.5.1 transforms3d --no-dependencies
pip install botorch==0.6.4 gpytorch==1.6.0
# Install other dependencies
pip install gym==0.13.1 params_proto==2.9.6 scikit-image==0.17.2 scikitvideo==1.1.11 scikit-learn==0.23.1 wandb
# Fix numpy version, otherwise it would raise environment error
pip install numpy==1.22.0
```

环境安装成功之后，需要进入这个路径，修改一下design-bench中approximate_oracle.py的代码

```
with zip_archive.open('rank_correlation.npy', "r") as file:
	rank_correlation = np.loads(file.read())
#把上面的内容修改为下面的内容
with zip_archive.open('rank_correlation.npy', "r") as file:
	rank_correlation = np.load(file, allow_pickle=True).item()
```



### 服务器配置

```
1. miniconda安装
参照https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2这个网站安装即可。linux版本。

2.在windws下，下载MobaXterm，下载链接如下，选择free就可以
https://mobaxterm.mobatek.net/download.html

3.miniconda配置好之后进入服务器配置环境，与常规的windows下配置一样。
tips:就是常规的什么pip install numpy这种。

4.wandb需要连接外网，这样才能把数据传上去，因此需要设置一下，是使得服务器可以访问外网。
可参考这篇博客进行 https://blog.csdn.net/XT139927/article/details/143509783
```

