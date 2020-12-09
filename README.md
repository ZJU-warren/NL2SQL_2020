### NL2SQL_2020 文件夹



#### 1. 功能说明

本项目为from\where\having\groupyby\orderby\limit\combnation各模块生成解决方案，包含select部分也是可以的，但最后没有使用，此外附带数据集的划分生成程序，以用来生成各个模块的数据集。



### 2. 运行

- 首先需要指定各个文件的地址

  在DataLinkSet.py里指定

  

- 生成训练集和测试集

  在主目录下有pre.py文件，在其中指定包含哪些数据集，并且指定测试集大小进行生成各个模块的数据集.

  ```python
  # 示例:
  python pre.py --test_data_ratio 0.3
  ```

  

- 运行train.py

  说明：这里需要到core.proxies.main_proxy把__init__和run函数下被注释掉的组件取消注释，就可以运行一次生成所有的结果。但是实际运行的时候，因为GPU的容量有限，所以还是得分模块跑。所以每次运行需要注释掉部分。

  

- 运行test.py

  说明：每次运行train.py后运行test.py，生成对应的模块结果

  

- 生成各个模块的结果在..\DataSet\Result中，模型在..\DataSet\Models中

  

### 3. 结果

- 生成的结果和select部分(见另外一份Readme.md)，一起经过condition填值模块和genquery来生成结果。