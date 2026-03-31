# Semantic Video Compression
执行顺序：

1. 按照`write_stream.txt`文件中的步骤，在Linux系统上编译C++模块，该模块包含了数字编码相关的API
2. 修改`dataset_config.json`中的数据集路径
3. 修改`test.sh`中的检查点路径
4. 运行`test.sh`，相关输出结果将呈现在项目根目录的`output_dir`文件夹下。如果想要修改输出文件夹，可更改`test_video.py`中的`--output_dir`选项
