源代码来自于 https://github.com/bbruceyuan/LLMs-Zero-to-Hero
本人对此代码进行了添加补充
主要有以下更新：

1. 更新generate函数，修改相应逻辑，当出现eos_token后输出停止；
2. 添加kv-cache推理优化，在Config.py中修改参数即可开启；
3. 添加inference.py文件，可以直接输入Question，运行即可得到输出；
4. model修改，将原始MultiHeadAttention修改为GroupQueryAttention，相应参数在Config.py中；
5. 附带模型权重，可以实现中文输出。我用夸克网盘给你分享了「gptwithhand」，点击链接或复制整段内容，打开「夸克APP」即可获取。
   /~d1c5386yCN~:/
   链接：https://pan.quark.cn/s/1eb6ae7200b4?pwd=GBUh
   提取码：GBUh
6. 不用下载原始数据集，在data文件夹中保存原始数据集前1000条，可以用来自己训练测试。
   使用方式：
7. 训练 python train.py
8. 推理 python inference.py
9. 修改参数在Config.py中 
