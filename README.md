# DQN

请在配置了`pytorch`和`matplotlib`的环境中运行

配置环境

```bash
pip install gym==0.10.5
pip install tensorboardX
```

运行

```bash
python main.py
```

运行时会在控制台打印输出

运行后会生成图片`plot.png`，绘制三条曲线

运行时，执行下面的指令，使用`tensorboard`实时查看训练效果

```bash
tensorboard --logdir=./log
```

