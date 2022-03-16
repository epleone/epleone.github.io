---
layout:     post
title:      "Tmux Guide"
subtitle:   "Tmux 简明使用手册"
date:       2019-09-19
author:     "epleone"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Tmux
    - Ubuntu
    - Linux
---



# Tmux简明使用手册

## 简介

tmux是一款优秀的终端复用软件，它包含 `session`  	`window`	 `pane` 三个概念。

![](https://image-static.segmentfault.com/113/222/1132220757-59dd99bf3edf3_articlex)



在linux上安装命令如下：

``` shell
sudo apt-get install tmux
```



## 常用命令

### session

|            功能             |            命令             |
| :-------------------------: | :-------------------------: |
|       进入最近的`session`        |     `tmux a`      |
|       创建 `session`        |     `tmux new -s 名字`      |
|       关闭 `session`        | `tmux kill-session -t 名字` |
|         挂起 `session`         |       `prefix` + `d`       |
|       切换 `session`        |       `prefix` + `s`       |
|       重命名 `session`        |       `prefix` + `$`       |
| 关闭 `Tmux` 杀死所有session |     `tmux kill-server`      |
|     列出已有`Tmux`列表      |          `tmux ls`          |



### windows

|            功能             |            命令             |
| :-------------------------: | :-------------------------: |
|        新建`windows`        |       `prefix` + `c`       |
|        关闭`windows`        |       `prefix` + `&`       |
|        切换`windows`        |      `prefix` + `0~9`      |
|        重命名`windows`        |      `prefix` + `,`      |



### panel

|            功能             |            命令             |
| :-------------------------: | :-------------------------: |
|        左右分`panel`        |      `prefix` + `%`      |
|        上下分`panel`        |      `prefix` + `"`      |
|        关闭 `panel`        |      `prefix` + `x`      |
|        选择 `panel`        |      `prefix` + `o` /   `prefix` + `q`  |
|        调整大小 `panel`        |      `prefix` + `Ctrl+方向键`   |
|        显示时钟	       	 |      `prefix` + `t`   	|




## 翻页翻屏

`prefix` + `[`  进入翻页模式， `PgUp`  `PgDn`  实现上下翻页,  `q` 退出



## 修改指令前缀

```shell
sudo vim ~/.tmux.conf

# 从tmux v1.6版起，支持设置第二个指令前缀
# 将`键作为第二个指令前缀
set-option -g prefix2 ` 
```





