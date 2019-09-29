---
layout:     post
title:      "VS Code Setting"
subtitle:   "VS code自定义设置"
date:       2017-09-20
author:     "epleone"
header-img: "img/home-bg-o.jpg"
tags:
    - VS Code
    - Tips
---


# VS Code的自定义设置

```
// 将设置放入此文件中以覆盖默认设置
{
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "editor.fontSize": 15,
    // "python.linting.flake8Args": ["--max-line-length=300","--ignore=E712,E1101,C0121,C0301,F405,E711,C0103,E0102"]
    "python.linting.flake8Args": [
        "--max-line-length=300"
    ],
    "workbench.colorTheme": "Visual Studio Light",
    "workbench.iconTheme": "vscode-icons",
    "editor.minimap.enabled": true,
    "vsicons.dontShowNewVersionMessage": true
} 

```
