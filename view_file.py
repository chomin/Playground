#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cgi
import cgitb
import os, sys
cgitb.enable()

print("Content-Type: text/html")
print("")
print("<html><body>")
print("<img src='vae_2/reconstruction_10.png'>")
# ↑ 画像のパスは適切なものに変更してください。絶対パスの方が簡単かも。。
print("</body><html>")
