# asc.language.basic.get_data_block_size_in_bytes

### asc.language.basic.get_data_block_size_in_bytes() → int

获取当前芯片版本一个datablock的大小，单位为byte。
开发者可以根据datablock的大小来计算API指令中待传入的repeatTime、
DataBlock Stride、Repeat Stride等参数值。

**对应的Ascend C函数原型**

```c++
__aicore__ inline constexpr int16_t GetDataBlockSizeInBytes()
```

**参数说明**

无。

**返回值说明**

当前芯片版本一个datablock的大小，单位为byte。

**调用示例**

```python
size = asc.get_data_block_size_in_bytes()
```
