# asc.language.basic.set_load_data_boundary

### asc.language.basic.set_load_data_boundary(boundary: int) → None

设置 Load3D 时 A1/B1 边界值。
如果 Load3D 指令在处理源操作数时，源操作数在 A1/B1 上的地址超出设置的边界，则会从 A1/B1 起始地址开始读取数据。

**对应的 Ascend C 函数原型**

```c++
__aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
```

**参数说明**

- boundaryValue
  : 边界值。
    Load3Dv1 指令：单位是 32 字节。
    Load3Dv2 指令：单位是字节。

**约束说明**

- 用于 Load3Dv1 时， boundaryValue 的最小值是 16 （单位： 32 字节）；用于 Load3Dv2 时， boundaryValue 的最小值是 1024 （单位：字节）。
- 如果使用 SetLoadDataBoundary 接口设置了边界值，配合 Load3D 指令使用时， Load3D 指令的 A1/B1 初始地址要在设置的边界内。
- 如果 boundaryValue 设置为 0 ，则表示无边界，可使用整个 A1/B1 。

**调用示例**

```python
import asc
asc.set_load_data_boundary(1024)
```
