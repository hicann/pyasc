# asc.language.basic.set_deq_scale

### asc.language.basic.set_deq_scale(scale: float) → None

### asc.language.basic.set_deq_scale(scale: float, offset: int, sign_mode: bool) → None

设置DEQSCALE寄存器的值。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetDeqScale(half scale)

__aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode)
```

**参数说明**

- scale(half)：scale量化参数，half类型。
- scale(float)：scale量化参数，float类型。
- offset：offset量化参数，int16_t类型，只有前9位有效。
- sign_mode：bool类型，表示量化结果是否带符号。

**调用示例**

```python
# Cast
scale = 1.0
asc.set_deq_scale(scale)
asc.cast(cast_dst_local, cast_dsrc_local, asc.RoundMode.CAST_NONE, src_size)
# CastDeq
scale = 1.0
offset = 0
sign_mode = True
asc.set_deq_scale(scale, offset, sign_mode)
asc.cast_deq(dst_local, src_local, count=src_size, is_vec_deq=False, half_block=False)
```
