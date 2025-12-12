# asc.language.basic.load_data

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: LoadData2DParams) → None

### asc.language.basic.load_data(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), params: LoadData2DParams) → None

源操作数/目的操作数的数据类型为uint8_t/int8_t时，分形矩阵大小在A1/A2上为16\*32， 在B1/B2上为32\*16。
源操作数/目的操作数的数据类型为uint16_t/int16_t/half/bfloat16_t时，分形矩阵在A1/B1/A2/B2上的大小为16\*16。
源操作数/目的操作数的数据类型为uint32_t/int32_t/float时，分形矩阵大小在A1/A2上为16\*8， 在B1/B2上为8\*16。
支持如下数据通路：
GM->A1; GM->B1; GM->A2; GM->B2;
A1->A2; B1->B2。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const LocalTensor<T>& src,
                                const LoadData2DParams& loadDataParams)
```

```c++
template <typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                const GlobalTensor<T>& src,
                                const LoadData2DParams& loadDataParams)
```

**参数说明**

- dst：目的操作数，类型为 LocalTensor。
  - 作为二维数据加载的目标 Tensor。
  - 支持的 TPosition 为 VECIN/VECCALC/VECOUT。
  - 起始地址需要 32 字节对齐。
- src：源操作数，类型为 LocalTensor 或 GlobalTensor。
  - 当为 LocalTensor 时，表示在芯片内部不同本地存储单元之间按 2D 方式搬运。
  - 当为 GlobalTensor 时，表示从 Global Memory 按 2D 方式加载数据到 LocalTensor。
  - 元素数据类型需与 dst 保持一致。
- params：二维加载参数，类型为 LoadData2DParams。
  - startIndex：分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：startIndex∈[0, 65535] 。单位：512B。默认为0。
  - repeatTimes：迭代次数，每个迭代可以处理512B数据。取值范围：repeatTimes∈[1, 255]。
  - srcStride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。
  - sid：预留参数，配置为0即可。
  - dstGap：相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。
  - ifTranspose：是否启用转置功能，对每个分形矩阵进行转置，默认为false:
  - addrMode：预留参数，配置为0即可。

**约束说明**

- dst 与 src 的数据需要满足起始地址对齐要求，具体可查看文档。
- 不使用或者不想改变的配置，建议保持默认值，有助于性能提升。

**调用示例**

- Local Memory 内部 2D 搬运（Local -> Local）
  ```python
  @asc.jit
  def kernel_load_data_l2l(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECIN,
                                addr=0, tile_size=512)
      y_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECOUT,
                                addr=0, tile_size=512)

      params = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)

      asc.load_data(y_local, x_local, params)
  ```
- Global Memory 到 Local Memory 的 2D 搬运（Global -> Local）
  ```python
  @asc.jit
  def kernel_load_data_g2l(x: asc.GlobalAddress) -> None:
      x_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECIN,
                                addr=0, tile_size=512)
      y_local = asc.LocalTensor(dtype=asc.float16,
                                pos=asc.TPosition.VECOUT,
                                addr=0, tile_size=512)

      x_gm = asc.GlobalTensor()
      x_gm.set_global_buffer(x)

      params = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)

      asc.load_data(y_local, x_local, params)
      asc.load_data(x_local, x_gm, params)
  ```
