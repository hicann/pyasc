# MatmulLeakyRelu算子样例

## 概述

本样例基于Ascend C Python工程，介绍了MatmulLeakyRelu算子的实现。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B

## 样例实现

- 样例规格

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">MatmulLeakyRelu</td></tr>
  </tr>
  <tr><td rowspan="5" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">\</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">\</td><td align="center">float16</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">\</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  <tr><td align="center">alpha</td><td align="center">-</td><td align="center">float</td><td align="center">-</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">\</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </table>

- 算子实现

  本样例中实现的是固定shape为[M, N, K] = [1024, 640, 256]，bias = [640]，alpha = 0.001的MatmulLeakyRelu算子。
  - kernel实现
    - 计算逻辑是：算子使用了Matmul高阶API和LeakyRelu基础API，方便用户快速实现MatmulLeakyRelu矩阵乘法的运算操作。MatmulLeakyRelu的计算公式为：
      ```
      C = A * B + Bias
      C = C > 0 ? C : C * 0.001
      ```
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
      - Bias为矩阵乘偏置，形状为[N]。对A*B结果矩阵的每一行都采用该bias进行偏置。
    - 实现MatmulLeakyRelu运算的具体步骤如下：
      - 创建Matmul对象。
      - 初始化操作。
      - 设置左矩阵A、右矩阵B、Bias。
      - 完成矩阵乘操作。
      - 完成LeakyRelu计算。
      - 计算结果搬出。
      - 结束矩阵乘操作。

      示例如下：
      ```python
      matmul = asc.adv.Matmul(
        a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, a_dtype, IS_TRANS_A),
        b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, b_dtype, IS_TRANS_B),
        c=asc.adv.MatmulType(asc.TPosition.VECCALC, asc.CubeFormat.ND, c_dtype),
        bias=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, bias_dtype),
      )
      asc.adv.register_matmul(pipe, matmul, tiling)
      matmul.set_tensor_a(a_global, IS_TRANS_A)
      matmul.set_tensor_b(b_global, IS_TRANS_B)
      matmul.set_bias(bias_global)
      with matmul.iterate() as count:
        relu_out_local = relu_out_queue.alloc_tensor(c.dtype)
        matmul.get_tensor_c(relu_out_local, en_sequential_write=True)
        asc.leaky_relu(relu_out_local, relu_out_local, alpha, count=tiling.base_m * tiling.base_n)
        relu_out_queue.enque(relu_out_local)
        relu_out_local = relu_out_queue.deque(c.dtype)
        round_m = tiling.single_core_m // tiling.base_m
        start_offset = count % round_m * tiling.base_m * tiling.n + count // round_m * tiling.base_n
        params = asc.DataCopyParams(
            block_count=tiling.base_m,
            block_len=tiling.base_n * c.dtype.sizeof() // asc.property(asc.DEFAULT_C0_SIZE),
            src_stride=0,
            dst_stride=(tiling.n - tiling.base_n) * c.dtype.sizeof() // asc.property(asc.DEFAULT_C0_SIZE),
        )
        asc.data_copy(c_global[start_offset:], relu_out_local, repeat_params=params)
        relu_out_queue.free_tensor(relu_out_local)
      matmul.end()
      ```
  - tiling实现
    - Ascend C Python提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。只需要传入A/B/C矩阵等信息，调用API接口，即可获取到TCubeTiling结构体中的相关参数。
      - 获取Tiling参数的流程如下：
        - 创建一个Tiling对象。
        - 设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等。
        - 调用get_tiling接口，获取Tiling信息。

      示例如下：
        ```python
        matmul_tiling = host.MultiCoreMatmulTiling(host.get_ascendc_platform())
        matmul_tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, False)
        matmul_tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16, False)
        matmul_tiling.set_c_type(host.TPosition.VECCALC, host.CubeFormat.ND, host.DataType.DT_FLOAT)
        matmul_tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)

        matmul_tiling.set_dim(2)
        matmul_tiling.set_org_shape(m, n, k)
        matmul_tiling.set_shape(m, n, k)
        matmul_tiling.enable_bias(False)
        matmul_tiling.set_buffer_space(-1, -1, -1)

        tiling = asc.adv.TCubeTiling()
        matmul_tiling.get_tiling(tiling)
        ```

## 编译执行
环境配置请参考[quick_start.md](../../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。
```
cd pyasc/python/tutorials/05_matmul_leakyrelu
python3 matmul_leakyrelu.py -r [RUN_MODE] -v [SOC_VERSION]
```

其中脚本参数说明如下：
- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
  - Ascend 910C
  - Ascend 910B

示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
```
python3 matmul_cube_only.py -r Model -v Ascendxxxyy
```
用例执行完成，打屏信息出现“Sample matmul_leakyrelu run success.”，说明样例执行成功。

注：torch_npu尚不支持Python 3.12，因此该用例在Python 3.12环境上无法执行NPU模式。