# MIX模式的Matmul算子样例

## 概述

本样例基于Ascend C Python工程，介绍了MIX模式（包含矩阵计算和矢量计算）下的Matmul算子实现。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B

## 样例实现

- 样例规格

  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">\</td><td align="center">float</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">\</td><td align="center">float</td><td align="center">ND</td><td align="center">false</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">\</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </table>

- 算子实现

  本样例中实现的是基于Ascend910B1平台（共24个核，每个核包含1个AIC核和2个AIV核），固定shape为[M, N, K] = [512, 1024, 512]的Matmul算子。通过设置Kernel核函数JIT编译参数matmul_cube_only=False定义MIX模式的Matmul算子，该参数含义见[architecture_introduction.md](../../../docs/architecture_introduction.md)编译和运行模块。
  - kernel实现
    - 计算逻辑是：Ascend C Python提供一组Matmul高阶API，方便用户快速实现Matmul矩阵乘法的运算操作。Matmul的计算公式为：C = A * B。
      - A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。
      - C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。
    - 实现Matmul矩阵乘运算的具体步骤如下：
      - 创建Matmul对象。
      - 初始化操作。
      - 设置左矩阵A、右矩阵B。
      - 完成矩阵乘操作。
      - 结束矩阵乘操作。

      示例如下：
      ```python
      matmul = asc.adv.Matmul(
        a=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, a_dtype, IS_TRANS_A),
        b=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, b_dtype, IS_TRANS_B),
        c=asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, c_dtype),
      )
      asc.adv.register_matmul(pipe, matmul, tiling)
      matmul.set_tensor_a(a_global, IS_TRANS_A)
      matmul.set_tensor_b(b_global, IS_TRANS_B)
      matmul.set_tail(tail_m, tail_n)
      matmul.iterate_all(c_global)
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
        matmul_tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
        matmul_tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)

        matmul_tiling.set_dim(USE_CORE_NUM)
        matmul_tiling.set_org_shape(m, n, k)
        matmul_tiling.set_shape(m, n, k)
        matmul_tiling.enable_bias(False)
        matmul_tiling.set_buffer_space(-1, -1, -1)

        tiling = asc.adv.TCubeTiling()
        matmul_tiling.get_tiling(tiling)
        ```

## 编译执行
- 环境配置

  环境配置请参考[quick_start.md](../../../docs/quick_start.md#envready)。
- 打开样例目录。
  ```
  cd pyasc/python/tutorials/03_matmul_mix
  ```
- 修改配置

  修改matmul_mix.py中USE_CORE_NUM参数为当前执行平台的实际Vector核数。
- 执行如下命令可进行功能验证
  ```
  python3 matmul_mix.py
  ```
  用例执行完成，打屏信息出现“Sample matmul_mix run success.”，说明样例执行成功。

注：torch_npu尚不支持Python 3.12，因此该用例在Python 3.12环境上无法执行NPU模式。