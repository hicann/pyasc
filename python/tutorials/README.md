<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->

# 算子开发样例

<table>
    <td> 算子样例 </td>
    <td> 功能描述 </td>
    <tr>
        <td><a href="./01_add"> 01_add</td>
        <td> 实现手动插入同步流水的Add算子。 </td>
    </tr>
    <tr>
        <td><a href="./02_add_framework"> 02_add_framework</td>
        <td> 实现通过Ascend C框架插入流水同步的Add算子。 </td>
    </tr>
    <tr>
        <td><a href="./03_matmul_mix"> 03_matmul_mix</td>
        <td> 实现MIX模式（包含矩阵计算和矢量计算）下的Matmul算子，计算公式为：C = A * B。 </td>
    </tr>
    <tr>
        <td><a href="./04_matmul_cube_only"> 04_matmul_cube_only</td>
        <td> 实现纯Cube模式（只有矩阵计算）的Matmul算子，计算公式为：C = A * B。 </td>
    </tr>
    <tr>
        <td><a href="./05_matmul_leakyrelu"> 05_matmul_leakyrelu</td>
        <td> 实现MatmulLeakyRelu算子，计算公式为：C = A * B + Bias， C = C > 0 ? C : C * 0.001。 </td>
    </tr>
</table>
