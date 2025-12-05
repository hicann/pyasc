# Ascend C Python算子调试调优指南
本文档介绍了Ascend C Python工程支持的算子调试调优方法使用指导。

## 算子功能调试

为了方便开发者Model仿真、NPU实际运行时，进行功能调试，pyasc提供printf、dump_tensor接口供开发者调用，其中printf主要用于打印标量和字符串信息，dump_tensor用于打印指定Tensor的数据。

### 使用方法
#### 1 printf示例
```python
import asc
@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, BLOCK_LENGTH: asc.ConstExpr[int],
                BUFFER_NUM: asc.ConstExpr[int], TILE_LENGTH: asc.ConstExpr[int], TILE_NUM: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * BLOCK_LENGTH
    ...
    # 使用printf打印字符串
    asc.printf("Before calculating.\\n")
    for i in range(TILE_NUM):
        # 使用printf打印当前循环次数
        asc.printf("current index is %d.\\n", i)
        copy_in(i, x_gm, y_gm, in_queue_x, in_queue_y, TILE_LENGTH)
        compute(z_gm, in_queue_x, in_queue_y, out_queue_z, TILE_LENGTH)
        copy_out(i, z_gm, out_queue_z, TILE_LENGTH)
```
执行结果样例如下：
```
opType=v, DumpHead: AIV-0, CoreType=AIV, block dim=16, total_block_num=16, block_remain_len=1048024, block_initial_space=1048576, rsv=0, magic=5aa5bccd
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
Before calculating.
current index is 0.
current index is 1.
current index is 2.
current index is 3.
current index is 4.
current index is 5.
current index is 6.
current index is 7.
```

- 约束

  - 使用asc.printf若需换行，需对'\n'进行转义；

  - asc.printf接口会对算子实际运行的性能带来一定影响，通常在调测阶段使用。

#### 2 dump_tensor示例
```python
import asc
@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, BLOCK_LENGTH: asc.ConstExpr[int],
                BUFFER_NUM: asc.ConstExpr[int], TILE_LENGTH: asc.ConstExpr[int], TILE_NUM: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * BLOCK_LENGTH
    x_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    ...
    # 使用dump_tensor打印x_gm输入
    tmp_array = asc.array(asc.uint32, [4, 16])
    tmp_shape_info = asc.ShapeInfo(tmp_array)
    asc.dump_tensor(x_gm, 0, 32, tmp_shape_info)
    for i in range(TILE_NUM):
        copy_in(i, x_gm, y_gm, in_queue_x, in_queue_y, TILE_LENGTH)
        compute(z_gm, in_queue_x, in_queue_y, out_queue_z, TILE_LENGTH)
        copy_out(i, z_gm, out_queue_z, TILE_LENGTH)

@asc.jit
def copy_in(i: int, x_gm: asc.GlobalAddress, y_gm: asc.GlobalAddress, in_queue_x: asc.TQue, in_queue_y: asc.TQue,
            TILE_LENGTH: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    asc.data_copy(x_local, x_gm[i * TILE_LENGTH:], count=TILE_LENGTH)
    # 使用dump_tensor打印Local Memory的Tensor
    if i == 0:
        asc.dump_tensor(x_local, 1, 32)
    ...
```
执行结果样例如下：
```
opType=v, DumpHead: AIV-0, CoreType=AIV, block dim=16, total_block_num=16, block_remain_len=1048024, block_initial_space=1048576, rsv=0, magic=5aa5bccd
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
DumpTensor: desc=0, addr=41200000, data_type=float32, position=GM, dump_size=32
[[19.000000, 4.000000, 38.000000, 50.000000, 39.000000, 67.000000, 84.000000, 98.000000, 21.000000, 36.000000, 18.000000, 46.000000, 10.000000, 92.000000, 26.000000, 38.000000],
[39.000000, 9.000000, 82.000000, 37.000000, 35.000000, 65.000000, 97.000000, 59.000000, 89.000000, 63.000000, 70.000000, 57.000000, 35.000000, 3.000000, 16.000000, 42.000000],
[-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-],
[-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-]]
DumpTensor: desc=1, addr=0, data_type=float32, position=UB, dump_size=32
[6.000000, 34.000000, 52.000000, 38.000000, 73.000000, 38.000000, 35.000000, 14.000000, 67.000000, 62.000000, 30.000000, 49.000000, 86.000000, 37.000000, 84.000000, 18.000000, 38.000000, 18.000000, 44.000000, 21.000000, 86.000000, 99.000000, 13.000000, 79.000000, 84.000000, 9.000000, 48.000000, 74.000000, 52.000000, 99.000000, 80.000000, 53.000000]
```

- 约束

  - asc.dump_tensor接口会对算子实际运行的性能带来一定影响，通常在调测阶段使用。


## 使用msprof采集性能数据

进行性能调优时，开发者可以使用msprof工具来采集和分析运行在昇腾AI处理器上的AI任务在各个运行阶段的关键性能指标，根据输出的性能数据，快速定位软、硬件性能瓶颈，提升AI任务性能分析的效率。关于msprof工具的详细介绍请参考官方文档：[性能调优工具](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/Profiling/atlasprofiling_16_0001.html)。

### 环境准备

工具使用前，需完成环境准备，详细参考[quick_start.md](quick_start.md#envready)。

### 采集性能数据

本文以add算子为例，介绍如何使用msprof工具采集性能数据。

add算子实现代码参考[02_add_framework.py](../python/tutorials/02_add_framework/add_framework.py),
确保代码中后端模式配置为NPU，如下：

```Python
if __name__ == "__main__":
    logging.info("[INFO] start process sample add_framework.")
    vadd_custom(config.Backend.NPU)
    logging.info("[INFO] Sample add_framework run success.")

```

算子上板信息采集，参考命令如下，详细命令参数请参考msprof工具文档的说明。
```Shell
msprof --output=./output python add_framework.py
```

成功执行后在output目录下生成如下文件：
```
├── host         //保存原始数据，用户无需关注
...
│    └── data
├── device_{id}  //保存原始数据，用户无需关注
...
│    └── data
...
├── mindstudio_profiler_log
├── mindstudio_profiler_output
      ├── api_static_{timestamp}.csv   //用于统计CANN层的API执行耗时信息
      ├── op_static_{timestamp}.csv    //AI Core和AI CPU算子调用次数及耗时统计
      ├── op_summary_{timestamp}.csv   //AI Core和AI CPU算子数据
      ├── task_time_{timestamp}.csv    //Task Scheduler任务调度信息
      ├── msprof_{timestamp}.json      //timeline数据总表
      ├── README.txt
```

## 使用Ascend PyTorch Profiler采集性能数据

pyasc支持针对PyTorch框架开发的性能分析工具Ascend PyTorch Profiler。PyTorch训练/在线推理场景下，推荐通过Ascend PyTorch Profiler接口采集并解析性能数据，用户可以根据结果自行分析和识别性能瓶颈，请参考官方文档：[Ascend PyTorch Profiler性能数据采集和自动解析](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0033.html)。

### 环境准备

pyasc的环境准备请参考[quick_start.md](quick_start.md#envready)，Ascend PyTorch Profiler的环境准备请参考[Ascend PyTorch Profiler性能数据采集和自动解析](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0033.html)的前提条件小节。

本文以add算子为例，介绍如何通过Ascend PyTorch Profiler接口采集性能数据。

### 示例代码（通过with语句进行采集）

```Python
import torch
import torch_npu

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt


@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, block_length: asc.ConstExpr[int],
                buffer_num: asc.ConstExpr[int], tile_length: asc.ConstExpr[int], tile_num: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * block_length
    ......


def vadd_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    z = torch.zeros_like(x)
    total_length = z.numel()
    use_core_num = 16
    block_length = (total_length + use_core_num - 1) // use_core_num
    tile_num = 8
    tile_length = (block_length + tile_num - 1) // tile_num
    buffer_num = 1
    vadd_kernel[use_core_num, rt.current_stream()](x, y, z, block_length, buffer_num, tile_length, tile_num)
    return z


def test_vadd(backend: config.Backend):
    config.set_platform(backend)
    size = 8192
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    x = torch.randn(size, dtype=torch.float32, device=device)
    y = torch.randn(size, dtype=torch.float32, device=device)
    z = vadd_launch(x, y)
    assert torch.allclose(z, x + y)


experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[
        torch_npu.profiler.ExportType.Text
        ],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=0),    
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config) as prof:
    steps = 2
    for step in range(steps):
        test_vadd(config.Backend.NPU)
        prof.step()    

```

### NPU性能数据目录结构

执行后生成数据目录结构示例如下：

```
├── ubuntu_239247_20251101101435_ascend_pt                
    ├── ASCEND_PROFILER_OUTPUT          
    ├── FRAMEWORK
    ├── logs                     
    ├── PROF_000001_20230628101435646_FKFLNPEPPRRCFCBA  
    ├── profiler_info.json
    ├── profiler_metadata.json
```
