# Ascend C Python编程接口开发指南
在浏览该文档前，建议您先根据开发需求浏览[代码架构说明](architecture_introduction.md)文档。
以Add算子为例，pyasc要新增对应的python的接口，需要修改以下模块：
Python前端模块、AST转ASC-IR模块、ASC-IR定义模块、Ascend C代码生成模块。
所有新开发代码请遵守下述的"编码规范"章节列举的规范内容。

## 各模块开发内容说明
### Python前端模块

Python 前端模块的主要功能是定义与[Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 一一对应的Python编程接口。
本部分分为目录接口、接口实现逻辑、Add API实现示例和解释三部分，说明前端接口的开发方式。

#### 目录结构

根据Ascend C API的目录，python侧的目录组织如下：

```
python/asc/language/adv        # 对应高阶API
python/asc/language/basic      # 对应基础API（除内存管理与同步控制部分）
python/asc/language/core       # 对应数据类型定义，另外包括枚举类型等。
python/asc/language/fwk        # 对应内存管理与同步控制，包括TPipe、TQue等框架相关的类型和接口
```

每个目录中的接口均按照类别或功能分在相应的文件下。例如，在`basic`目录下，接口被细分为以下几个功能模块:

- `common.py`：通用接口
- `data_copy.py`：数据搬运接口
- `dump_tensor.py`：算子调测接口
- `sys_var.py`：系统变量接口
- `vec_binary_scalar.py`：标量运算接口
- `vec_binary.py`：矢量双目运算接口
- `vec_duplicate.py`：变量或立即数复制接口
- `vec_unary.py`：矢量单运算接口
- `vec_vconv.py`：精度转换接口以及带精度转换的计算接口


根据接口的具体功能，选择合适的文件进行添加。若现有文件无法满足需求，可考虑新增文件以保持文件的解耦和模块的独立性。

新增代码时，建议与Ascend C中的接口文件分类方法和顺序保持一致。

#### 接口实现逻辑


接口命名和实现需要按照python的命名规则命名函数，具体可参考[Python前端模块编码规范](#python前端模块编码规范)。

整体实现逻辑如下：

- 获取IR构建器
- 参数转换：将接口中定义的参数转换为中间表示（IR）
- 创建IR：调用IR构建器创建IR
- 返回：如有返回值，将生成的IR包装成需要返回的对象
- 在文件目录下的`__init__.py`和 `python\asc\language\__init__.py` 中增加引用

#### Add API实现示例和解释

本部分将以实现Add API为例，详细描述新增函数、类、结构体、枚举类型四种编程场景的完整流程。若所实现的接口仅涉及某一特定编程场景，可单独参考相应章节的详细说明。

  - 定义函数

    1. 根据函数类型选择合适的文件添加，如add属于双目运算，对应文件为`python\asc\language\basic\vec_binary.py` 。

    2. 实现函数

        完整实现请参考<a href="../python/asc/language/basic/vec_binary.py">vec_binary.py</a>。

        - 编写函数重载

            `@overload`修饰的函数用于提供类型和重载的提示信息，但本身不具备实际功能，需要列出所有重载函数方式。
            参数中避免出现`RuntimeInt`等用户不感知的运行时参数。

            ```python
            @overload
            def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int) -> None:
                ...

            @overload
            def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
                    repeat_params: BinaryRepeatParams) -> None:
                ...


            @overload
            def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
                    repeat_params: BinaryRepeatParams) -> None:
                ...

            ```
        - 实现主函数

            使用`@require_jit`标记函数，表示需要JIT编译。`@set_docstring`自动生成代码解释文档。

            在主函数中，获取IR构建器 `builder = global_builder.get_ir_builder()`，并调用`op_impl`函数传递相应的参数和IR构建方法。

            ```python
            @require_jit
            @set_docstring("sum", cpp_name="Add")
            def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
                # 按照Ascend C的函数定义入参需要支持位置参数args，即不需要显示声明传入的参数
                builder = global_builder.get_ir_builder()      # 全局管理器，用于生成ast节点以及生成ir
                op_impl("add", dst, src0, src1, args, kwargs, builder.create_asc_AddL0Op, builder.create_asc_AddL2Op)
                # builder.create_asc_xxxOp 代表对应的ir创建方法，需要确保ir处定义了对应op，并且参数对应
                # build_l0 和 build_l2 对应双目指令不带mask和带mask两种计算模式
            ```

        - 实现`op_impl`

            使用`OverloadDispatcher`自动识别和处理不同的参数组合。
            ```python
            dispatcher = OverloadDispatcher(callee)
            @dispatcher.register_auto # register_auto表示自动识别参数列表
            def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams): # RuntimeInt表示运行时参数
                build_l0(dst.to_ir(), src0.to_ir(), src1.to_ir(),
                        _mat(mask,  KT.uint64).to_ir(),
                        _mat(repeat_times, KT.int8).to_ir(), params.to_ir())

            @dispatcher.register_auto
            def _(mask: list, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams):
                mask = [_mat(v,  KT.uint64).to_ir() for v in mask]
                build_l0(dst.to_ir(), src0.to_ir(), src1.to_ir(), mask,
                        _mat(repeat_times, KT.int8).to_ir(), params.to_ir())

            @dispatcher.register_auto
            def _(count: RuntimeInt):
                build_l2(dst.to_ir(), src0.to_ir(), src1.to_ir(), _mat(count, KT.int32).to_ir())

            dispatcher(*args, **kwargs)
            ```

            将接口中定义的参数转换成IR，并调用builder创建IR。
            ```python
            build_l0(dst.to_ir(), src0.to_ir(), src1.to_ir(),  # Tensor等类定义了to_ir()方法
                        resolve_vector_mask(mask).to_ir(),
                        _mat(repeat_times, KnownTypes.int8).to_ir(), params.to_ir())
                    # repeat_times等基础数据通过 _mat(repeat_times, KnownTypes.int8).to_ir() 包装成ir
            ```
    3. 在`python\asc\language\basic\__init__.py` 以及 `python\asc\language\__init__.py` 中增加引用。

  - 定义类

    Add需要LocalTensor存放数据，完整实现请参考<a href="../python/asc/language/core/tensor.py">tensor.py</a>。

    实现步骤如下：

    1. 根据类别选择合适的文件夹，LocalTensor为核心数据类型，对应文件为`python\asc\language\core\tensor.py`。

    2. 实现类

        新增接口方式与上部分类似。此外需要添加`to_ir`以及`from_ir`方法以支持IR的生成。

        - 编写初始化函数。

            ```python
            class LocalTensor(BaseTensor):

            def __init__(self, *args, **kwargs) -> None:
                dispatcher = OverloadDispatcher(__name__)

                # 当引入 from __future__ import annotations 类型评估会被延迟评估，需要手动指定重载参数类型
                @dispatcher.register(dtype=DataType, pos=Optional[TPosition], addr=RuntimeInt, tile_size=RuntimeInt)
                def _(dtype: DataType, pos: Optional[TPosition] = TPosition.VECIN, \
                        addr: RuntimeInt = 0, tile_size: RuntimeInt = 0):   # 根据Ascend C的函数定义确定是否需要赋默认值
                    super(LocalTensor, self).__init__(dtype)
                    builder = global_builder.get_ir_builder()
                    self.handle = builder.create_asc_LocalTensorV2Op(ir.get_local_tensor_type(dtype.to_ir()), \
                                        ir.TPosition.symbolize(pos), _mat(addr, KnownTypes.uint32).to_ir(),\
                                        _mat(tile_size, KnownTypes.uint32).to_ir())
                    # 当函数有返回值时，builder.create_asc_xxxOp第一个参数是返回值的类型
                    # 当返回值是类时，通过ir._get_xxx_type获取，当返回值是数据是通过builder.get_xxx_type获取
                    # 枚举类型Position通过ir.TPosition.symbolize(pos) 包装

                dispatcher(*args, **kwargs)
            ```

        - 编写`to_ir`以及`from_ir`方法。
            ```python
            @classmethod
            def from_ir(cls, handle: IRHandle) -> LocalTensor:
                ir_type = handle.get_type()
                return cls(handle, DataType.from_ir(ir.get_element_type(ir_type)), ir.get_shape(ir_type))
                #调用类本身的初始化方法，将IR转化成对象

            def to_ir(self) -> IRHandle:
                return self.handle
            ```
    3. 在`python\asc\language\core\__init__.py` 以及 `python\asc\language\__init__.py` 中增加引用。

  - 定义结构体

    双目指令提供了切分计算方式，需要 `BinaryRepeatParams`用于控制操作数地址步长，完整实现请参考 <a href="../python/asc/language/core/types.py">types.py</a>。

    实现步骤如下：

    1. 结构体定义在`python\asc\language\core\types.py`下。与高阶API相关的则定义在`python\asc\language\adv\types.py`。

    2. 根据Ascend C定义结构体。

        - 定义结构体以及初始化方法。
            ```python
            class BinaryRepeatParams(IRValue):     # 定义的类都需要继承IRValue

            def __init__(self, dst_blk_stride: RuntimeInt = 1, src0_blk_stride: RuntimeInt = 1, src1_blk_stride: RuntimeInt = 1, \
                        dst_rep_stride: RuntimeInt = 8, src0_rep_stride: RuntimeInt = 8, src1_rep_stride: RuntimeInt = 8,
                        handle: Optional[IRHandle] = None) -> None:             # 默认值与Ascend C中定义的一致
            ```

        - 获取IR构建器 `builder = global_builder.get_ir_builder()`，调用handler创建IR。
            ```python
            self.handle = builder.create_asc_ConstructOp(       # 使用 builder.create_asc_ConstructOp 创建结构体
                builder.get_asc_BinaryRepeatParamsType(),       # 第一个参数是结构体类型
                [                                               # 第二个参数是结构体中数据的ir列表
                    _mat(dst_blk_stride).to_ir(), _mat(src0_blk_stride).to_ir(),
                    _mat(src1_blk_stride).to_ir(), _mat(dst_rep_stride).to_ir(),
                    _mat(src0_rep_stride).to_ir(), _mat(src1_rep_stride).to_ir(),
                ],
                builder.get_type_array_attr([builder.get_ui8_type()] * 6), # 第三个参数是数据类型列表
            )
            ```
        - 编写`to_ir`以及`from_ir`方法。

    3. 在`python\asc\language\core\__init__.py` 以及 `python\asc\language\__init__.py` 中增加引用。

  - 定义枚举

    `TPosition`类用于确定`LocalTensor`的存放位置，实现步骤如下：

    1. 枚举类型定义在`python\asc\language\core\enums.py`下。

    2. 继承`IntEnum`, 根据Ascend C定义给枚举值赋整数值。

        ```python
        class TPosition(IntEnum):
            GM = 0
            A1 = 1
            A2 = 2
            B1 = 3
            B2 = 4
            C1 = 5
            C2 = 6
            CO1 = 7
            CO2 = 8
            VECIN = 9
            VECOUT = 10
            VECCALC = 11
        ```

    3. 在`python\asc\language\core\__init__.py` 以及 `python\asc\language\__init__.py` 中增加引用。

#### UT测试

在完成Python前端模块后，可以添加相应的UT测试模块，验证功能的正确性。
本部分分为目录结构、UT测试实现框架、示例代码解释三部分，说明前端接口UT测试的开发方式。

##### 目录结构

与Python前端接口目录结构相同，UT测试目录结构如下：

```
python/test/unit/language/adv       #对应高阶API的UT测试
python/test/unit/language/basic     #对应基础API的UT测试
python/test/unit/language/core      #对应数据类型定义的UT测试
python/test/unit/language/fwk       #对应内存管理的UT测试
```

按照不同的类别或功能，在每个目录下对应的测试文件中，进行Python前端的UT测试编码。例如在`fwk`目录下，测试被细分为以下几个模块：

- `test_tbuf.py`：TBuf类功能测试
- `test_tbuf_pool.py`：TBufPool类功能测试
- `test_tpipe.py`：TPipe类功能测试
- `test_tque.py`：TQue类功能测试
- `test_tque_bind.py`：TQueBind类功能测试


根据待添加UT测试的接口类别，选择合适的文件或者在合适目录下新增文件，进行UT编码。

##### UT测试实现框架

本模块基于`pytest`框架构建，整体实现过程如下：
- 初始化环境，注入`mock_launcher_run`桩函数；
- 定义内核函数，调用被测对象及方法；
- 触发内核运行；
- 通过断言验证编译以及执行流程是否按照预期触发。

##### UT测试代码示例

本部分将以实现双目矢量指令add的UT测试为例，介绍不同接口的UT编码方式。

add是一个支持多种重载形式的双目矢量运算指令，其函数原型如下：

```python
@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int) -> None:
    ...

@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: AnyMask, repeat_times: int,
        repeat_params: BinaryRepeatParams) -> None:
    ...
```

  1. 根据函数类型选择合适的文件编码，如add属于双目矢量指令，则在`python\test\unit\language\basic\test_vector_binary.py`文件中编写UT测试代码。
    
  2. 编写测试函数

    ```python
    def test_add_kernel(mock_launcher_run):

        @asc.jit
        def add_kernel():
            x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
            y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
            z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
            asc.add(z_local, x_local, y_local, count=512)
            params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
            asc.add(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
            uint64_max = 2**64 - 1
            mask = [uint64_max, uint64_max]
            asc.add(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

        add_kernel[1]()
        assert mock_launcher_run.call_count == 1

    ```

  3. 执行测试。在完成UT测试编码后，以双目矢量指令为例，在项目根目录执行如下命令进行UT单元测试。运行未报错说明UT测试通过。

      ```bash
      pytest ./python/test/unit/language/basic/test_vector_binary.py
      ```
    

### AST转ASC-IR模块
若在AST转ASC-IR的过程中，AST中的节点语法结构无法支持（当前已支持的语法结构参见下方接口列表），则可能涉及需要在AST语法树遍历类内新增节点处理接口。

- 代码文件：python\asc\codegen\function_visitor.py。

- 代码开发示例：

  - 以支持双目运算符表达式（如 *、+、-）为例，定义形如visit_Xxxxx的函数，根据运算逻辑处理对应节点。

    ```python
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        method_name = self.get_binary_method_name(type(node.op))
        return self.apply_binary_method(method_name, lhs, rhs)
    ```

- 已支持的语法接口列表：

  ```python
  visit_arguments
  visit_arg
  visit_region
  visit_statements
  visit_AnnAssign
  visit_Assert
  visit_Assign
  visit_AugAssign
  visit_Attribute
  visit_BinOp
  visit_BoolOp
  visit_Call
  visit_Compare
  visit_Constant
  visit_Expr
  visit_For
  visit_FormattedValue
  visit_FunctionDef
  visit_If
  visit_IfExp
  visit_JoinedStr
  visit_keyword
  visit_List
  visit_Name
  visit_Pass
  visit_Return
  visit_Slice
  visit_Subscript
  visit_Tuple
  visit_UnaryOp
  visit_With
  visit_While
  ```

如果新增了语法处理接口，优先通过TableGen工具自动生成pybind绑定代码，若自动生成代码无法满足需求，则需要在[pybind11绑定定义](../python/src/OpBuilder.cpp)中手动添加对应IR创建接口。

- 代码文件：

  - 自动生成：tools\ascir-tblgen\GenPybindDefs.cpp。
  - 手动添加：python\src\OpBuilder.cpp文件内的pyasc_init_ir_builder函数。

- 代码开发示例：

  - OpBuilder.cpp 增加TPipe初始化IR创建接口。

    ```python
    .def("create_asc_PipeOp", [](PyOpBuilder &self) -> Value {
    		    return self.create<ascendc::PipeOp>();
      })
    ```

其他情况下，本模块理论上不涉及修改。

### ASC-IR定义模块

ASC-IR是基于[MLIR](https://mlir.llvm.org/)定义的Dialect（方言）。由于ASC-IR基于LLVM中的TableGen工具翻译为MLIR，所以ASC-IR基于[TableGen语法](https://llvm.org/docs/TableGen/)编写。

注：下述全部文件的目录均位于`pyasc仓根目录/include/ascir/Dialect/Asc/IR`。
#### ASC-IR编写规范与约束
本模块主要基于APIOp类开发ASC-IR，APIOp类继承自位于`Base.td`目录的AscendC_Op类，AscendC_Op类的定义如下。
```
class AscendC_Op<string mnemonic, list<Trait> traits = []>
    : Op<AscendC_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::ascendc";
  let assemblyFormat = "operands attr-dict `:` qualified(type(operands))";
  bit genEmitter = !foldl(
    0, traits, init, trait,
    !or(init,
        !eq(trait, AscConstructor),
        !eq(trait, AscMemberFunc),
        !eq(trait, AscFunc)
    )
  );
  list<int> paramTypeLists = [];
}
```
- `cppNamespace` 是用户自定义开发IR的命名空间，默认使用 `"::mlir::ascendc"`。
- `assemblyFormat`中 `operands`表示操作数列表，`qualified(type(operands))`表示操作数类型。
- `genEmitter`是自动化生成AscendC代码的开关，其中：
    - `AscConstructor`表示构造函数。
    - `AscMemberFunc`表示成员函数。
    - `AscFunc`表示普通函数。
  如果出现任一上述函数，则会自动生成对应的代码。
- `paramTypeLists` 定义了各个`operands`对应的参数。
  为了自动化处理，我们规定类型模板参数不作为模板参数，普通参数直接作为函数参数传参。
  此外，我们还规定了一些参数需要处理为模板参数：
  - 从该参数中提取模板类型(`e.g., <typename T> from function(T arg)`)。
  - 从模板类型中提取元素类型(`LocalTensor<T> -> T`)。
  - 非类型模板参数：枚举值`enum value`。
  - 非类型模板参数：常规值`regular value`。

APIOp类是Operation类模板，用于开发Ascend C API所对应的ASC-IR。在如下`AscendC_GlobalTensorGetPhyAddrOp`的示例中，说明APIOp类的开发规则。
```
class APIOp<string mnemonic, string apiName, list<Trait> traits = []>
    : AscendC_Op<mnemonic, [APIOpInterface] # traits> {
    ...
}

def AscendC_GlobalTensorGetPhyAddrOp
    : APIOp<"global_tensor.get_phy_addr", "GetPhyAddr", [AttrSizedOperandSegments]> {
  let summary = "Call `AscendC::GlobalTensor::GetPhyAddr` method";
  let arguments = (ins AscendC_GlobalTensor:$tensor, Optional<UI64>:$offset);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = [{
    $tensor (`,` $offset^)? attr-dict `:` qualified(type($tensor))
    `,` qualified(type($result)) (`,` type($offset)^)?
  }];
}
```
ASC-IR中的Operation类命名规则：AscendC + 下划线 + Ascend C类名 + 成员函数，示例中为`AscendC + _ + GlobalTensor + GetPhyAddr + Op`。
根据APIOp类的模板，在AscendC_GlobalTensorGetPhyAddrOp中：
  - `mnemonic` 为 `global_tensor.get_phy_addr`，规则为 "Ascend C类名 + . + 成员函数"。
  - `apiName` 为`GetPhyAddr`，实际开发中建议参考Ascend C API的命名。
  - `traits` 为`[AttrSizedOperandSegments]`，可以用","分割需要填入的MLIR的[traits特征](https://mlir.llvm.org/docs/Traits/)，例如`AttrSizedOperandSegments`表示支持多个`Optional`入参。

在GlobalTensorGetPhyAddr的Operation类中定义接口，以下接口根据API功能可选实现，建议至少填入` summary `提高调试和二次开发的效率。
  - ` summary ` 用于存储用户自定义的注释。
  - ` arguments ` 入参定义，可选参数用`Optional/OptionalAttr<ParamType>`表示，`AnyType`表示根据入参自动推导的变量类型。
    由于是1:1映射Ascend C API，且为了自动化映射代码，所以arguments需要遵守映射规则：
    - 类型模板参数不需要映射在IR中，非类型模板参数放函数参数后面。
    - 整体顺序如下：必选参数（**函数参数、模板参数**），可选参数（**函数参数、模板参数**）。
    - 参数名建议和API保持一致。
  - ` results `  表示出参定义，不支持可选。
  - ` assemblyFormat ` 表示匹配格式，代码格式为"变量名:推导类型"，其中``` (`,` $xxx^)? ``` 表示与optional对应的可选参数。

此外，由于当前op如果不存在特殊的[builder](https://mlir.llvm.org/docs/DefiningDialects/Operations/#builder-methods)会自动生成pybind的代码。同时，对于未定义的API Type类型会自动生成def类型定义和后端发射代码，建议新增简单结构体类型基于API Type进行定义。
  - 对于类似`Core/Types.td`中的简单Type类型Mask，当前op会自动生成继承`AscendC_Type`的代码。
    ```
    def AscendC_Mask : AscendC_Type<"Mask", "mask"> {
      let summary = "Represents vector mask (bit mode)";
    }
    ```
  - 对于复杂的Type类型，则需要基于API Type进行定义。
    - `./Core/Attributes.td`中的枚举类：
      ```
      def AscendC_AddressSpaceAttr : I32EnumAttr<"AddressSpace", "", [
        I32EnumAttrCase<"Default", 0>,
        I32EnumAttrCase<"gm", 22>,
        I32EnumAttrCase<"ca", 23>,
        I32EnumAttrCase<"cb", 24>,
        I32EnumAttrCase<"cc", 25>,
        I32EnumAttrCase<"ubuf", 26>,
        I32EnumAttrCase<"cbuf", 27>,
        I32EnumAttrCase<"fbuf", 28>,
      ]> {
        let cppNamespace = "::mlir::ascendc";
        let description = "CCE-C address space";
        let underlyingType = "uint8_t";
      }
      ```
    - 结构体类型：
    以`Core/Types.td`中queue为例：
      ```
      def AscendC_Queue : AscendC_BaseQueueType<"Queue", "queue"> {
        let description = "Represents AscendC::TQue";
        let parameters = (ins "TPositionAttr":$tPositionAttr, "int64_t":$depth);
        let assemblyFormat = [{
          `<` custom<PrettyTPosition>($tPositionAttr) `,` $depth `>`
        }];
        let builders = [
          TypeBuilder<(ins "TPosition":$position, "int64_t":$depth), [{
            return $_get($_ctxt, TPositionAttr::get($_ctxt, position), depth);
          }]>,
        ];
        let extraClassDeclaration = [{
          TPosition getPosition() { return getTPositionAttr().getValue(); }
        }];
      }
      ```

#### ASC-IR开发示例
本模块已根据Ascend C API用法提供了相关的模板，这些模板在`interfaces.td`目录下。对于符合模板的API，开发者可以直接使用模板开发，不符合模板的API，开发者需要自行定制化开发。根据是否符合模板，分别进行如下内容介绍。
- Ascend C API符合模板：
  以Ascend C API双目指令add为例，介绍ASC-IR的开发过程。
  `interfaces.td`中实现了双目指令的Operation类参数模板`BinaryOpInterface`，用于匹配Ascend C API中双目指令的参数用法。
  ```
  def BinaryOpInterface
      : AscendC_OpInterface<"BinaryOp", [VectorOpInterface, OpWithDstInterface]> {
    let description = "Vector binary operation";
    let methods = [getSrc0, getSrc1];
  }
  def BinaryL3OpInterface : AscendC_OpInterface<"BinaryL3Op", [BinaryOpInterface]> {
    let description = "Vector binary operation (L3 API)";
  }
  ```
  `base.td`的`BinaryOp`模板继承自`BinaryOpInterface`。其中：`L0Op`表示所有参数都使用的Operation类模板，此处是6个参数，`L2Op`表示缺失2个参数用法的Operation类模板，此处是4个参数，同理，`L3Op`表示3个参数的Operation类模板；`L02Op`则表示`L0Op`和`L2Op`两种用法结合的Operation类模板，以此类推，`L023Op`表示L0、L2、L3三种用法结合的Operation类模板。
  ```
  class BinaryL0Op<string mnemonic, string apiName, list<Trait> traits = []>
      : BinaryOp<mnemonic, apiName, traits> {
    let description = "`AscendC::" # apiName # "` is a vector binary operation (L0 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                    AnyType:$mask, AnyType:$repeatTimes, 
                    AscendC_BinaryRepeatParams:$repeatParams);
  }
  class BinaryL1Op<string mnemonic, string apiName, list<Trait> traits = []>
      : BinaryOp<mnemonic, apiName, traits> {
    let description = "`AscendC::" # apiName # "` is a vector binary operation (L1 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                    Variadic<UI64>:$mask, AnyType:$repeatTimes, 
                    AscendC_BinaryRepeatParams:$repeatParams);
  }
  class BinaryL2Op<string mnemonic, string apiName, list<Trait> traits = []>
      : BinaryOp<mnemonic, apiName, traits> {
    let description = "`AscendC::" # apiName # "` is a vector binary operation (L2 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1,
                    AnyType:$calCount);
  }
  class BinaryL3Op<string mnemonic, string apiName, list<Trait> traits = []>
    : BinaryOp<mnemonic, apiName, [BinaryL3OpInterface] # traits> {
    let summary = "Call `LocalTensor::" # apiName # "` method";
    let description = "`LocalTensor::" # apiName # "` performs a vector binary operation (L3 API).\n";
    let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1);
  }
  multiclass BinaryL02Op<string baseMnemonic, string apiName, list<Trait> traits = []> {
    def L0Op : BinaryL0Op<baseMnemonic # "_l0", apiName, traits>;
    def L2Op : BinaryL2Op<baseMnemonic # "_l2", apiName, traits>;
  }
  multiclass BinaryL023Op<string baseMnemonic, string apiName, string l3operator,
                    list<Trait> traits = []> {
    defm "" : BinaryL02Op<baseMnemonic, apiName, traits>;
    def L3Op : BinaryL3Op<baseMnemonic # "_l3", l3operator, traits>;
  }
  ```
  基于Ascend C API中的Add函数有L0、L2、L3的用法，所以选择BinaryL023Op，用BinaryL023Op模板按照格式补充ASC-IR。
  ```
  defm Add : BinaryL023Op<"add", "Add", "operator+">;
  ```
- Ascend C API不符合模板：
  - 以Ascend C API双目指令Gather为例，介绍ASC-IR的开发过程。
    查看Ascend C API中Gather的用法：
    ```
    template <typename T>
    __aicore__ inline void Gather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint64_t mask[],  const  uint8_t repeatTimes, const uint16_t dstRepStride)
    ```
    按照函数的参数格式补充IR。
    ```
    def AscendC_GatherL0Op : VectorOp<"gather_l0", "Gather", [OpWithDstInterface]> {
      let description = "Gather elements from source tensor (L0 API)";
      let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                          AscendC_LocalTensor:$srcOffset, AnyType:$srcBaseAddr,
                          AnyType:$mask, AnyType:$repeatTimes,
                          AnyType:$dstRepStride);
    }
    ```
  - 以Ascend C API中TQue中的AllocTensor为例，介绍ASC-IR的开发过程。
    查看AllocTensor函数用法：
    ```
    AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
    ```
    按照函数的参数格式补充IR，根据规范需要在入参中加入que本身。
    ```
    def AscendC_TQueBindAllocTensorOp : APIOp<"alloc_tensor", "AllocTensor"> {
      let summary = "Allocate tensor on queue wrapped buffer";
      let arguments = (ins AscendC_BaseQueueTypeInterface:$queue);
      let results = (outs AscendC_LocalTensor:$tensor);
      let assemblyFormat = [{
        $queue attr-dict `:` qualified(type($queue)) `,` qualified(type($tensor))
      }];
    }
    ```


### Ascend C代码生成模块
在Ascend C代码生成模块中增加生成该接口的方式，保证ASC-IR能正确生成Ascend C代码。
#### 目录结构
    ├── include/
    │   ├── ascir/
    │   │   ├── Target/                // 目标平台相关接口声明
    │   │   |   └──Asc/                // Ascend C代码生成相关接口声明
    │   │   │      ├── Adv/         // 高阶API相关接口声明
    |   │   │      ├── Basic/       // 基础API相关接口声明
    |   │   │      └── .../
    ├── lib/
    │   ├── Target/                    // 目标平台相关接口实现
    │   │   ├── AscendC/               // Ascend C代码生成相关接口实现
    │   │   │   ├── CMakeLists.txt
    │   │   │   ├── Translation.cpp    // 代码生成模块入口
    │   │   │   ├── Adv/            // 高阶API相关接口实现
    |   │   │   ├── Basic/          // 基础API相关接口实现
    |   │   │   └── .../
    ├── test/
    │   ├── Target/                    // 目标平台相关接口测试
    │   │   ├── AscendC/               // Ascend C代码生成模块的单元测试
    │   │   │   ├── ascendc.mlir       // 单元测试MLIR文件
    │   │   │   └── ...
    └── ...
#### 开发步骤
以新增GlobalTensor.GetPhyAddr()接口为例，介绍Ascend C代码生成模块的开发步骤。  
首先，需要在[Translation.cpp](../lib/Target/AscendC/Translation.cpp)中的PrintableOpTypes中注册新增的接口，如将ascendc::GlobalTensorGetPhyAddrOp追加注册在已有接口的后面。其中，Op的命名由ASC-IR中的定义决定。
```cpp
// GlobalTensor operations
ascendc::GlobalTensorSubIndexOp, ascendc::GlobalTensorSetGlobalBufferOp, ascendc::GlobalTensorGetPhyAddrOp,
```
然后，根据新增接口的类型，在include\ascir\Target\Asc\目录下对应的头文件中，增加接口声明。例如，在Core\GlobalTensor.h文件中增加如下接口声明。
```h
LogicalResult printOperation(CodeEmitter& emitter, ascendc::GlobalTensorGetPhyAddrOp op);
```
最后，根据新增接口的类型，在lib\Target\AscendC\目录下对应的cpp文件中，增加接口实现，例如，在Core\GlobalTensor.cpp中增加如下接口实现。其中getTensor、getOffset等接口是ASC-IR中定义的接口，用于获取对应节点中的值。
```cpp
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GlobalTensorGetPhyAddrOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = " << emitter.getOrCreateName(op.getTensor()) << "." << op.getAPIName() << "(";
    if (auto offset = op.getOffset()) {
        os << emitter.getOrCreateName(offset);
    }
    os << ")";
    return success();
}
```


## 编码规范
### python代码编码规范
遵守python官方代码风格pep8，具体要求参见 [pep主页](https://peps.python.org/pep-0008/)。

在代码上库前使用`ruff`和`yapf`工具包，自动规范代码。
```bash
ruff check --fix .\python\asc       # windows下路径
yapf -i --parallel -r .\setup.py
```

### C++代码编码规范
基本参考LLVM代码风格，在此基础上根据项目特点做出一些约束，具体要求见 [CodeStyle](./codestyle.rst)。

### Python前端模块编码规范
#### 命名调整
由于Python和C++遵守不同的命名规范，因而我们需要修改python侧相关接口的名称来符合相关规范，典型的修改规则如下表所示：
||Ascend C命名|pyasc命名|
|:-----------:|:-----------:|:-------------:|
|包/模块(namespace)|CapWords|lower_with_under|
|类|CapWords|CapWords|
|函数|CapWords|lower_with_under|
|变量|capWords(全局变量前面加上g_前缀)|lower_with_under|
|常量|CAPS_WITH_UNDER|CAPS_WITH_UNDER|

#### 接口重载规范
python基础语法不支持C++中的函数重载，对于需要重载的函数，pyasc使用@overload装饰所有对外API，使用@require_jit装饰实际调用Pybind创建MLIR节点的函数实现。下面为Add双目向量预算接口的C++接口和python接口对比：

***Ascend C Add 接口***
```cpp
// tensor前n个数据计算
template <typename T>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)

// mask逐bit模式
template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)

// mask连续模式
template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

***pyasc add 接口***
```python
@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, count: int) -> None:
    ...

@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: int, repeat_times: int,
        repeat_params: BinaryRepeatParams) -> None:
    ...

@overload
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, mask: List[int], repeat_times: int,
        repeat_params: BinaryRepeatParams) -> None:
    ...

@require_jit
@set_binary_docstring("sum", cpp_name="Add")
def add(dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    op_impl("add", dst, src0, src1, args, kwargs, builder.create_asc_AddL0Op,
            builder.create_asc_AddL1Op, builder.create_asc_AddL2Op)
```


#### 模板参数处理规则
##### DataType模板参数
对于Tensor的数据类型，在python类型的AI编程语言中，一般将DataType作为Tensor的成员变量，调用计算接口时根据Tensor的DataType成员变量推导实际类型的计算接口，pyasc也使用这种方式处理DataType，因而Ascend C的API中涉及DataType的模板参数均通过Tensor参数进行推导，不在pyasc的接口中显示定义该参数。
##### 非DataType模板参数
对于非DataType模板参数，由于python中不存在*模板*，因而按照以下规则统一完成Ascend C模板参数到pyasc的映射：
1. 模板参数修改为运行时参数；
2. 如果模板参数是枚举、bool等常量类型，可以直接使用；如果是非常量类型，则加上asc.ConstExpr[origin_type]泛型标记，表明为参数应该传入常量；
3. 由于模板参数和运行时参数均可能存在*必选参数*和*可选参数*，而python语法要求*可选参数*必须放在*必选参数*后，因而我们按照以下顺序重新组装pyasc的参数：*运行时必选参数*， *模板必选参数*， *运行时可选参数*， *模板可选参数*。参见下面的例子：

***Ascend C接口***
```cpp
tempate <typename T, bool tplRequire, bool tplOptional = false>
__aicore__ inline void Func(const LocalTensor<T>& rtRequire, int32_t rtOptional=128);
```

***pyasc接口***
```python
def func(rt_require: LocalTensor, tpl_require: bool, rt_optional: int = 128, tpl_optional: bool = False):...
```
##### 其他模板处理规则
- 默认模板参数
Ascend C接口中存在的默认值，pyasc应该保持同样的默认值。
- 非类型模板参数
用来指定模板中一些常量值的参数，在pyasc中引入对应的枚举或者常量定义。

- 类型模板参数
模板中非常量的class参数，在C++中使用typename或class标记。
如果该class的字段在C++中全部通过模板参数指定，在pyasc中使用不可变数据类@dataclass(frozen=True)定义，该类型的实例一旦创建就不可修改。
如果该class的字段中有部分非模板参数指定，则在pyasc中通过Generic进行封装。

- 模板模板参数
与类型模板参数规则一致，递归进行处理内部模板。
- 可变模板参数
模板参数转为运行时参数，利用python的变长参数能力进行支持。

- 模板特化
在pyasc中通过运行时参数的实际类型进行分发。
