# dateline_unwrap — 球面环线的日期变更线展开

单位球面闭环的**有向面积**计算与**日期变更线（经度 ±180° 半子午线）展开**。
每条边取相邻顶点间唯一的**较短大圆弧**；所有真实几何（弧上取点、求交、
面积）都在单位球面三维向量上完成，**不对经纬度做平面线性插值**，因此不会
在切缝附近制造假交点。

- 运行环境：**Python 3.14.7**，仅用标准库（`math`、`dataclasses`、`unittest` 等）。
- 安装 Python 后**无需安装任何第三方依赖**，离线运行；不需要网络、账号、
  密钥、数据库、中间件、云 API 或 Docker。

## 快速开始

```bash
python -m unittest discover -s tests -v   # 必要测试
python demo.py                            # 固定输入演示（亚秒级）
```

## 接口

```python
from dateline_unwrap import unwrap_ring

result = unwrap_ring([(lon0, lat0), (lon1, lat1), ...])
result.area                  # 有向面积（弧度²），左转为正，范围 (-2π, 2π]
for edge in result.edges:
    edge.arc.edge_index      # 边下标：边 i 从顶点 i 到顶点 i+1（末边回顶点 0）
    edge.arc.length          # 较短大圆弧长（弧度，0 < length < π）
    edge.arc.azimuth         # 起点初始方位角（弧度，正北=0，顺时针，东=π/2）
    edge.arc.plane_normal    # 弧所在大圆平面单位法向 (x,y,z)，左侧 dot>0
    for seg in edge.segments:
        seg.kind             # "normal" | "dateline" | "ambiguous" | "coincident"
        seg.t0, seg.t1       # 片段在原弧 t∈[0,1] 上的起止位置
        seg.lon_a, seg.lon_b # normal 片段两端的连续展开经度（弧度，可超出 [-π,π]）
        seg.point_a, seg.point_b  # 端点单位球坐标 (x,y,z)
        seg.pole             # ambiguous: "north"/"south"，否则 None
result.segments              # 全部片段的扁平视图
# 只要面积：
from dateline_unwrap import area_of_ring
```

角度约定：对外经纬度均为**弧度**；经度域 `[-π, π]`（`+π` 与 `-π` 是切缝
同一条半子午线），纬度域 `[-π/2, π/2]`。笛卡尔约定
`x=(0°,0°), y=(90°E,0°), z=北极`。

### 片段语义

| kind | 含义 |
|---|---|
| `normal` | 不跨缝的连续经度片段；`lon_a → lon_b` 单调连续，跨度 ≤ π，可直接渲染 |
| `dateline` | 沿 ±180° 切缝走向的子弧（共面边的一部分）；无经度，不得跨缝连线 |
| `ambiguous` | 弧与切缝在**极点**相交：退化为 `t0==t1` 的单点标注，给出 `pole`。极点经度无定义，本库不替渲染端决定它与相邻片段如何连接 |
| `coincident` | **整条原弧**就在切缝半子午线上；保留原弧参数原样输出，`t∈[0,1]`，不生成任何跨缝连接 |

### 错误语义（均可定位，不静默纠正）

| 异常 | 触发 | 定位字段 |
|---|---|---|
| `InvalidCoordinateError` | 非有限实数（NaN/Infinity）、`bool`、非数、经度越界、纬度越界 | `vertex_index`, `field`（`"lon"`/`"lat"`） |
| `RingSizeError` | 顶点数不在 **[3, 100]**、容器类型错误 | — |
| `DegenerateEdgeError` | 相邻顶点重合，较短大圆弧不存在 | `edge_index` |
| `AntipodalEndpointError` | 相邻顶点对跖，大圆弧有两条半大圆、不唯一 | `edge_index` |
| `SelfIntersectionError` | 非相邻边相交，或相邻边在公共顶点之外接触（含共面重叠、非相邻顶点落在另一边上） | `edge_i`, `edge_j`, `point` |
| `ComputationBudgetError` | 面积辅助枢轴的 65+ 个确定性候选全部数值退化（预算耗尽，**不等于**输入无解/非法） | — |

前五个均继承 `SphericalLoopError`（`ValueError` 子类）。计算是纯函数、
无状态、不修改输入；没有批次/部分提交语义，因此不存在未说明的部分变更。

## 算法与设计取舍

1. **校验与转换**：严格类型/范围检查（先排 `bool`，再排 NaN/Infinity）；
   经纬度转单位向量时显式再归一化，保证相同输入 bit 级一致。
2. **边长/方向**：用 `atan2(|a×b|, a·b)` 求稳定角距（近重合/近对跖时
   `acos(dot)` 不稳定）；法向 `normalize(a×b)`，左侧半球 `dot(n,p)>0`。
3. **简单性检查**：O(n²)（n≤100）枚举边对。两条大圆的交点为
   `±normalize(n_i × n_j)`，再用“短弧 = 两非负锥系数”的稳定判据
   判定交点是否落在两条较短弧上；共面边对单独检查重叠。
4. **有向面积**：以辅助枢轴 q 做有向扇形三角化
   `Σ 2·atan2(det(q,a,b), 1+q·a+a·b+b·q)`（Oosterom–Strackee 立体角
   公式），对极小三角形无 L'Huilier 四 tan 连乘的灾难性抵消。q **优先取
   环自身顶点**（小多边形时扇形三角形就是多边形本身），退化时按确定性
   序列尝试其余顶点与 65 个通用点；全退化抛 `ComputationBudgetError`。
   有向扇形和与 q 无关（绕数恒等），结果模 4π 折叠进 `(-2π, 2π]`。
5. **切缝分裂**：边所在大圆与切缝大圆（法向 y 轴）求交，只保留落在
   ±180° **半**子午线（`x≤0, y=0`）且在短弧上的点；非极点横穿点把边
   切成连续经度片段，边界经度极限取交点两侧真实近点的所在侧（`±π`）；
   极点交点只产 `ambiguous` 单点，相邻片段各自以自己的趋近经度收尾。
   边大圆与切缝共面时单独处理：整段在缝上 → `coincident`；跨极点连到
   0° 子午线一侧 → 缝侧 `dateline` + 0° 侧 `normal` + 极点 `ambiguous`。

### 误差口径（不用宽容差掩盖边界错误）

- **面积**：在 1200 个确定性随机简单环（角半径 1e-5…约半球、3–12 边、
  随机中心含切缝/极点）上与两个独立预言比较，绝对误差 ≤ **8e-14 弧度²**；
  八分之一球解析值误差 4.4e-16；任意轴旋转前后面积差 ≤ 6e-15。
  独立预言为 `tests/reference.py` 中**不调用被测代码**的
  Gauss–Bonnet 外角和公式（结构独立，主预言）与独立重写的
  Oosterom–Strackee 扇形（不同固定枢轴交叉验证）。
- **极小三角形**：ε=1e-7 的赤道直角三角形面积相对误差约 2e-15。
- **拓扑判定容差**：共线/共面 1e-10、切缝归属 1e-12、短弧锥系数 -1e-9、
  事件参数去重 1e-11。这些只决定“如何分段/拒绝”，不用于涂抹面积数值；
  对跖按 atan2 稳定角距判定：边长与 π 之差小于 **1e-14 弧度** 时按对跖
  拒绝——再短于此，“哪条是较短弧”已无工程意义。
- **边界约定**：西向完整大圆的真实有向面积为 -2π，位于开区间端点；
  按模 4π 同余归并报告为闭端 **+2π**（数值尾巴容差 1e-12）。
- 随机旋转后的测试数据偶发对跖/共线退化时跳过该随机实例（测试数据
  问题，非被测行为）。

## 已知限制 / 非目标

- 只实现本题契约的数学子集：单位球面、较短大圆弧边、3–100 顶点的
  **简单**闭环；不宣称兼容 OGC/GeoJSON 等行业标准，不做椭球/大地测量。
- 不做环的套叠（内环/洞）、多部件、自交多边形的带符号绕数面积——
  自交一律拒绝。
- 极点处经度本就无定义：本库只标注 `ambiguous`，不规定地图投影端的
  具体连线方式。
- 输入为经纬角（弧度）；不做十进制度↔弧度之外的坐标参考系转换。
- 面积枢轴的确定性候选序列是内部实现细节，不在公开接口承诺范围内。

## 目录

```
dateline_unwrap/        本题库（errors.py / geometry.py / models.py / unwrap.py）
tests/                  必要单元测试 + 独立面积预言 reference.py（不 import 核心）
demo.py                 固定输入演示
README.md
requirements.txt        空（无第三方依赖）
```
