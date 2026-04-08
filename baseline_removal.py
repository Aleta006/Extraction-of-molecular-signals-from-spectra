# coding: utf-8
"""
基线去除（Baseline Removal）模块 - 红外光谱分析

使用AsLSS（Asymmetric Least Squares Smoothing）方法进行基线估计和去除。
该方法基于Eilers & Boelens的经典论文，结合Whittaker平滑器和非对称加权迭代。

主要特点：
  * Whittaker平滑器：使用惩罚最小二乘法，自动控制平滑度
  * 非对称加权：区分峰值和基线，防止过度平滑峰值
  * 特征窗口加权：可针对已知特征位置进行权重调整
  * 双模式校正：支持相减定义(y-baseline)和相除定义(y/baseline-1)

应用领域：
  * 红外(IR)光谱基线校正
  * 反射率光谱分析
  * 吸收光谱处理
  * 任何需要去除渐变背景的信号处理

关键文献：
  * Eilers, P.H.C. and Boelens, H.M.F. (2005): Baseline Correction with 
    Asymmetric Least Squares Smoothing
  * NIST publications (ac034173t.pdf)
  * Applied Optics special issue on nanoantennas (huck2015.pdf)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class BaselineRemoval:
    """
    基线去除类，实现Whittaker平滑器和AsLS/AsLSS算法。
    
    该类提供了一套用于光谱基线估计的静态方法，包括：
      * 差分矩阵构造（用于平滑约束）
      * 加权Whittaker平滑（独立平滑，无非对称性）
      * AsLS/AsLSS基线估计（含非对称加权迭代）
    
    典型使用流程：
      1. 加载光谱数据（2列：波长 | 反射率）
      2. 构造特征窗口权重向量
      3. 调用estimate_baseline_aslss()进行基线拟合
      4. 计算两种校正定义：相减和相除
      5. 提取目标波长处的特征指标
      6. 可视化结果
    """

    def __init__(self):
        """
        初始化基线去除对象。
        
        属性：
          lam: float
            Whittaker平滑参数(平滑度), 范围[1e2, 1e9]
            - 越大：基线越光滑，可能丢失细节
            - 越小：基线更贴近数据，可能包含噪声
            典型值：1e2-1e6用于红外反射率光谱
            
          p: float
            非对称加权参数, 范围[0.001, 0.1]
            控制峰值附近权重与基线附近权重的比例
            - p越小：对峰值的惩罚越轻，基线可能上升
            - p越大：对峰值的惩罚越重，基线越接近数据最小值
            典型值：0.01-0.1
        """
        self.lam = 1e5  # 1e2 - 1e9
        self.p = 1e-3   # 1e-3 - 1e-1

    @staticmethod
    def _diff_matrix(m: int, d: int = 2) -> sparse.csc_matrix:
        r"""
        构造有限差分矩阵（Finite-difference matrix）用于Whittaker平滑。
        
        数学原理：
        ----------
        Whittaker平滑器使用L2正则化来约束解的光滑性。差分矩阵D用于定义二阶差分约束：
        
        对于d=2（二阶差分），矩阵D的形状为(m-2, m)，每行表示一个差分操作：
        
            [1, -2, 1, 0, 0, ...]  <- δ²z₀ = z₀ - 2z₁ + z₂
            [0, 1, -2, 1, 0, ...]  <- δ²z₁ = z₁ - 2z₂ + z₃
            ...
        
        完整优化问题（约束最小二乘）：
        
            minimize: S = Σᵢ wᵢ(yᵢ - zᵢ)² + λ·(D·z)ᵀ(D·z)
        
        其中：
            - y: 原始信号
            - z: 平滑后信号/基线
            - w: 点权重向量
            - λ: 平滑参数，控制数据拟合度与平滑度的平衡
            - D: 二阶差分矩阵
            - (D·z)ᵀ(D·z)：二阶差分惩罚项，鼓励z的光滑性
        
        参数：
        ------
        m: int
            信号长度（采样点数），必须≥3
        d: int
            差分阶数，当前仅支持d=2（二阶差分）
        
        返回：
        ------
        sparse.csc_matrix
            形状为(m-2, m)的稀疏矩阵（CSC格式便于线性求解）
        
        异常：
        ------
        ValueError: 如果d≠2或m<3则抛出异常
        
        例子：
        ------
        >>> D = _diff_matrix(5, d=2)
        >>> D.toarray()
        array([[ 1., -2.,  1.,  0.,  0.],
               [ 0.,  1., -2.,  1.,  0.],
               [ 0.,  0.,  1., -2.,  1.]])
        """
        if d != 2:
            raise ValueError('Only d=2 is supported in this implementation')
        if m < 3:
            raise ValueError('Signal length must be >= 3')
        # sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], ...)
        # 创建具有对角线为[1,-2,1]的稀疏矩阵，实现二阶差分：
        # - 主对角线：+1
        # - 上对角线（偏移1）：-2
        # - 上对角线(偏移2)：+1
        return sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(m - 2, m), format='csc')

    @staticmethod
    def whittaker_smooth(y: np.ndarray, lam: float, weights: np.ndarray | None = None) -> np.ndarray:
        """
        加权Whittaker平滑（Weighted Whittaker Smoother）。
        
        Whittaker平滑是一个经典的信号处理方法，通过求解带约束的最小二乘问题来平滑数据。
        
        数学原理：
        ----------
        本方法求解如下优化问题（约束最小二乘）：
        
            minimize: S = Σᵢ wᵢ(yᵢ - zᵢ)² + λ·(D·z)ᵀ(D·z)
        
        其中：
            - y: 原始信号
            - z: 平滑后的信号/基线估计
            - w: 点权重向量（用户可指定不同点的置信度）
            - λ: 平滑参数（控制平滑与贴近原始数据的平衡）
            - D: 二阶差分矩阵
            - (D·z)ᵀ(D·z) = zᵀ(DᵀD)z: 二阶差分惩罚项
        
        闭式解（Closed-form solution）的推导：
        
        目标函数对z的导数：
            ∇S = -2·Wᵀ·(y - z) + 2λ·DᵀD·z = 0
        
        整理得法线方程（Normal Equation）：
            (W + λ·DᵀD)·z = W·y
        
        由于(W + λ·DᵀD)是稀疏三对角或五对角矩阵，可以高效地求解。
        
        参数：
        ------
        y: np.ndarray
            输入信号，形状为(m,)或(m,1)，m为信号长度
        lam: float
            平滑参数λ，范围通常为[1e2, 1e9]
            - λ=0: z=y（无平滑，完全贴近数据）
            - λ→∞: z为线性函数或常数（高度平滑）
            建议：对不同λ做对数网格搜索验证效果
            经验值：红外光谱通常选择1e2-1e6
        weights: np.ndarray | None
            点权重向量，形状为(m,)
            默认None时使用w=1（所有点等权）
            用途：
            - 强调高信噪比的点
            - 降低噪声或异常点的影响
            - 在特征窗口使用低权重以保护特征
            
        返回：
        ------
        np.ndarray
            平滑后的信号，形状为(m,)
        
        算法步骤：
        --------
        1. 构造对角矩阵W，其对角元素为权重向量w
        2. 构造二阶差分矩阵D（调用_diff_matrix）
        3. 构造系统矩阵：Z = W + λ·DᵀD (稀疏矩阵)
           - 此矩阵为正定对称矩阵（SPD），可用稀疏Cholesky分解求解
        4. 使用稀疏线性求解器求解：z = Z⁻¹(W·y)
           - scipy.sparse.linalg.spsolve: 自动选择最优求解器
           - 计算复杂度：O(m)（因为矩阵稀疏）
        
        数值稳定性应用举例：
        -------------------
        # 例1：标准平滑，所有点等权
        y = np.array([1.0, 1.1, 1.05, 3.0, 2.9, 1.0])  # 有噪声和异常值
        z = whittaker_smooth(y, lam=1e3)  # λ越大越平滑
        
        # 例2：降权噪声点
        w = np.array([1, 1, 1, 0.1, 0.1, 1])  # 位置3,4是异常值，给低权重
        z = whittaker_smooth(y, lam=1e3, weights=w)
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        m = len(y)
        D = BaselineRemoval._diff_matrix(m, d=2)
        if weights is None:
            w = np.ones(m, dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if len(w) != m:
                raise ValueError('weights must have the same length as y')
        # 构造对角权重矩阵W，其中W[i,i] = w[i]
        W = sparse.spdiags(w, 0, m, m)
        # 构造系统矩阵：Z = W + λ·DᵀD
        # DᵀD是 (m-2,m)ᵀ · (m-2,m) = (m,m) 的对称矩阵
        Z = W + lam * (D.T @ D)
        # 求解稀疏线性系统：Z·z = W·y，得到平滑信号z
        return spsolve(Z, w * y)

    @staticmethod
    def baseline_als(
        y,
        lam,
        p,
        niter=10,
        base_weights: np.ndarray | None = None,
        z0: np.ndarray | None = None,
    ):
        """
        非对称最小二乘平滑（Asymmetric Least Squares Smoothing, AsLS/AsLSS）。
        
        核心理论与创新：
        ----------------
        标准Whittaker平滑对所有点给予相同权重，导致基线可能会被信号峰值"拉起"。
        AsLS通过引入非对称权重w(z)来解决这个问题：
        
        基本思想：
          - 当数据点y在基线上方时（峰值）：给予较小权重p（p < 0.5）
          - 当数据点y在基线下方时（真实基线）：给予较大权重1-p（1-p > 0.5）
        
        这样基线会自然地沿着信号的"谷底"走，而不会被峰值影响，因为峰值点的残差
        对目标函数的贡献被大大降低（权重变小），而基线点的残差更受重视（权重变大）。
        
        迭代算法（Iterative Algorithm）：
        --------------------------------
        该算法是一个EM风格的迭代过程，交替优化权重和基线估计：
        
        第k次迭代求解如下优化问题：
        
            minimize: S = Σᵢ wᵢ⁽ᵏ⁾(yᵢ - zᵢ)² + λ·(D·z)ᵀ(D·z)
        
        其中非对称权重定义为：
        
            wᵢ⁽ᵏ⁾ = base_wᵢ · [p·𝟙(yᵢ > zᵢ⁽ᵏ˻¹⁾) + (1-p)·𝟙(yᵢ < zᵢ⁽ᵏ˻¹⁾)]
                  = base_wᵢ · [p  if yᵢ > zᵢ  else  1-p]
        
        其中：
            - base_w：额外的基础权重（用于强调/压制某些特征窗口）
            - 𝟙(·)：指示函数，当条件为真时=1，否则=0
            - p：非对称参数，以确定峰值权重
            - 1-p：基线权重
        
        权重参数的物理意义：
            - p=0.001：峰值权重0.001，基线权重0.999→基线贴最小值
            - p=0.01：峰值权重0.01，基线权重0.99→平衡
            - p=0.1：峰值权重0.1，基线权重0.9→允许基线有更多上升空间
            - p=0.5：等权重（对称）→退化为标准Whittaker平滑
        
        收敛性：
            初始的z⁽⁰⁾通过加权Whittaker平滑估计。随着迭代进行：
            - 权重根据新的z⁽ᵏ⁾动态调整
            - 基线逐渐收敛到一个自洽解
            - 通常10-20次迭代可达收敛
            - 更多次迭代可确保收敛但增加计算量（O(m·niter)）
        
        参数：
        ------
        y: np.ndarray
            原始信号，形状为(m,)
        lam: float
            Whittaker平滑参数λ，通常[1e2, 1e9]
            与whittaker_smooth()中的意义相同
        p: float
            非对称加权参数，范围[0.001, 0.1]
            - p=0.001：峰值权重越小，基线倾向于贴近数据最小值（激进）
            - p=0.01-0.05：推荐范围（平衡）
            - p=0.1：峰值权重增加，基线允许更多上升（保守）
            - p=0.5：等权重（对称），等价于whittaker_smooth
        niter: int
            迭代次数，默认10次
            - 推荐10-20次，足以收敛
            - 更多次迭代可减少权重波动，但计算成本↑
        base_weights: np.ndarray | None
            用户规定的基础权重向量，形状为(m,)
            用于进一步控制特定点或区域的重要性：
            - base_w=1.0：正常权重（完全参与基线拟合）
            - base_w=0.5：降权50%
            - base_w=0.05：大幅降权（如在已知特征位置）
            - base_w=0.001：基本忽略此点
            典型应用：在3.376, 3.438 μm附近使用0.05来保护特征
            默认None时使用base_w=1.0（所有点等权）
        z0: np.ndarray | None
            初始基线估计，形状为(m,)
            若为None，则首先用加权Whittaker平滑初始化（推荐）
            若为np.ndarray，则直接用作初始值（高级用法）
            
        返回：
        ------
        np.ndarray
            基线估计z⁽ⁿⁱᵗᵉʳ⁾，形状为(m,)
        
        数值稳定性说明：
        ----------------
        - base_weights被自动裁剪到[1e-8, inf)以避免完全零权重导致的奇异性
        - 每次迭代重新计算非对称权重，防止权重过早收敛
        - 稀疏矩阵求解器自动选择高效算法（三角因化或共轭梯度法）
        
        应用示例：
        ---------
        # 例1：基本用法
        y = load_spectrum(...)  # 三个吸收峰
        z = baseline_als(y, lam=1e3, p=0.01, niter=20)
        
        # 例2：保护已知特征
        base_w = np.ones_like(y)
        base_w[3000:3100] = 0.05  # 降权第3000-3100个点
        z = baseline_als(y, lam=1e3, p=0.01, niter=20, base_weights=base_w)
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        m = len(y)
        D = BaselineRemoval._diff_matrix(m, d=2)

        # 处理基础权重
        if base_weights is None:
            w_base = np.ones(m, dtype=float)
        else:
            w_base = np.asarray(base_weights, dtype=float).reshape(-1)
            if len(w_base) != m:
                raise ValueError('base_weights must have the same length as y')
            # 数值稳定性：防止权重为0
            w_base = np.clip(w_base, 1e-8, None)

        # 初始化基线z（第0次迭代）
        if z0 is None:
            # 首次调用Whittaker平滑，使用base_weights初始化
            z = BaselineRemoval.whittaker_smooth(y, lam=lam, weights=w_base)
        else:
            z = np.asarray(z0, dtype=float).reshape(-1)
            if len(z) != m:
                raise ValueError('z0 must have the same length as y')

        # 初始化非对称权重：根据y与z的相对位置分配权重
        # y > z（峰值）：权重=p；y <= z（基线）：权重=1-p
        w = w_base * (p * (y > z) + (1 - p) * (y < z))

        # 迭代求解（k=1, 2, ..., niter）
        for iteration in range(int(niter)):
            # 构造带非对称权重的Whittaker平滑系统
            W = sparse.spdiags(w, 0, m, m)
            Z = W + lam * (D.T @ D)
            # 求解：z⁽ᵏ⁾ = (W⁽ᵏ˻¹⁾ + λ·DᵀD)⁻¹ · W⁽ᵏ˻¹⁾ · y
            z = spsolve(Z, w * y)
            # 根据新的基线z⁽ᵏ⁾更新非对称权重
            w = w_base * (p * (y > z) + (1 - p) * (y < z))

        return z

    def baseline_removing(self, data):
        """
        批量基线去除（对多列数据应用基线去除）。
        
        沿着轴0应用baseline_als方法到data的每一列。
        适用于处理多通道或多条谱线的批量数据。
        
        参数：
        ------
        data: np.ndarray
            输入数据矩阵，形状为(n_samples, n_lines)或(n_samples,)
        
        返回：
        ------
        np.ndarray
            基线去除后的数据，形状同input
        """
        return np.apply_along_axis(lambda x: self.baseline_als(x, self.lam, self.p), 0, data)


if __name__ == '__main__':
    # ============================================================================
    #                         用户配置部分 (User Configuration)
    # ============================================================================
    # 说明：仅需修改以下几行参数，其余代码自动处理数据流程
    
    # 输入文件名称（两列格式：波数或波长 | 反射率/透射率 | 吸光度等）
    INPUT_FILENAME = 'try2D40.txt'  # 修改此行切换输入文件（如 'try3D40.txt' ）
    # 输出文件前缀自动从INPUT_FILENAME提取（e.g., 'try2D40.txt' -> 'try2D40'）
    OUTPUT_PREFIX = INPUT_FILENAME.rsplit('.', 1)[0] if '.' in INPUT_FILENAME else INPUT_FILENAME
    INPUT_FILE = INPUT_FILENAME
    # ============================================================================

    # 目标提取特征的波长位置（微米, μm）
    # 例如：3.376 μm 和 3.438 μm 对应红外吸收峰
    TARGET_WAVELENGTHS_UM = (3.376, 3.438)

    # 目标特征窗口设置
    TARGET_WINDOW_UM = 0.01           # 特征附近的窗口宽度（两侧各+/-0.01μm）
    SEARCH_WINDOW_UM = 0.01           # 提取局部极值的搜索范围
    TARGET_WINDOW_WEIGHT = 0.05       # 目标窗口的相对权重（防止基线被特征吸引）
                                      # 1.0: 正常权重（参与基线拟合）
                                      # 0.05: 大幅降低权重（保护已知特征）

    # AsLSS/Whittaker 平滑参数（可根据频谱特性调整）
    LAM = 1e2      # 平滑参数: 1e2(粗糙基线) ... 1e9(光滑基线)
                   # 建议范围：1e2-1e6 用于红外反射率光谱
                   # 更小值→基线跟踪噪声；更大值→基线过平滑
                   # 推荐做参数网格搜索验证：[1e2, 1e3, 1e4, 1e5, 1e6]
    P = 0.1        # 非对称加权参数: 0.001(贴最小值) ... 0.1(允许更多偏差)
                   # 推荐值：0.001-0.1，通常0.01-0.1效果好
                   # 较小值(0.001)：基线贴近数据最小值，保留更多特征
                   # 中等值(0.01-0.05)：平衡基线平滑与特征保护
                   # 较大值(0.1)：允许基线更高，可能丢失微弱特征
    NITER = 20     # 迭代次数：通常10-20次收敛，更多次确保收敛但增加计算量
                   # 推荐10-20，足以达到收敛

    # ============================================================================
    #                        辅助函数定义
    # ============================================================================

    def load_spectrum_txt(path: str):
        """
        从文本文件加载光谱数据。
        
        支持格式：
        -----------
        1. 单列格式：
           数值1
           数值2
           ...
           → 自动生成x轴坐标(0,1,2,...)，y值为给定数据
           
        2. 两列或多列格式（推荐）：
           波长1  反射率1
           波长2  反射率2
           ...
           → 第一列为x轴(波长/波数)，第二列为y值(反射率/透射率/吸光度)
           → 自动移除NaN行并按x轴递增排序
           → 支持注释行（以#开头）
        
        参数：
        ------
        path: str
            输入文件路径
        
        返回：
        ------
        (x_, y_): Tuple[np.ndarray, np.ndarray]
            x_: 横坐标（波长或波数），形状(m,)，递增排序
            y_: 纵坐标（强度），形状(m,)，与x_对应排序
        """
        arr = np.genfromtxt(path, dtype=float, comments='#', invalid_raise=False)
        if arr.ndim == 1:
            # 单列数据
            y_ = np.asarray(arr, dtype=float)
            x_ = np.arange(len(y_), dtype=float)
        else:
            # 多列数据：取前两列，移除含NaN的行
            arr = np.asarray(arr, dtype=float)
            arr = arr[~np.isnan(arr).any(axis=1)]
            x_ = arr[:, 0]
            y_ = arr[:, 1]
        # 按x轴排序（从小到大）
        order = np.argsort(x_)
        return x_[order], y_[order]

    def estimate_baseline_aslss(
        y_: np.ndarray,
        lam: float,
        p: float,
        niter: int,
        base_weights: np.ndarray,
        polarity: str,
    ):
        """
        使用AsLSS估计基线（支持多种极性模式）。
        
        该函数是对BaselineRemoval.baseline_als的高级包装，
        支持自动检测信号极性（峰值向上还是向下）。
        
        参数：
        ------
        y_: np.ndarray
            原始信号，形状(m,)
        lam: float
            平滑参数
        p: float
            非对称加权参数
        niter: int
            迭代次数
        base_weights: np.ndarray
            基础权重向量
        polarity: str
            极性模式，取值：
            - 'positive': 峰值向上（标准IR吸收峰）
              基线从下方跟踪峰值，使用标准AsLS
            - 'negative': 峰值向下（反射率下降的吸收）
              反转信号→应用AsLS→反转回→基线从上方跟踪谷底
            - 'symmetric': 对称（等权重）
              使用p=0.5的Whittaker平滑，无非对称性
        
        返回：
        ------
        (z, used_polarity): Tuple[np.ndarray, str]
            z: 估计的基线
            used_polarity: 实际使用的极性模式
        """
        if polarity == 'symmetric':
            z = BaselineRemoval.whittaker_smooth(y_, lam=lam, weights=base_weights)
            return z, polarity
        if polarity == 'positive':
            z = BaselineRemoval.baseline_als(y_, lam, p, niter=niter, base_weights=base_weights)
            return z, polarity
        if polarity == 'negative':
            # 负极性：反转数据→拟合→反转回
            z = -BaselineRemoval.baseline_als(-y_, lam, p, niter=niter, base_weights=base_weights)
            return z, polarity
        raise ValueError("polarity must be one of: 'positive', 'negative', 'symmetric'")

    def interpolate_at(x_: np.ndarray, y_: np.ndarray, x0: float) -> float:
        """
        在指定波长处线性插值获取信号值。
        
        参数：
        ------
        x_: np.ndarray
            横坐标数组（递增）
        y_: np.ndarray
            纵坐标数组
        x0: float
            插值位置的横坐标
        
        返回：
        ------
        float
            插值结果
        """
        return float(np.interp(x0, x_, y_))

    def extract_local_metrics(x_: np.ndarray, y_: np.ndarray, x0: float, window_um: float):
        """
        提取目标波长附近的局部指标。
        
        在target_um±window_um范围内计算：
        1. 极值位置（峰值或谷底）
        2. 极值处的值
        3. 峰-谷差（peak-to-peak）：max - min
        4. 窗口内的积分面积（用梯形法则）
        
        参数：
        ------
        x_: np.ndarray
            横坐标（递增排序）
        y_: np.ndarray
            纵坐标
        x0: float
            目标波长
        window_um: float
            搜索窗口半宽度（±window_um）
        
        返回：
        ------
        (x_extremum, y_extremum, y_min, y_max): Tuple[float, float, float, float]
            x_extremum: 极值处的横坐标
            y_extremum: 极值处的纵坐标（绝对值最大的极值）
            y_min: 窗口内y的最小值
            y_max: 窗口内y的最大值
        
        说明：
        ------
        - 若指定窗口内无数据点，自动选择最近的点
        - 极值选择：选择绝对值最大的（可能是最小值或最大值）
        - 用途：评估特征的强度和形状
        """
        # 获得目标波长±window范围内的数据
        mask = (x_ >= x0 - window_um) & (x_ <= x0 + window_um)
        if not np.any(mask):
            # 若窗口内无点，取最近点
            idx = int(np.argmin(np.abs(x_ - x0)))
            yy = float(y_[idx])
            return float(x_[idx]), yy, yy, yy
        xx = x_[mask]
        yy = y_[mask]
        y_min = float(np.min(yy))
        y_max = float(np.max(yy))
        # 选择极值：取绝对值最大的（可能是最大值或最小值）
        j = int(np.argmin(yy)) if abs(y_min) >= abs(y_max) else int(np.argmax(yy))
        return float(xx[j]), float(yy[j]), y_min, y_max

    # ============================================================================
    #                            主程序执行流程
    # ============================================================================

    # 第1步：加载光谱数据
    print(f"[1/5] 加载光谱数据: {INPUT_FILE}")
    x, y = load_spectrum_txt(INPUT_FILE)
    print(f"      数据点数：{len(x)}, 波长范围：{x[0]:.4f}-{x[-1]:.4f} μm")

    # 第2步：构造基础权重向量（用于强调/压制已知特征窗口）
    print(f"[2/5] 构造权重向量...")
    # 初始化：所有点权重=1（等权）
    base_w = np.ones_like(y, dtype=float)
    # 在目标特征附近（如3.376, 3.438 μm）降低权重
    # 这防止基线被测量噪声或强特征"拉动"，保护了特征的完整性
    for t in TARGET_WAVELENGTHS_UM:
        base_w[np.abs(x - t) <= TARGET_WINDOW_UM] = TARGET_WINDOW_WEIGHT
    print(f"      目标窗口：{TARGET_WAVELENGTHS_UM}")
    print(f"      窗口宽度：±{TARGET_WINDOW_UM} μm, 相对权重：{TARGET_WINDOW_WEIGHT}")

    # 第3步：自动检测信号极性（峰值向上或向下）
    print(f"[3/5] 自动检测信号极性...")
    # 使用对称Whittaker平滑（无非对称性）作为参考基线
    z_sym = BaselineRemoval.whittaker_smooth(y, lam=LAM, weights=base_w)
    # 计算目标窗口内原始数据与参考基线的平均偏差
    win_mask = np.zeros_like(y, dtype=bool)
    for t in TARGET_WAVELENGTHS_UM:
        win_mask |= (np.abs(x - t) <= TARGET_WINDOW_UM)
    mean_dev = float(np.mean(y[win_mask] - z_sym[win_mask])) if np.any(win_mask) else 0.0
    # 根据偏差符号判断：负偏差表示吸收峰（反射率下降）=负极性；正偏差=正极性
    polarity = 'negative' if mean_dev < 0 else 'positive'
    print(f"      目标窗口平均偏差：{mean_dev:.6g}")
    print(f"      检测极性：{polarity} ({'反射率下降' if polarity == 'negative' else '反射率上升'})")

    # 第4步：使用AsLSS估计基线
    print(f"[4/5] 基线拟合 (AsLSS)...")
    print(f"      参数：lam={LAM:.1e}, p={P:.4f}, niter={NITER}")
    baseline, used_polarity = estimate_baseline_aslss(
        y_=y,
        lam=LAM,
        p=P,
        niter=NITER,
        base_weights=base_w,
        polarity=polarity,
    )

    # 第5步：计算校正后的信号（两种定义）
    print(f"[5/5] 信号校正...")
    # 定义1：相减法（Subtraction definition）
    #   corrected_sub = raw - baseline
    #   物理意义：信号增量（绝对变化），对绝对基线值敏感
    #   应用：传统的吸光度计算，与标准相符
    corrected_sub = y - baseline
    # 定义2：相除法（Division/relative definition）
    #   corrected_div = raw / baseline - 1
    #   物理意义：相对变化，对基线的动态变化更鲁棒
    #   应用：反射率分析，消除刀片反射等系统偏移
    corrected_div = y / np.clip(baseline, 1e-12, None) - 1.0

    # ============================================================================
    #                          输出文件生成
    # ============================================================================
    print(f"\n[输出] 生成基线文件: {OUTPUT_PREFIX}_baseline.txt")
    # 保存基线估计（两列：波长, 基线值）
    np.savetxt(f'{OUTPUT_PREFIX}_baseline.txt', np.column_stack([x, baseline]), fmt='%.10g', delimiter='\t')
    
    print(f"[输出] 生成校正文件: {OUTPUT_PREFIX}_corrected_sub.txt")
    # 保存相减法校正结果（两列：波长, 校正值）
    np.savetxt(f'{OUTPUT_PREFIX}_corrected_sub.txt', np.column_stack([x, corrected_sub]), fmt='%.10g', delimiter='\t')
    
    print(f"[输出] 生成校正文件: {OUTPUT_PREFIX}_corrected_div.txt")
    # 保存相除法校正结果（两列：波长, 相对变化）
    np.savetxt(f'{OUTPUT_PREFIX}_corrected_div.txt', np.column_stack([x, corrected_div]), fmt='%.10g', delimiter='\t')

    # 向后兼容：历史命名 "corrected" = "corrected_sub"
    print(f"[输出] 生成校正文件: {OUTPUT_PREFIX}_corrected.txt (兼容名)")
    np.savetxt(f'{OUTPUT_PREFIX}_corrected.txt', np.column_stack([x, corrected_sub]), fmt='%.10g', delimiter='\t')

    # ============================================================================
    #                  特征提取与指标计算
    # ============================================================================
    print(f"\n[特征提取] 在目标波长处提取特征...")
    lines = []
    # 生成输出文件头（包含元数据和参数信息）
    header = (
        '# 基线校正与特征提取摘要\n'
        f'# 输入文件: {INPUT_FILE}\n'
        f'# 方法: AsLSS (Eilers族; 加权Whittaker平滑 + 非对称权重迭代)\n'
        f'# 参数: lam={LAM:g}, p={P:g}, niter={NITER}\n'
        f'# 目标窗口: ±{TARGET_WINDOW_UM:g}μm, 相对权重={TARGET_WINDOW_WEIGHT:g}\n'
        f'# 极性(自动): {used_polarity} (窗口平均偏差={mean_dev:.6g})\n'
        '# 校正定义:\n'
        '#   corrected_sub = raw - baseline         (相减法，绝对变化)\n'
        '#   corrected_div = raw / baseline - 1    (相除法，相对变化)\n'
        '# 列说明:\n'
        '# target_um\traw\tbaseline\tcorrected_sub\tcorrected_div\t'
        'local_extremum_um\tlocal_extremum_sub\tlocal_p2p_sub\tlocal_p2p_div\tarea_sub\tarea_div\n'
    )
    lines.append(header)

    # 遍历每个目标波长，计算和输出指标
    for t in TARGET_WAVELENGTHS_UM:
        # 基本指标：在目标波长处的值（线性插值）
        raw_i = interpolate_at(x, y, t)
        base_i = interpolate_at(x, baseline, t)
        sub_i = raw_i - base_i
        div_i = raw_i / max(base_i, 1e-12) - 1.0

        # 校正后信号的局部指标：
        # 1. 搜索窗口内的极值位置和值
        ex_x, ex_sub, sub_min, sub_max = extract_local_metrics(x, corrected_sub, t, SEARCH_WINDOW_UM)
        _, ex_div, div_min, div_max = extract_local_metrics(x, corrected_div, t, SEARCH_WINDOW_UM)

        # 峰-谷差（Peak-to-peak）：最大值与最小值的差，表示特征的幅度
        p2p_sub = sub_max - sub_min
        p2p_div = div_max - div_min

        # 积分面积（Integrated area）：使用梯形法则在窗口内积分
        # 代表特征的"强度"或"信号总量"，更稳健于峰值位置的偏移
        area_mask = (x >= t - SEARCH_WINDOW_UM) & (x <= t + SEARCH_WINDOW_UM)
        area_sub = float(np.trapezoid(corrected_sub[area_mask], x[area_mask])) if np.any(area_mask) else float('nan')
        area_div = float(np.trapezoid(corrected_div[area_mask], x[area_mask])) if np.any(area_mask) else float('nan')

        # 输出该目标波长的所有指标
        lines.append(
            f'{t:.6f}\t{raw_i:.10g}\t{base_i:.10g}\t{sub_i:.10g}\t{div_i:.10g}\t'
            f'{ex_x:.10g}\t{ex_sub:.10g}\t{p2p_sub:.10g}\t{p2p_div:.10g}\t{area_sub:.10g}\t{area_div:.10g}\n'
        )
        print(f"      {t:.4f} μm: raw={raw_i:.6f}, baseline={base_i:.6f}, sub={sub_i:.6f}, div={div_i:.6f}")

    # 保存提取的特征指标到文件
    print(f"\n[输出] 生成特征文件: {OUTPUT_PREFIX}_extracted_signals.txt")
    with open(f'{OUTPUT_PREFIX}_extracted_signals.txt', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # ============================================================================
    #                        可视化与结果展示
    # ============================================================================
    print(f"\n[可视化] 绘制原始数据、基线与校正结果...")
    fig, axes = plt.subplots(2, 1, sharex=True)
    
    # 上图：原始数据 vs 拟合基线
    axes[0].plot(x, y, label='原始数据 (raw)', linewidth=1.5)
    axes[0].plot(x, baseline, label='估计基线 (baseline)', linewidth=1.5, linestyle='--')
    axes[0].set_ylabel('反射率')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'基线去除: {INPUT_FILE} (方法: AsLSS, λ={LAM:.1e}, p={P:.4f})')

    # 下图：校正后的信号 + 目标波长标记
    axes[1].plot(x, corrected_sub, label='校正后信号 (raw-baseline)', linewidth=1.5)
    # 在目标波长位置标记竖线
    for idx, t in enumerate(TARGET_WAVELENGTHS_UM):
        axes[1].axvline(t, color='red', linestyle='--', alpha=0.7, 
                       label=f'目标: {t:.3f} μm' if idx == 0 else '')
    axes[1].set_xlabel('波长 (μm)')
    axes[1].set_ylabel('反射率差值 (相减法)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n[完成] 基线去除和特征提取完成！")
