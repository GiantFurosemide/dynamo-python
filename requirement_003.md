继续实现 response_002.md的限制内容，上一轮的requirement为 requirement_002.md。将计划更新到 requirement_refined_003.md

本轮进度输出到response_003.md中

# 输入

* mrc map实例：{{sythetic_map}} = "synthetic_data/emd_32820_bin4.mrc"
* synthetic data path：{{synthetic_path}} = "synthetic_data"

# 计划

1. 完善该requirement的计划和其他细节，将计划更新到 requirement_refined_003.md
2. 修改之前的相关代码，所有的关键变量，默认变量，要全部显性的整理在 yaml文件中。
3. 继续实现之前response和requirement未完成的功能
4. 生成sythetic data：生辰一个XYZ 尺寸为 2000 2000 800的tomogram，pixel size 是和{{sythetic_map}}一样， missing wedge +48到-48度。 {{sythetic_map}} 作为模版，随机取向 随机位置（tomogram内）采样生成1000个 取向（欧拉角描述）+坐标，整合到tomogram中 , 保存在 {{synthetic_path}} 下的 out_tomograms下。并生成对应tbl vll和 star 文件。随后用之前完成的crop功能将，subtomogram提取并保存在{{synthetic_path}} 下的 out_subtomograms下。在生成和subtomogram一样尺寸的噪声subtomogram，和之前生成的subtomogram一同保存在{{synthetic_path}} 下的 out_tomograms4classification下，并生成对应的tbl vll和 star 文件。之后的测试crop，reconstruction以及align 或multi classification。就用这个数据集测试。
5. 测试并完善代码
6. update相关文档 以及环境文件
7. 更新进度和遇到的问题到response_003.md

# 终止条件

需完成以下所有任务：

* 完成关于yaml的修改
* 完成生成sythetic data的生成
* 完成基于numpy 和 pytorch的核心代码实现和最终的实例测试。（本机现在没有GPU，但需要你实现，不过针对gpu的测试可先不运行）
* 完成每个功能的一键启动（需实现crop，reconstruction， alignment， classid）
