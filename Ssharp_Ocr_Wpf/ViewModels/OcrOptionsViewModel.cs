using System;

namespace Ssharp_Ocr_Wpf.ViewModels
{
    /// <summary>
    /// OCR 配置视图模型。
    /// </summary>
    public sealed class OcrOptionsViewModel : ViewModelBase
    {
        /// <summary>
        /// 定位模型输入尺寸。
        /// </summary>
        public int DetImgSize { get; set; } = 640;
        /// <summary>
        /// 定位置信度阈值。
        /// </summary>
        public double DetConf { get; set; } = 0.25;
        /// <summary>
        /// 定位 NMS IoU 阈值。
        /// </summary>
        public double DetIou { get; set; } = 0.7;
        /// <summary>
        /// 定位最大检测数。
        /// </summary>
        public int DetMaxDet { get; set; } = 3;
        /// <summary>
        /// 定位原始最大检测数。
        /// </summary>
        public int DetRawMaxDet { get; set; } = 30;
        /// <summary>
        /// 定位类别数。
        /// </summary>
        public int DetClassCount { get; set; } = 1;
        /// <summary>
        /// 定位去重 IoU 阈值。
        /// </summary>
        public double DetDistinctIou { get; set; } = 0.7;
        /// <summary>
        /// 定位去重中心距离比例。
        /// </summary>
        public double DetDistinctMinCenterRatio { get; set; } = 0.12;
        /// <summary>
        /// 定位输出格式。
        /// </summary>
        public OutputFormat DetOutputFormat { get; set; } = OutputFormat.Auto;

        /// <summary>
        /// 识别模型输入尺寸。
        /// </summary>
        public int RecImgSize { get; set; } = 320;
        /// <summary>
        /// 识别自动尺寸开关。
        /// </summary>
        public bool RecAutoImgSize { get; set; } = true;
        /// <summary>
        /// 识别输入最小尺寸。
        /// </summary>
        public int RecImgSizeMin { get; set; } = 60;
        /// <summary>
        /// 识别输入最大尺寸。
        /// </summary>
        public int RecImgSizeMax { get; set; } = 640;
        /// <summary>
        /// 识别置信度阈值。
        /// </summary>
        public double RecConf { get; set; } = 0.20;// 0.25;
        /// <summary>
        /// 识别 NMS IoU 阈值。
        /// </summary>
        public double RecIou { get; set; } = 0.7;
        /// <summary>
        /// 识别 TopN 数量。
        /// </summary>
        public int RecTopN { get; set; } = 14;
        /// <summary>
        /// 识别最小框边长。
        /// </summary>
        public int RecMinBox { get; set; } = 2;//4
        /// <summary>
        /// 识别最小分数。
        /// </summary>
        public double RecMinScore { get; set; } = 0.35;//0.5;
        /// <summary>
        /// 识别翻转增强开关。
        /// </summary>
        public bool RecFlipEnable { get; set; } = false;
        /// <summary>
        /// 识别翻转最小分数。
        /// </summary>
        public double RecFlipMinScore { get; set; } = 0.5;
        /// <summary>
        /// 识别最大检测数。
        /// </summary>
        public int RecMaxDet { get; set; } = 14;
        /// <summary>
        /// 识别类别数。
        /// </summary>
        public int RecClassCount { get; set; } = 13;
        /// <summary>
        /// 识别输出格式。
        /// </summary>
        public OutputFormat RecOutputFormat { get; set; } = OutputFormat.NmsXyxy;

        /// <summary>
        /// 识别输出模式。
        /// </summary>
        public RecOutputMode RecMode { get; set; } = RecOutputMode.YoloLike;
        /// <summary>
        /// CTC Blank 索引。
        /// </summary>
        public int RecCtcBlankIndex { get; set; } = -1;
        /// <summary>
        /// CTC 类别数。
        /// </summary>
        public int RecCtcClassCount { get; set; } = 0;
        /// <summary>
        /// 识别是否使用 objectness。
        /// </summary>
        public bool RecUseObjectness { get; set; } = true;
        /// <summary>
        /// 识别是否使用 Sigmoid。
        /// </summary>
        public bool RecUseSigmoid { get; set; } = false;
        /// <summary>
        /// 识别通道顺序。
        /// </summary>
        public ImageChannelOrder RecChannelOrder { get; set; } = ImageChannelOrder.Rgb;

        /// <summary>
        /// ROI Padding 比例。
        /// </summary>
        public double RoiPadRatio { get; set; } = 0.06;//0;
        /// <summary>
        /// ROI Padding 像素。
        /// </summary>
        public int RoiPadPx { get; set; } = 4;//0;
        /// <summary>
        /// 识别行分组阈值。
        /// </summary>
        public double RecRowThresh { get; set; } = 0.6;
        /// <summary>
        /// ROI 期望长度字符串。
        /// </summary>
        public string RoiExpectedLengthsText { get; set; } = "8,14,2";

        /// <summary>
        /// 是否启用 GPU。
        /// </summary>
        public bool UseGpu { get; set; } = false;
        /// <summary>
        /// 线程数。
        /// </summary>
        public int NumThreads { get; set; } = 0;
        /// <summary>
        /// 排序方式。
        /// </summary>
        public SortMode SortBy { get; set; } = SortMode.X;

        /// <summary>
        /// 词表路径。
        /// </summary>
        public string VocabPath { get; set; } =AppDomain.CurrentDomain.BaseDirectory + "vocab_rec.txt";

        /// <summary>
        /// 生成推理配置。
        /// </summary>
        /// <param name="logHandler">日志回调。</param>
        /// <returns>推理配置。</returns>
        public OcrOptions ToOptions(Action<string> logHandler)
        {
            return new OcrOptions
            {
                DetImgSize = DetImgSize,
                DetConf = DetConf,
                DetIou = DetIou,
                DetMaxDet = DetMaxDet,
                DetRawMaxDet = DetRawMaxDet,
                DetClassCount = DetClassCount,
                DetDistinctIou = DetDistinctIou,
                DetDistinctMinCenterRatio = DetDistinctMinCenterRatio,
                DetOutputFormat = DetOutputFormat,
                RecImgSize = RecImgSize,
                RecAutoImgSize = RecAutoImgSize,
                RecImgSizeMin = RecImgSizeMin,
                RecImgSizeMax = RecImgSizeMax,
                RecConf = RecConf,
                RecIou = RecIou,
                RecTopN = RecTopN,
                RecMinBox = RecMinBox,
                RecMinScore = RecMinScore,
                RecFlipEnable = RecFlipEnable,
                RecFlipMinScore = RecFlipMinScore,
                RecMaxDet = RecMaxDet,
                RecClassCount = RecClassCount,
                RecOutputFormat = RecOutputFormat,
                RecMode = RecMode,
                RecCtcBlankIndex = RecCtcBlankIndex,
                RecCtcClassCount = RecCtcClassCount,
                RecUseObjectness = RecUseObjectness,
                RecUseSigmoid = RecUseSigmoid,
                RecChannelOrder = RecChannelOrder,
                RoiPadRatio = RoiPadRatio,
                RoiPadPx = RoiPadPx,
                RecRowThresh = RecRowThresh,
                RoiExpectedLengths = OcrOptions.ParseRoiExpectedLengths(RoiExpectedLengthsText),
                UseGpu = UseGpu,
                NumThreads = NumThreads,
                SortBy = SortBy,
                VocabPath = VocabPath,
                Log = logHandler
            };
        }
    }
}
