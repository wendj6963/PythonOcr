using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Controls;
using System.Xml.Linq;
using Size = OpenCvSharp.Size;

namespace Ssharp_Ocr_Wpf
{

    public sealed class OcrRunner : IDisposable
    {
        private readonly InferenceSession _detSession;
        private readonly InferenceSession _recSession;
        private readonly OcrOptions _options;
        private readonly Dictionary<int, string> _vocab;
        private readonly int? _detInputSize;
        private readonly int? _recInputSize;

        // 构造并加载加密模型，初始化推理会话与输入尺寸
        public OcrRunner(string detEncPath, string recEncPath, string passphrase, OcrOptions options)
        {
            _options = options ?? new OcrOptions();
            _vocab = LoadVocab(_options.VocabPath);

            var detEncBytes = File.ReadAllBytes(detEncPath);
            var recEncBytes = File.ReadAllBytes(recEncPath);

            LogModelSha256("det", detEncPath, detEncBytes, _options.Log);
            LogModelSha256("rec", recEncPath, recEncBytes, _options.Log);

            var detBytes = DecryptModel(detEncBytes, passphrase);
            var recBytes = DecryptModel(recEncBytes, passphrase);

            var detOptions = CreateSessionOptions(_options);
            var recOptions = CreateSessionOptions(_options);

            _detSession = new InferenceSession(detBytes, detOptions);
            _recSession = new InferenceSession(recBytes, recOptions);

            _detInputSize = GetFixedSquareInputSize(_detSession);
            _recInputSize = GetFixedSquareInputSize(_recSession);

            // 对齐识别类别数（优先使用 vocab，其次使用模型输出维度推断）
            AlignRecClassCount(_recSession, _options, _vocab);

            LogDetOutputShape(_detSession, _options);
            LogRecOutputShape(_recSession, _options);
        }

        // 基于 vocab/模型输出维度对齐识别类别数
        private static void AlignRecClassCount(InferenceSession session, OcrOptions options, Dictionary<int, string> vocab)
        {
            var vocabCount = GetVocabClassCount(vocab);
            if (vocabCount > 0 && options.RecClassCount != vocabCount)
            {
                options.RecClassCount = vocabCount;
                options.Log?.Invoke($"rec 类别数已按 vocab 对齐: {vocabCount}");
                return;
            }

            if (options.RecClassCount > 0)
                return;

            var inferred = InferClassCountFromOutput(session, options.RecOutputFormat);
            if (inferred.HasValue && inferred.Value > 0)
            {
                options.RecClassCount = inferred.Value;
                options.Log?.Invoke($"rec 类别数已按模型输出推断: {inferred.Value}");
            }
        }

        private static int GetVocabClassCount(Dictionary<int, string> vocab)
        {
            if (vocab.Count == 0)
                return 0;
            return vocab.Keys.Max() + 1;
        }

        private static int? InferClassCountFromOutput(InferenceSession session, OutputFormat format)
        {
            var meta = session.OutputMetadata.Values.FirstOrDefault();
            if (meta?.Dimensions == null || meta.Dimensions.Length < 3)
                return null;

            var c = Math.Min(meta.Dimensions[1], meta.Dimensions[2]);
            if (c <= 0)
                return null;

            if (format == OutputFormat.NmsXyxy || format == OutputFormat.NmsXywha)
                return c >= 6 ? (int?)1 : null;

            // Auto：rec 常见 raw 格式为 5 + classCount
            if (c > 6)
                return c - 5;

            return 1;
        }

        // 打印 det 输出形状信息
        private static void LogDetOutputShape(InferenceSession session, OcrOptions options)
        {
            var meta = session.OutputMetadata.Values.FirstOrDefault();
            if (meta?.Dimensions == null)
                return;

            var dims = meta.Dimensions.ToArray();
            var channels = dims.Length >= 3 ? Math.Min(dims[1], dims[2]) : 0;
            var hasAngle = channels >= 7;
            var msg = $"det output shape: [{string.Join(",", dims)}], channels={channels}, 含角度={(hasAngle ? "True" : "False")}, format={options.DetOutputFormat}";
            Console.WriteLine(msg);
            options.Log?.Invoke(msg);
        }

        // 打印 rec 输出形状信息
        private static void LogRecOutputShape(InferenceSession session, OcrOptions options)
        {
            var meta = session.OutputMetadata.Values.FirstOrDefault();
            if (meta?.Dimensions == null)
                return;

            var dims = meta.Dimensions.ToArray();
            var channels = dims.Length >= 3 ? Math.Min(dims[1], dims[2]) : 0;
            var msg = $"rec output shape: [{string.Join(",", dims)}], channels={channels}";
            Console.WriteLine(msg);
            options.Log?.Invoke(msg);
        }

        // 过滤识别框并做排序
        private static List<DetBox> FilterRecBoxes(List<DetBox> boxes, SortMode sortBy, double minScore, int topN, int minBox, double rowThresh, double iouThresh)
        {
            var filtered = boxes
                .Where(b => b.Score >= minScore && b.Rect.Width >= minBox && b.Rect.Height >= minBox)
                .OrderByDescending(b => b.Score)
                .ToList();

            if (topN > 0)
                filtered = filtered.Take(topN).ToList();

            if (iouThresh > 0 && filtered.Count > 1)
            {
                var keep = NmsBoxes(filtered, iouThresh, topN > 0 ? topN : 0);
                filtered = keep.Select(i => filtered[i]).ToList();
            }

            return SortBoxes(filtered, sortBy, rowThresh);
        }

        private static List<DetPoly> FilterRecPolys(List<DetPoly> polys, SortMode sortBy, double minScore, int topN, int minBox, double rowThresh, double iouThresh)
        {
            var filtered = polys
                .Where(p => p.Conf >= minScore)
                .Select(p => new { Poly = p, Rect = QuadToAxisAligned(p.Quad) })
                .Where(p => p.Rect.Width >= minBox && p.Rect.Height >= minBox)
                .OrderByDescending(p => p.Poly.Conf)
                .Select(p => p.Poly)
                .ToList();

            if (topN > 0)
                filtered = filtered.Take(topN).ToList();

            if (iouThresh > 0 && filtered.Count > 1)
            {
                filtered = NmsPolys(filtered, iouThresh, topN > 0 ? topN : 0);
            }

            return SortPolys(filtered, sortBy, rowThresh);
        }


        // 将识别框转换为文本与置信度
        private (List<string> text, List<double> scores, List<CharBox> charBoxes) RecTextFromBoxes(List<DetBox> boxes)
        {
            var textParts = new List<string>();
            var scores = new List<double>();
            var charBoxes = new List<CharBox>();
            foreach (var b in boxes)
            {
                var text = _vocab.TryGetValue(b.Cls, out var v) ? v : b.Cls.ToString();
                textParts.Add(text);
                scores.Add(b.Score);
                charBoxes.Add(new CharBox(b.Rect, null, text, b.Score));
            }
            return (textParts, scores, charBoxes);
        }

        private (List<string> text, List<double> scores, List<CharBox> charBoxes) RecTextFromPolys(List<DetPoly> polys)
        {
            var textParts = new List<string>();
            var scores = new List<double>();
            var charBoxes = new List<CharBox>();
            foreach (var p in polys)
            {
                var text = _vocab.TryGetValue(p.Cls, out var v) ? v : p.Cls.ToString();
                textParts.Add(text);
                scores.Add(p.Conf);
                var rect = QuadToAxisAligned(p.Quad);
                charBoxes.Add(new CharBox(rect, p.Quad, text, p.Conf));
            }
            return (textParts, scores, charBoxes);
        }

        // 选择识别输入尺寸（自动或固定）
        private static int ChooseRecImgSize(Mat roi, OcrOptions options, int? fixedModelSize)
        {
            if (fixedModelSize.HasValue && fixedModelSize.Value > 0)
                return fixedModelSize.Value;
            if (!options.RecAutoImgSize)
                return options.RecImgSize;

            var side = Math.Max(roi.Rows, roi.Cols);
            side = Math.Max(options.RecImgSizeMin, Math.Min(options.RecImgSizeMax, side));
            return Math.Max(32, (int)Math.Round(side / 32.0) * 32);
        }

        // 兼容旧签名
        private static int ChooseRecImgSize(Mat roi, OcrOptions options)
        {
            return ChooseRecImgSize(roi, options, null);
        }

        // 从模型输入元数据中读取固定方形尺寸
        private static int? GetFixedSquareInputSize(InferenceSession session)
        {
            var meta = session.InputMetadata.Values.FirstOrDefault();
            if (meta == null || meta.Dimensions == null || meta.Dimensions.Length < 4)
                return null;

            var h = meta.Dimensions[meta.Dimensions.Length - 2];
            var w = meta.Dimensions[meta.Dimensions.Length - 1];
            if (h > 0 && w > 0 && h == w)
                return (int)h;
            return null;
        }

        // 对矩形框进行 NMS 去重
        private static List<int> NmsBoxes(List<DetBox> boxes, double iouThresh, int maxDet)
        {
            var order = boxes.Select((b, i) => new { b.Score, i })
                .OrderByDescending(x => x.Score)
                .Select(x => x.i)
                .ToList();

            var keep = new List<int>();
            while (order.Count > 0)
            {
                var idx = order[0];
                order.RemoveAt(0);
                keep.Add(idx);

                if (maxDet > 0 && keep.Count >= maxDet)
                    break;

                order = order.Where(i => IoU(boxes[idx].Rect, boxes[i].Rect) < iouThresh).ToList();
            }
            return keep;
        }

        // 按类别执行 NMS（class-aware），与 Ultralytics 默认一致
        private static List<int> NmsBoxesByClass(List<DetBox> boxes, double iouThresh, int maxDet)
        {
            if (boxes.Count == 0)
                return new List<int>();

            var kept = new List<int>();
            foreach (var group in boxes.Select((b, i) => new { b, i }).GroupBy(x => x.b.Cls))
            {
                var list = group.Select(x => x.i).ToList();
                var order = list
                    .OrderByDescending(i => boxes[i].Score)
                    .ToList();

                while (order.Count > 0)
                {
                    var idx = order[0];
                    order.RemoveAt(0);
                    kept.Add(idx);

                    if (maxDet > 0 && kept.Count >= maxDet)
                        return kept;

                    order = order.Where(i => IoU(boxes[idx].Rect, boxes[i].Rect) < iouThresh).ToList();
                }
            }

            return kept;
        }

        // 计算 IoU
        private static double IoU(Rect a, Rect b)
        {
            var inter = a & b;
            if (inter.Width <= 0 || inter.Height <= 0)
                return 0.0;
            var interArea = inter.Width * inter.Height;
            var union = a.Width * a.Height + b.Width * b.Height - interArea;
            return union <= 0 ? 0.0 : interArea / union;
        }

        // 多边形排序
        private static List<DetPoly> SortPolys(List<DetPoly> polys, SortMode sortBy, double rowThresh)
        {
            if (sortBy == SortMode.Y)
                return polys.OrderBy(p => p.Center.Y).ToList();

            if (sortBy != SortMode.Line)
                return polys.OrderBy(p => p.Center.X).ToList();

            var rows = new List<List<DetPoly>>();
            foreach (var p in polys.OrderBy(p => p.Center.Y))
            {
                var placed = false;
                foreach (var row in rows)
                {
                    var cy = row[0].Center.Y;
                    if (Math.Abs(p.Center.Y - cy) <= rowThresh * row[0].Height)
                    {
                        row.Add(p);
                        placed = true;
                        break;
                    }
                }
                if (!placed)
                    rows.Add(new List<DetPoly> { p });
            }

            var outList = new List<DetPoly>();
            foreach (var row in rows)
            {
                outList.AddRange(row.OrderBy(p => p.Center.X));
            }
            return outList;
        }

        // 矩形排序
        private static List<DetBox> SortBoxes(List<DetBox> boxes, SortMode sortBy, double rowThresh)
        {
            if (sortBy == SortMode.Y)
                return boxes.OrderBy(b => b.Center.Y).ToList();

            if (sortBy != SortMode.Line)
                return boxes.OrderBy(b => b.Center.X).ToList();

            var rows = new List<List<DetBox>>();
            foreach (var b in boxes.OrderBy(b => b.Center.Y))
            {
                var placed = false;
                foreach (var row in rows)
                {
                    var cy = row[0].Center.Y;
                    if (Math.Abs(b.Center.Y - cy) <= rowThresh * row[0].Height)
                    {
                        row.Add(b);
                        placed = true;
                        break;
                    }
                }
                if (!placed)
                    rows.Add(new List<DetBox> { b });
            }

            var outList = new List<DetBox>();
            foreach (var row in rows)
            {
                outList.AddRange(row.OrderBy(b => b.Center.X));
            }
            return outList;
        }

        // 解密 AES-256-CBC + HMAC 加密模型
        private static byte[] DecryptModel(byte[] data, string passphrase)
        {
            if (data.Length < 4 + 16 + 16 + 32)
                throw new InvalidOperationException("Encrypted data too short");
            if (data[0] != (byte)'O' || data[1] != (byte)'C' || data[2] != (byte)'B' || data[3] != (byte)'C')
                throw new InvalidOperationException("Invalid encrypted header");

            var salt = data.Skip(4).Take(16).ToArray();
            var iv = data.Skip(20).Take(16).ToArray();
            var tag = data.Skip(36).Take(32).ToArray();
            var ciphertext = data.Skip(68).ToArray();

            using (var kdf = new Rfc2898DeriveBytes(passphrase, salt, 200_000, HashAlgorithmName.SHA256))
            {
                var key = kdf.GetBytes(64);
                var encKey = key.Take(32).ToArray();
                var macKey = key.Skip(32).ToArray();

                using (var hmac = new HMACSHA256(macKey))
                {
                    var calc = hmac.ComputeHash(iv.Concat(ciphertext).ToArray());
                    if (!calc.SequenceEqual(tag))
                        throw new InvalidOperationException("Invalid encrypted tag");
                }

                using (var aes = Aes.Create())
                {
                    aes.Key = encKey;
                    aes.IV = iv;
                    aes.Mode = CipherMode.CBC;
                    aes.Padding = PaddingMode.PKCS7;

                    using (var decryptor = aes.CreateDecryptor())
                    using (var ms = new MemoryStream(ciphertext))
                    using (var cs = new CryptoStream(ms, decryptor, CryptoStreamMode.Read))
                    using (var outMs = new MemoryStream())
                    {
                        cs.CopyTo(outMs);
                        return outMs.ToArray();
                    }
                }
            }
        }

        // 读取词表
        private static Dictionary<int, string> LoadVocab(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                return new Dictionary<int, string>();

            var lines = File.ReadAllLines(path);
            var map = new Dictionary<int, string>();
            for (var i = 0; i < lines.Length; i++)
            {
                var s = lines[i].Trim();
                if (!string.IsNullOrEmpty(s))
                    map[i] = s;
            }
            return map;
        }

        // 执行一次完整 OCR 推理流程
        public OcrResult Run(string imagePath)
        {
            var img = ReadImageUnicode(imagePath);
            var h = img.Rows;
            var w = img.Cols;

            var detStart = DateTime.UtcNow;
            var detSize = _detInputSize ?? _options.DetImgSize;
            var detRawMax = _options.DetRawMaxDet > 0 ? _options.DetRawMaxDet : _options.DetMaxDet;

            var detOut = RunDetector(img, detSize, _options.DetConf, _options.DetIou, detRawMax);
            var detTime = (DateTime.UtcNow - detStart).TotalMilliseconds;

            var recStart = DateTime.UtcNow;
            var outBoxes = new List<OcrBox>();
            var roiPreviews = new List<Mat>();

            var minScore = Math.Max(_options.RecConf, _options.RecMinScore);
            var flipMinScore = Math.Max(_options.RecConf, _options.RecFlipMinScore);

            if (detOut.Polys.Count > 0)
            {
                var sortedPolys = SortPolys(detOut.Polys, _options.SortBy, _options.RecRowThresh);
                if (_options.DetDistinctMinCenterRatio > 0)
                {
                    sortedPolys = FilterDistinctPolysByDistance(sortedPolys, w, h, _options.DetDistinctMinCenterRatio, _options.DetMaxDet);
                }
                sortedPolys = SortPolys(sortedPolys, _options.SortBy, _options.RecRowThresh);

                var roiIndex = 0;
                foreach (var poly in sortedPolys)
                {
                    roiIndex++;
                    if (poly.Conf < _options.DetConf)
                        continue;

                    if (poly.Quad == null || poly.Quad.Length != 4)
                        continue;

                    var quad = OrderPoints(poly.Quad);
                    // 按 ROI 序号选择 pre-warp 扩边比例（例如 Roi3 给更大值，让 rec 看到完整大字字形）。
                    var roiPadRatio = ResolveRoiPadRatio(roiIndex);
                    var quadExpanded = (roiPadRatio > 0 || _options.RoiPadPx > 0)
                        ? ExpandQuad(quad, roiPadRatio, _options.RoiPadPx)
                        : quad;

                    var roi = WarpQuad(img, quadExpanded);
                    roiPreviews.Add(roi.Clone());

                    var recResult = RunRecognizerWithContextPad(roi, roiIndex, minScore, flipMinScore);
                    var rect = QuadToAxisAligned(quad);
                    // 修复：字符框反映射必须与 WarpQuad 使用同一 quadExpanded，
                    // 否则 ROI 实际尺寸与反向透视 src 尺寸不一致，字符回显框整体偏移/缩放，
                    // 在小 ROI（如 Roi3 仅 2 位数字）上偏差最明显，表现为"回显字符不正确"。
                    // 注：边界 clamp 已经在 RunRecognizer 内部 ROI-local 空间完成，
                    // 不再在图像空间用轴对齐 rect 去 clamp 旋转 quad（那样会夷平旋转角度）。
                    var mappedCharBoxes = MapCharBoxesToImage(recResult.CharBoxes, quadExpanded);
                    // Qt 同款：定位回显显示原始 det quad；识别与字符框映射仍使用 quadExpanded。
                    outBoxes.Add(new OcrBox(rect, quad, recResult.Text, recResult.Score, roiIndex, mappedCharBoxes));
                }

                if (_options.DetFallbackToBoxesWhenPolys && outBoxes.Count < _options.DetMaxDet && detOut.Boxes.Count > 0)
                {
                    var fillBoxes = SortBoxes(detOut.Boxes, _options.SortBy, _options.RecRowThresh);
                    foreach (var b in fillBoxes)
                    {
                        if (outBoxes.Count >= _options.DetMaxDet)
                            break;
                        if (b.Score < _options.DetConf)
                            continue;

                        var candidate = PadAxisRect(b.Rect, w, h, _options.RoiPadRatio, _options.RoiPadPx);
                        if (candidate.Width <= 1 || candidate.Height <= 1)
                            continue;

                        var minDist = Math.Max(4.0, Math.Min(w, h) * Math.Max(_options.DetDistinctMinCenterRatio, 0.0));
                        var isDistinct = outBoxes.All(o => IoU(candidate, o.Rect) < _options.DetDistinctIou && CenterDistance(candidate, o.Rect) >= minDist);
                        if (!isDistinct)
                            continue;

                        roiIndex++;
                        var roi = img.SubMat(candidate);
                        roiPreviews.Add(roi.Clone());
                        var recResult = RunRecognizerWithContextPad(roi, roiIndex, minScore, flipMinScore);
                        var mappedCharBoxes = MapCharBoxesToImage(recResult.CharBoxes, candidate);
                        outBoxes.Add(new OcrBox(candidate, null, recResult.Text, recResult.Score, roiIndex, mappedCharBoxes));
                    }
                }

                if (_options.DetFallbackToBoxesWhenPolys && outBoxes.Count < _options.DetMaxDet && detOut.Boxes.Count > 0)
                {
                    var remaining = SortBoxes(detOut.Boxes, _options.SortBy, _options.RecRowThresh)
                        .Where(b => b.Score >= _options.DetConf)
                        .ToList();
                    while (outBoxes.Count < _options.DetMaxDet && remaining.Count > 0)
                    {
                        DetBox best = null;
                        Rect bestRect = new Rect();
                        var bestDist = -1.0;
                        foreach (var b in remaining)
                        {
                            var candidate = PadAxisRect(b.Rect, w, h, _options.RoiPadRatio, _options.RoiPadPx);
                            if (candidate.Width <= 1 || candidate.Height <= 1)
                                continue;

                            var minDist = outBoxes.Count == 0
                                ? double.MaxValue
                                : outBoxes.Min(o => CenterDistance(candidate, o.Rect));
                            if (minDist > bestDist)
                            {
                                bestDist = minDist;
                                best = b;
                                bestRect = candidate;
                            }
                        }

                        if (best == null)
                            break;

                        remaining.Remove(best);
                        roiIndex++;
                        var roi = img.SubMat(bestRect);
                        roiPreviews.Add(roi.Clone());
                        var recResult = RunRecognizerWithContextPad(roi, roiIndex, minScore, flipMinScore);
                        var mappedCharBoxes = MapCharBoxesToImage(recResult.CharBoxes, bestRect);
                        outBoxes.Add(new OcrBox(bestRect, null, recResult.Text, recResult.Score, roiIndex, mappedCharBoxes));
                    }
                }
            }
            else
            {
                var sortedBoxes = SortBoxes(detOut.Boxes, _options.SortBy, _options.RecRowThresh);
                if (_options.DetDistinctMinCenterRatio > 0)
                {
                    sortedBoxes = FilterDistinctBoxesByDistance(sortedBoxes, w, h, _options.DetDistinctMinCenterRatio, _options.DetMaxDet);
                }
                if (sortedBoxes.Count == 0)
                {
                    return new OcrResult(outBoxes, detTime, 0, 0, 0.0, roiPreviews);
                }

                if (_options.DetMaxDet > 0 && sortedBoxes.Count > _options.DetMaxDet)
                {
                    sortedBoxes = sortedBoxes.Take(_options.DetMaxDet).ToList();
                }

                var roiIndex = 0;
                foreach (var box in sortedBoxes)
                {
                    roiIndex++;
                    if (box.Score < _options.DetConf)
                        continue;

                    var rect = PadAxisRect(box.Rect, w, h, _options.RoiPadRatio, _options.RoiPadPx);
                    if (rect.Width <= 1 || rect.Height <= 1)
                        continue;

                    var roi = img.SubMat(rect);
                    roiPreviews.Add(roi.Clone());

                    var recResult = RunRecognizerWithContextPad(roi, roiIndex, minScore, flipMinScore);
                    var mappedCharBoxes = MapCharBoxesToImage(recResult.CharBoxes, rect);
                    outBoxes.Add(new OcrBox(rect, null, recResult.Text, recResult.Score, roiIndex, mappedCharBoxes));
                }
            }

            var recTime = (DateTime.UtcNow - recStart).TotalMilliseconds;
            var total = detTime + recTime;
            var mean = outBoxes.Count > 0 ? outBoxes.Average(b => b.Score) : 0.0;

            return new OcrResult(outBoxes, detTime, recTime, total, mean, roiPreviews);
        }

        // 保存 ROI 预览图到项目根目录（默认保存到 "roi_previews" 文件夹）
        public void SaveRoiPreviews(List<Mat> roiPreviews, string outputDir = null)
        {
            if (roiPreviews == null || roiPreviews.Count == 0)
                return;

            var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
            var folder = string.IsNullOrWhiteSpace(outputDir)
                ? Path.Combine(projectRoot, "roi_previews")
                : Path.IsPathRooted(outputDir)
                    ? outputDir
                    : Path.Combine(projectRoot, outputDir);

            Directory.CreateDirectory(folder);

            for (var i = 0; i < roiPreviews.Count; i++)
            {
                var fileName = Path.Combine(folder, $"roi_{i + 1:D3}.png");
                Cv2.ImWrite(fileName, roiPreviews[i]);
            }
        }

        // 释放模型会话资源
        public void Dispose()
        {
            _detSession.Dispose();
            _recSession.Dispose();
        }

        // 识别单个 ROI
        private RecResult RunRecognizer(Mat roi, int roiIndex, double minScore, double flipMinScore)
        {
            var recImgSize = ChooseRecImgSize(roi, _options, _recInputSize);
            var recOut = RunRecognizerRaw(roi, recImgSize, _options.RecConf, _options.RecIou);

            var expectedLen = ExpectedLenForRoi(_options, roiIndex);
            var perTopN = expectedLen ?? _options.RecTopN;
            var recPostIou = _options.RecOutputFormat == OutputFormat.Auto ? _options.RecIou : 0.0;

            List<string> textParts;
            List<double> scores;
            List<CharBox> charBoxes;
            if (recOut.Polys.Count > 0)
            {
                var filteredPolys = FilterRecPolys(recOut.Polys, _options.SortBy, minScore, perTopN, _options.RecMinBox, _options.RecRowThresh, recPostIou);
                var recTexts = RecTextFromPolys(filteredPolys);
                textParts = recTexts.text;
                scores = recTexts.scores;
                charBoxes = recTexts.charBoxes;
            }
            else
            {
                var filtered = FilterRecBoxes(recOut.Boxes, _options.SortBy, minScore, perTopN, _options.RecMinBox, _options.RecRowThresh, recPostIou);
                var recTexts = RecTextFromBoxes(filtered);
                textParts = recTexts.text;
                scores = recTexts.scores;
                charBoxes = recTexts.charBoxes;
            }

            if (expectedLen.HasValue && textParts.Count > expectedLen.Value)
            {
                textParts = textParts.Take(expectedLen.Value).ToList();
                scores = scores.Take(expectedLen.Value).ToList();
            }

            if (_options.RecFlipEnable)
            {
                var roiFlip = roi.Flip(FlipMode.XY);
                var recImgSizeFlip = ChooseRecImgSize(roiFlip, _options, _recInputSize);
                var recOutFlip = RunRecognizerRaw(roiFlip, recImgSizeFlip, _options.RecConf, _options.RecIou);
                var filteredFlip = FilterRecBoxes(recOutFlip.Boxes, _options.SortBy, flipMinScore, perTopN, _options.RecMinBox, _options.RecRowThresh, recPostIou);
                var (flipText, flipScores, _) = RecTextFromBoxes(filteredFlip);

                if (expectedLen.HasValue && flipText.Count > expectedLen.Value)
                {
                    flipText = flipText.Take(expectedLen.Value).ToList();
                    flipScores = flipScores.Take(expectedLen.Value).ToList();
                }

                var baseMean = scores.Count > 0 ? scores.Average() : 0.0;
                var flipMean = flipScores.Count > 0 ? flipScores.Average() : 0.0;
                if (flipMean > baseMean && flipText.Count > 0)
                {
                    textParts = flipText;
                    scores = flipScores;
                }
            }

            var text = string.Concat(textParts);
            var score = scores.Count > 0 ? scores.Average() : 0.0;

            // 在 ROI 局部坐标系内对字符框做按 ROI 序号的扩边，
            // 用于补偿大/粗字符（典型：Roi3 的 2 位数字）上 YOLO 回归头偏紧的倾向。
            // 必须在 MapCharBoxesToImage 之前完成，因为这里的坐标系还是 ROI warp 后的方形空间。
            var charPadRatio = ResolveCharPadRatio(roiIndex);
            if (charPadRatio > 0 && charBoxes != null && charBoxes.Count > 0)
            {
                charBoxes = ExpandCharBoxesInRoi(charBoxes, charPadRatio, roi.Cols, roi.Rows);
                _options.Log?.Invoke($"roi#{roiIndex} char-box 扩边 {charPadRatio:F2}（rec 端补偿）");
            }

            // 不再依赖定位 ROI 扩边比例来控制字符框范围。
            // 识别上下文扩边由 RunRecognizerWithContextPad 单独处理，并在返回前反投回原 ROI 边界。

            // 保持输出文本与最终回显字符框一致，避免"只有1个框却有2个字符"。
            if (charBoxes != null)
            {
                if (charBoxes.Count == 0)
                {
                    text = string.Empty;
                    score = 0.0;
                }
                else
                {
                    text = string.Concat(charBoxes.Select(c => c.Text));
                    score = charBoxes.Average(c => c.Score);
                }
            }

            return new RecResult(text, score, charBoxes);
        }

        // 根据 ROI 序号获取期望长度（例如 8,14,2）
        private static int? ExpectedLenForRoi(OcrOptions options, int roiIndex)
        {
            if (options.RoiExpectedLengths == null || options.RoiExpectedLengths.Length == 0)
                return null;

            var idx = roiIndex - 1;
            if (idx < 0 || idx >= options.RoiExpectedLengths.Length)
                return null;

            var expected = options.RoiExpectedLengths[idx];
            return expected > 0 ? (int?)expected : null;
        }

        // 解析每个 ROI 的 pre-warp 扩边比例：优先 RoiPadRatios[index]，否则回退 RoiPadRatio
        private double ResolveRoiPadRatio(int roiIndex)
        {
            var arr = _options.RoiPadRatios;
            if (arr != null)
            {
                var idx = roiIndex - 1;
                if (idx >= 0 && idx < arr.Length && arr[idx] >= 0)
                    return arr[idx];
            }
            return _options.RoiPadRatio;
        }

        // 解析每个 ROI 的 char-box 扩边比例：优先 RecCharPadRatios[index]，否则回退 RecCharPadRatio
        private double ResolveCharPadRatio(int roiIndex)
        {
            var arr = _options.RecCharPadRatios;
            if (arr != null)
            {
                var idx = roiIndex - 1;
                if (idx >= 0 && idx < arr.Length && arr[idx] >= 0)
                    return arr[idx];
            }
            return _options.RecCharPadRatio;
        }

        // 解析每个 ROI 的识别上下文扩边比例：优先 RecRoiPadRatios[index]，否则回退 RecRoiPadRatio
        private double ResolveRecRoiPadRatio(int roiIndex)
        {
            var arr = _options.RecRoiPadRatios;
            if (arr != null)
            {
                var idx = roiIndex - 1;
                if (idx >= 0 && idx < arr.Length && arr[idx] >= 0)
                    return arr[idx];
            }
            return _options.RecRoiPadRatio;
        }

        // 不改变定位回显框的前提下，为识别单独增加 ROI 上下文（仿 Qt 的 ROI 扩边输入效果）。
        private RecResult RunRecognizerWithContextPad(Mat roi, int roiIndex, double minScore, double flipMinScore)
        {
            var recRoiPadRatio = ResolveRecRoiPadRatio(roiIndex);
            using (var recInput = BuildRecInputRoi(roi, recRoiPadRatio, _options.RecRoiPadPx, out var padX, out var padY))
            {
                var recResult = RunRecognizer(recInput, roiIndex, minScore, flipMinScore);
                var remapped = UnpadCharBoxesFromRoi(recResult.CharBoxes, padX, padY, roi.Cols, roi.Rows);
                if (remapped == null)
                    return recResult;

                var text = remapped.Count == 0 ? string.Empty : string.Concat(remapped.Select(c => c.Text));
                var score = remapped.Count == 0 ? 0.0 : remapped.Average(c => c.Score);
                return new RecResult(text, score, remapped);
            }
        }

        // 在 ROI 局部坐标系内扩展字符框（rect + quad），用于补偿大/粗字符上的过紧回归。
        // 仅修改尺寸，不改变中心；rect 边界裁剪到 [0, roiW]×[0, roiH]，quad 不裁剪以保持透视形状。
        private static List<CharBox> ExpandCharBoxesInRoi(List<CharBox> charBoxes, double ratio, int roiW, int roiH)
        {
            if (charBoxes == null || charBoxes.Count == 0 || ratio <= 0)
                return charBoxes;

            var maxX = Math.Max(roiW, 1);
            var maxY = Math.Max(roiH, 1);

            var result = new List<CharBox>(charBoxes.Count);
            foreach (var b in charBoxes)
            {
                var pad = (int)Math.Round(Math.Max(b.Rect.Width, b.Rect.Height) * ratio);
                if (pad < 1) pad = 1;
                var x1 = Math.Max(0, b.Rect.Left - pad);
                var y1 = Math.Max(0, b.Rect.Top - pad);
                var x2 = Math.Min(maxX, b.Rect.Right + pad);
                var y2 = Math.Min(maxY, b.Rect.Bottom + pad);
                var newRect = new Rect(x1, y1, Math.Max(1, x2 - x1), Math.Max(1, y2 - y1));

                Point2f[] newQuad = null;
                if (b.Quad != null && b.Quad.Length == 4)
                {
                    var cx = (b.Quad[0].X + b.Quad[1].X + b.Quad[2].X + b.Quad[3].X) / 4f;
                    var cy = (b.Quad[0].Y + b.Quad[1].Y + b.Quad[2].Y + b.Quad[3].Y) / 4f;
                    var s = (float)(1.0 + ratio);
                    // 关键：放大后必须把 quad 顶点 clamp 到 ROI 局部 [0..roiW]×[0..roiH]。
                    // 否则下游 MapCharBoxesToImage 用 GetPerspectiveTransform(src=ROI 局部矩形 → dst=det quadExpanded)
                    // 时，src 中超出 [0..w]×[0..h] 的点会被映射到 dst（det quad）外部，
                    // 表现为 Roi3 的字符框越界踩进相邻 Roi2 的区域。
                    newQuad = b.Quad.Select(p =>
                    {
                        var nx = (p.X - cx) * s + cx;
                        var ny = (p.Y - cy) * s + cy;
                        if (nx < 0) nx = 0;
                        if (ny < 0) ny = 0;
                        if (nx > maxX) nx = maxX;
                        if (ny > maxY) ny = maxY;
                        return new Point2f(nx, ny);
                    }).ToArray();
                }

                result.Add(new CharBox(newRect, newQuad, b.Text, b.Score));
            }
            return result;
        }

        // 为识别输入在 ROI 四周补边（不改变定位框）。
        private static Mat BuildRecInputRoi(Mat roi, double padRatio, int padPx, out int padX, out int padY)
        {
            var ratio = Math.Max(0.0, padRatio);
            var px = Math.Max(0, padPx);
            var pad = Math.Max((int)Math.Round(Math.Max(roi.Cols, roi.Rows) * ratio), px);
            if (pad <= 0)
            {
                padX = 0;
                padY = 0;
                return roi.Clone();
            }

            padX = pad;
            padY = pad;
            var outMat = new Mat();
            Cv2.CopyMakeBorder(roi, outMat, pad, pad, pad, pad, BorderTypes.Constant, new Scalar(114, 114, 114));
            return outMat;
        }

        // 将在补边 ROI 上得到的字符框反投回原 ROI 坐标，并裁剪到原 ROI 边界。
        private static List<CharBox> UnpadCharBoxesFromRoi(List<CharBox> charBoxes, int padX, int padY, int roiW, int roiH)
        {
            if (charBoxes == null)
                return null;
            if (charBoxes.Count == 0)
                return charBoxes;
            if (padX <= 0 && padY <= 0)
                return charBoxes;

            var maxX = Math.Max(roiW, 1);
            var maxY = Math.Max(roiH, 1);
            var remapped = new List<CharBox>(charBoxes.Count);
            foreach (var b in charBoxes)
            {
                var x1 = Math.Max(0, b.Rect.Left - padX);
                var y1 = Math.Max(0, b.Rect.Top - padY);
                var x2 = Math.Min(maxX, b.Rect.Right - padX);
                var y2 = Math.Min(maxY, b.Rect.Bottom - padY);
                if (x2 <= x1 || y2 <= y1)
                    continue;

                Point2f[] quad = null;
                if (b.Quad != null && b.Quad.Length == 4)
                {
                    quad = b.Quad.Select(p => new Point2f(
                        Math.Max(0, Math.Min(maxX, p.X - padX)),
                        Math.Max(0, Math.Min(maxY, p.Y - padY)))).ToArray();
                }

                remapped.Add(new CharBox(new Rect(x1, y1, x2 - x1, y2 - y1), quad, b.Text, b.Score));
            }
            return remapped;
        }

        private static bool ShouldApplyRecPostNms(OcrOptions options)
        {
            // ONNX 已导出 NMS 时，避免再次 NMS 造成短文本字符被抑制（如 Roi3）
            return options.RecOutputFormat == OutputFormat.Auto;
        }

        // ROI 局部坐标系内的轴对齐裁剪。
        // 用于把扩边后的字符框 clamp 回"原始 det quad 在 ROI-local 空间对应的中心矩形"，
        // 从而保证：
        //   1) 字符框不会越过原始 det quad（视觉上不出 Roi3 显示框）；
        //   2) ROI-local 空间内 char quad 是轴对齐的，逐点 clamp 不损失旋转，
        //      下游 MapCharBoxesToImage 做透视变换时，旋转角度由 det quad 自然带入。
        private static List<CharBox> ClampCharBoxesInRoi(List<CharBox> charBoxes, Rect clipRect)
        {
            if (charBoxes == null || charBoxes.Count == 0)
                return charBoxes;
            if (clipRect.Width <= 0 || clipRect.Height <= 0)
                return charBoxes;

            var minX = clipRect.Left;
            var minY = clipRect.Top;
            var maxX = clipRect.Right;
            var maxY = clipRect.Bottom;

            var result = new List<CharBox>(charBoxes.Count);
            foreach (var b in charBoxes)
            {
                var x1 = Math.Max(minX, b.Rect.Left);
                var y1 = Math.Max(minY, b.Rect.Top);
                var x2 = Math.Min(maxX, b.Rect.Right);
                var y2 = Math.Min(maxY, b.Rect.Bottom);
                if (x2 <= x1 || y2 <= y1)
                    continue; // 完全在 ROI 中心区外，丢弃
                var newRect = new Rect(x1, y1, x2 - x1, y2 - y1);

                Point2f[] newQuad = null;
                if (b.Quad != null && b.Quad.Length == 4)
                {
                    newQuad = b.Quad.Select(p => new Point2f(
                        Math.Max(minX, Math.Min(maxX, p.X)),
                        Math.Max(minY, Math.Min(maxY, p.Y)))).ToArray();
                }
                result.Add(new CharBox(newRect, newQuad, b.Text, b.Score));
            }
            return result;
        }

        // 已弃用：原"图像空间下用轴对齐 rect 裁剪 char quad" 的实现
        // 会把任何越界顶点 snap 到轴对齐边缘，从而把旋转 quad 夷平成轴对齐矩形，
        // 表现为 Roi3 字符框失去角度。新方案改在 ROI 局部空间用 ClampCharBoxesInRoi 完成裁剪，
        // 此处不再保留实现，避免误用。

        private static List<CharBox> MapCharBoxesToImage(List<CharBox> charBoxes, Rect roiRect)
        {
            if (charBoxes == null || charBoxes.Count == 0)
                return charBoxes;

            var mapped = new List<CharBox>(charBoxes.Count);
            foreach (var box in charBoxes)
            {
                Point2f[] quad = null;
                if (box.Quad != null && box.Quad.Length == 4)
                {
                    quad = new[]
                    {
                        new Point2f(box.Quad[0].X + roiRect.X, box.Quad[0].Y + roiRect.Y),
                        new Point2f(box.Quad[1].X + roiRect.X, box.Quad[1].Y + roiRect.Y),
                        new Point2f(box.Quad[2].X + roiRect.X, box.Quad[2].Y + roiRect.Y),
                        new Point2f(box.Quad[3].X + roiRect.X, box.Quad[3].Y + roiRect.Y)
                    };
                }

                var rect = quad != null ? QuadToAxisAligned(quad) : new Rect(
                    roiRect.X + box.Rect.X,
                    roiRect.Y + box.Rect.Y,
                    box.Rect.Width,
                    box.Rect.Height);
                mapped.Add(new CharBox(rect, quad, box.Text, box.Score));
            }
            return mapped;
        }

        private static List<CharBox> MapCharBoxesToImage(List<CharBox> charBoxes, Point2f[] quad)
        {
            if (charBoxes == null || charBoxes.Count == 0)
                return charBoxes;

            var rect = OrderPoints(quad);
            var w1 = Distance(rect[0], rect[1]);
            var w2 = Distance(rect[2], rect[3]);
            var h1 = Distance(rect[0], rect[3]);
            var h2 = Distance(rect[1], rect[2]);
            var width = Math.Max(1, (int)Math.Round(Math.Max(w1, w2)));
            var height = Math.Max(1, (int)Math.Round(Math.Max(h1, h2)));

            var src = new[]
            {
                new Point2f(0, 0),
                new Point2f(width - 1, 0),
                new Point2f(width - 1, height - 1),
                new Point2f(0, height - 1),
            };

            using (var m = Cv2.GetPerspectiveTransform(src, rect))
            {
                var mapped = new List<CharBox>(charBoxes.Count);
                foreach (var box in charBoxes)
                {
                    Point2f[] pts;
                    if (box.Quad != null && box.Quad.Length == 4)
                    {
                        pts = box.Quad;
                    }
                    else
                    {
                        pts = new[]
                        {
                            new Point2f(box.Rect.Left, box.Rect.Top),
                            new Point2f(box.Rect.Right, box.Rect.Top),
                            new Point2f(box.Rect.Right, box.Rect.Bottom),
                            new Point2f(box.Rect.Left, box.Rect.Bottom),
                        };
                    }

                    var dst = Cv2.PerspectiveTransform(pts, m);
                    var mappedRect = QuadToAxisAligned(dst);
                    mapped.Add(new CharBox(mappedRect, dst, box.Text, box.Score));
                }

                return mapped;
            }
        }

        // 透视矫正裁剪四点 ROI
        private static Mat WarpQuad(Mat img, Point2f[] quad)
        {
            var rect = OrderPoints(quad);
            var w1 = Distance(rect[0], rect[1]);
            var w2 = Distance(rect[2], rect[3]);
            var h1 = Distance(rect[0], rect[3]);
            var h2 = Distance(rect[1], rect[2]);
            var width = Math.Max(1, (int)Math.Round(Math.Max(w1, w2)));
            var height = Math.Max(1, (int)Math.Round(Math.Max(h1, h2)));

            var dst = new[]
            {
                new Point2f(0, 0),
                new Point2f(width - 1, 0),
                new Point2f(width - 1, height - 1),
                new Point2f(0, height - 1),
            };

            var m = Cv2.GetPerspectiveTransform(rect, dst);
            var outMat = new Mat();
            Cv2.WarpPerspective(img, outMat, m, new Size(width, height));
            return outMat;
        }

        // 扩展四点框用于 padding
        private static Point2f[] ExpandQuad(Point2f[] quad, double padRatio, int padPx)
        {
            var rect = OrderPoints(quad);
            var w1 = Distance(rect[0], rect[1]);
            var w2 = Distance(rect[2], rect[3]);
            var h1 = Distance(rect[0], rect[3]);
            var h2 = Distance(rect[1], rect[2]);
            var width = Math.Max(w1, w2);
            var height = Math.Max(h1, h2);
            var pad = Math.Max((int)Math.Round(Math.Max(width, height) * padRatio), padPx);
            var scale = 1.0 + pad / Math.Max(width, height);
            var center = new Point2f((rect[0].X + rect[2].X) / 2f, (rect[0].Y + rect[2].Y) / 2f);
            for (var i = 0; i < rect.Length; i++)
            {
                rect[i] = new Point2f(
                    (float)((rect[i].X - center.X) * scale + center.X),
                    (float)((rect[i].Y - center.Y) * scale + center.Y)
                );
            }
            return rect;
        }

        // 计算两点距离
        private static double Distance(Point2f a, Point2f b)
        {
            var dx = a.X - b.X;
            var dy = a.Y - b.Y;
            return Math.Sqrt(dx * dx + dy * dy);
        }

        // 将四点按左上/右上/右下/左下排序
        private static Point2f[] OrderPoints(Point2f[] pts)
        {
            var rect = new Point2f[4];
            var sum = pts.Select(p => p.X + p.Y).ToArray();
            rect[0] = pts[Array.IndexOf(sum, sum.Min())];
            rect[2] = pts[Array.IndexOf(sum, sum.Max())];
            var diff = pts.Select(p => p.Y - p.X).ToArray();
            rect[1] = pts[Array.IndexOf(diff, diff.Min())];
            rect[3] = pts[Array.IndexOf(diff, diff.Max())];
            return rect;
        }

        // 执行定位模型推理
        private DetOutput RunDetector(Mat img, int imgSize, double conf, double iou, int maxDet)
        {
            var input = PrepareInput(img, imgSize, out var scale, out var padX, out var padY);
            var tensor = ToTensor(input);

            var inputValue = NamedOnnxValue.CreateFromTensor(_detSession.InputMetadata.Keys.First(), tensor);
            using (var results = _detSession.Run(new[] { inputValue }))
            {
                var outputTensor = results.First().AsTensor<float>();
                LogDetRawOutput(outputTensor, 3, _options.Log);
                var decoded = DecodeYoloOutput(
                    outputTensor,
                    imgSize,
                    imgSize,
                    scale,
                    padX,
                    padY,
                    conf,
                    iou,
                    maxDet,
                    _options.DetClassCount,
                    _options.DetOutputFormat,
                    img.Cols,
                    img.Rows,
                    _options.Log,
                    false
                );

                return decoded;
            }
        }

        private static void LogDetRawOutput(Tensor<float> output, int rows, Action<string> log)
        {
            var shape = output.Dimensions.ToArray();
            if (shape.Length != 3)
                return;

            var channelsFirst = shape[1] < shape[2];
            var n = channelsFirst ? shape[2] : shape[1];
            var count = Math.Min(rows, n);

            for (var i = 0; i < count; i++)
            {
                var x = GetValue(output, channelsFirst, i, 0);
                var y = GetValue(output, channelsFirst, i, 1);
                var w = GetValue(output, channelsFirst, i, 2);
                var h = GetValue(output, channelsFirst, i, 3);
                var conf = GetValue(output, channelsFirst, i, 4);
                var cls = GetValue(output, channelsFirst, i, 5);
                var angle = GetValue(output, channelsFirst, i, 6);
                var msg = $"det raw[{i}]: x={x:F4}, y={y:F4}, w={w:F4}, h={h:F4}, conf={conf:F4}, cls={cls:F4}, angle={angle:F4}";
                Console.WriteLine(msg);
            }
        }

        // 执行识别模型推理（原始输出）
        private RecOutput RunRecognizerRaw(Mat img, int imgSize, double conf, double iou)
        {
            var input = PrepareInput(img, imgSize, out var scale, out var padX, out var padY);
            var tensor = ToTensor(input);

            var inputValue = NamedOnnxValue.CreateFromTensor(_recSession.InputMetadata.Keys.First(), tensor);
            using (var results = _recSession.Run(new[] { inputValue }))
            {
                var outputTensor = results.First().AsTensor<float>();
                if (_options.RecLogValueStats)
                {
                    LogValueStats("rec stats", outputTensor, _options.Log);
                }

            var recOffset = _options.RecClassStartOffset;
            var recHasObj = !_options.RecForceNoObj;
            var recHasAngle = _options.RecHasAngle;
            if (_options.RecAutoLayout)
            {
                var layout = DetectRecLayout(
                    outputTensor,
                    _options.RecClassCount,
                    _options.RecAutoChannelProbe,
                    _options.RecForceNoObj,
                    _options.Log
                );
                recOffset = layout.Offset;
                recHasObj = layout.HasObj;
            }
            else if (_options.RecAutoChannelOffset)
            {
                recOffset = DetectRecChannelOffset(outputTensor, _options.RecClassCount, _options.RecAutoChannelProbe, _options.Log);
            }

            var recUseSigmoid = ShouldApplySigmoidForRec(outputTensor, _options, recOffset, recHasObj, recHasAngle);

            LogRecRawOutput(outputTensor, 3, _options.Log, _options.RecClassCount, recUseSigmoid, recOffset, recHasObj, recHasAngle);

                if (_options.RecTestNoLetterbox)
                {
                    var noLbInput = PrepareInputNoLetterbox(img, imgSize);
                    var noLbTensor = ToTensor(noLbInput);
                    var noLbValue = NamedOnnxValue.CreateFromTensor(_recSession.InputMetadata.Keys.First(), noLbTensor);
                    using (var noLbResults = _recSession.Run(new[] { noLbValue }))
                    {
                        var noLbOutput = noLbResults.First().AsTensor<float>();
                        LogValueStats("rec stats (no-letterbox)", noLbOutput, _options.Log);
                    }
                }

            // Python 侧 rec 使用 NMS 默认 max_det（常见为 300），这里对齐
            var recMaxDet = _options.RecMaxDet;
            var decoded = DecodeYoloOutput(
                outputTensor,
                imgSize,
                imgSize,
                scale,
                padX,
                padY,
                conf,
                iou,
                recMaxDet,
                _options.RecClassCount,
                _options.RecOutputFormat,
                img.Cols,
                img.Rows,
                _options.Log,
                recUseSigmoid,
                recOffset,
                recHasObj,
                recHasAngle,
                true
            );
                var msg = $"rec decode: boxes={decoded.Boxes.Count}, polys={decoded.Polys.Count}";
                Console.WriteLine(msg);
                _options.Log?.Invoke(msg);

                return new RecOutput(decoded.Boxes, decoded.Polys);
            }
        }

        private static DetOutput DecodeYoloOutput(
            Tensor<float> output,
            int inputW,
            int inputH,
            float scale,
            int padX,
            int padY,
            double confThresh,
            double iouThresh,
            int maxDet,
            int classCount,
            OutputFormat format,
            int origW,
            int origH,
            Action<string> log,
            bool? applySigmoidRaw,
            int classStartOffset = 0,
            bool? recHasObj = null,
            bool? recHasAngle = null,
            bool nmsUseLetterbox = false)
        {
            var shape = output.Dimensions.ToArray();
            if (shape.Length != 3)
                throw new InvalidOperationException("Unsupported output shape");

            int c;
            int n;
            bool channelsFirst = shape[1] < shape[2];
            if (channelsFirst)
            {
                c = shape[1];
                n = shape[2];
            }
            else
            {
                n = shape[1];
                c = shape[2];
            }

            // Heuristic: if NMS export did not run, N is large (e.g., 8400). Treat as raw output.
            if (format == OutputFormat.NmsXyxy && n > 1000)
                format = OutputFormat.Auto;

            // Heuristic: OBB NMS output is often xywha with 7 channels (x,y,w,h,angle,conf,cls).
            if (format == OutputFormat.NmsXyxy && c == 7 && LooksLikeXywha(output, channelsFirst, n))
                format = OutputFormat.NmsXywha;

            var boxes = new List<DetBox>();
            var polys = new List<DetPoly>();

            var useSigmoidRaw = false;
            if (format == OutputFormat.Auto && c >= 6)
            {
                if (applySigmoidRaw.HasValue)
                {
                    useSigmoidRaw = applySigmoidRaw.Value;
                    var msg = $"raw sigmoid={(useSigmoidRaw ? "on" : "off")}";
                    Console.WriteLine(msg);
                    log?.Invoke(msg);
                }
                else
                {
                    useSigmoidRaw = ShouldApplySigmoid(output, channelsFirst, n, c, classCount);
                    var msg = $"raw sigmoid={(useSigmoidRaw ? "on" : "off")}";
                    Console.WriteLine(msg);
                    log?.Invoke(msg);
                }
            }

            for (var i = 0; i < n; i++)
            {
                float x;
                float y;
                float w;
                float h;
                float angle = 0f;
                float obj;
                int cls = 0;
                float score;

                if (format == OutputFormat.NmsXyxy)
                {
                    if (c < 5)
                        continue;
                    // NMS 输出常见格式：x1,y1,x2,y2,score,(cls)
                    var x1 = GetValue(output, channelsFirst, i, 0);
                    var y1 = GetValue(output, channelsFirst, i, 1);
                    var x2 = GetValue(output, channelsFirst, i, 2);
                    var y2 = GetValue(output, channelsFirst, i, 3);
                    obj = GetValue(output, channelsFirst, i, 4);
                    score = obj;
                    cls = c >= 6 ? (int)Math.Round(GetValue(output, channelsFirst, i, 5)) : 0;

                    if (score < confThresh)
                        continue;

                    var rectRaw = RectFromXyxy(x1, y1, x2, y2, origW, origH);
                    var rectUnpad = DecodeRectFromXyxy(x1, y1, x2, y2, scale, padX, padY, origW, origH);
                    var selected = IsMostlyInside(rectUnpad, origW, origH) ? rectUnpad : rectRaw;
                    boxes.Add(new DetBox(selected, score, cls));
                    continue;
                }
                else if (format == OutputFormat.NmsXywha)
                {
                    if (c < 6)
                        continue;
                    x = GetValue(output, channelsFirst, i, 0);
                    y = GetValue(output, channelsFirst, i, 1);
                    w = GetValue(output, channelsFirst, i, 2);
                    h = GetValue(output, channelsFirst, i, 3);
                    obj = GetValue(output, channelsFirst, i, 4);   // conf
                    score = obj;
                    cls = c >= 6 ? (int)Math.Round(GetValue(output, channelsFirst, i, 5)) : 0;
                    angle = c >= 7 ? GetValue(output, channelsFirst, i, 6) : 0f;

                    if (score < confThresh)
                        continue;

                    var rectCandidate = nmsUseLetterbox
                        ? ClampRect(DecodeRect(x, y, w, h, inputW, inputH, scale, padX, padY), origW, origH)
                        : ClampRect(DecodeRectRaw(x, y, w, h), origW, origH);
                    boxes.Add(new DetBox(rectCandidate, score, cls));
                    if (Math.Abs(angle) > 0.0001f)
                    {
                        var quad = nmsUseLetterbox
                            ? DecodeQuad(x, y, w, h, angle, inputW, inputH, scale, padX, padY)
                            : DecodeQuadRaw(x, y, w, h, angle);
                        polys.Add(new DetPoly(quad, score, cls));
                    }
                    continue;
                }
                else if (format == OutputFormat.Auto)
                {
                    if (c == 4 + classCount)
                    {
                        x = GetValue(output, channelsFirst, i, 0);
                        y = GetValue(output, channelsFirst, i, 1);
                        w = GetValue(output, channelsFirst, i, 2);
                        h = GetValue(output, channelsFirst, i, 3);
                        var clsStart = Math.Max(4, Math.Min(c - classCount, 4 + classStartOffset));
                        var (bestScore, bestIdx) = BestClassScore(output, channelsFirst, i, clsStart, classCount);
                        var clsProb = useSigmoidRaw ? Sigmoid(bestScore) : bestScore;
                        score = clsProb;
                        cls = bestIdx;
                    }
                    else if (c == 6)
                    {
                        x = GetValue(output, channelsFirst, i, 0);
                        y = GetValue(output, channelsFirst, i, 1);
                        w = GetValue(output, channelsFirst, i, 2);
                        h = GetValue(output, channelsFirst, i, 3);
                        if (classCount == 1)
                        {
                            // OBB 单类导出常见格式：x,y,w,h,angle,conf
                            angle = GetValue(output, channelsFirst, i, 4);
                            obj = GetValue(output, channelsFirst, i, 5);
                            score = useSigmoidRaw ? Sigmoid(obj) : obj;
                            cls = 0;
                        }
                        else
                        {
                            // 兼容旧格式：x,y,w,h,conf,cls
                            obj = GetValue(output, channelsFirst, i, 4);
                            score = useSigmoidRaw ? Sigmoid(obj) : obj;
                            cls = (int)Math.Round(GetValue(output, channelsFirst, i, 5));
                        }
                    }
                    else if (c == 7)
                    {
                        x = GetValue(output, channelsFirst, i, 0);
                        y = GetValue(output, channelsFirst, i, 1);
                        w = GetValue(output, channelsFirst, i, 2);
                        h = GetValue(output, channelsFirst, i, 3);
                        // NMS OBB 常见顺序：x,y,w,h,conf,cls,angle
                        obj = GetValue(output, channelsFirst, i, 4);
                        score = useSigmoidRaw ? Sigmoid(obj) : obj;
                        cls = (int)Math.Round(GetValue(output, channelsFirst, i, 5));
                        angle = GetValue(output, channelsFirst, i, 6);
                    }
                    else if (c == 5 + classCount)
                    {
                        x = GetValue(output, channelsFirst, i, 0);
                        y = GetValue(output, channelsFirst, i, 1);
                        w = GetValue(output, channelsFirst, i, 2);
                        h = GetValue(output, channelsFirst, i, 3);
                        var hasObj = recHasObj ?? true;
                        var hasAngle = recHasAngle ?? false;
                        if (hasObj)
                        {
                            var objIndex = Math.Max(4, Math.Min(c - 1, 4 + classStartOffset));
                            var clsStart = Math.Max(5, Math.Min(c - classCount, 5 + classStartOffset));
                            obj = GetValue(output, channelsFirst, i, objIndex);
                            var (bestScore, bestIdx) = BestClassScore(output, channelsFirst, i, clsStart, classCount);
                            var objProb = useSigmoidRaw ? Sigmoid(obj) : obj;
                            var clsProb = useSigmoidRaw ? Sigmoid(bestScore) : bestScore;
                            score = objProb * clsProb;
                            cls = bestIdx;
                        }
                        else
                        {
                            var clsStart = hasAngle
                                ? Math.Max(5, Math.Min(c - classCount, 5 + classStartOffset))
                                : Math.Max(4, Math.Min(c - classCount, 4 + classStartOffset));
                            var (bestScore, bestIdx) = BestClassScore(output, channelsFirst, i, clsStart, classCount);
                            var clsProb = useSigmoidRaw ? Sigmoid(bestScore) : bestScore;
                            score = clsProb;
                            cls = bestIdx;
                            if (hasAngle)
                            {
                                angle = GetValue(output, channelsFirst, i, 4);
                            }
                        }
                    }
                    else if (c == 6 + classCount)
                    {
                        x = GetValue(output, channelsFirst, i, 0);
                        y = GetValue(output, channelsFirst, i, 1);
                        w = GetValue(output, channelsFirst, i, 2);
                        h = GetValue(output, channelsFirst, i, 3);
                        angle = GetValue(output, channelsFirst, i, 4);
                        var hasObj = recHasObj ?? true;
                        if (hasObj)
                        {
                            var objIndex = Math.Max(5, Math.Min(c - 1, 5 + classStartOffset));
                            var clsStart = Math.Max(6, Math.Min(c - classCount, 6 + classStartOffset));
                            obj = GetValue(output, channelsFirst, i, objIndex);
                            var (bestScore, bestIdx) = BestClassScore(output, channelsFirst, i, clsStart, classCount);
                            var objProb = useSigmoidRaw ? Sigmoid(obj) : obj;
                            var clsProb = useSigmoidRaw ? Sigmoid(bestScore) : bestScore;
                            score = objProb * clsProb;
                            cls = bestIdx;
                        }
                        else
                        {
                            var clsStart = Math.Max(5, Math.Min(c - classCount, 5 + classStartOffset));
                            var (bestScore, bestIdx) = BestClassScore(output, channelsFirst, i, clsStart, classCount);
                            var clsProb = useSigmoidRaw ? Sigmoid(bestScore) : bestScore;
                            score = clsProb;
                            cls = bestIdx;
                        }
                    }
                    else
                    {
                        continue;
                    }
                }
                else
                {
                    // For explicit formats you can extend this branch.
                    continue;
                }

                if (score < confThresh)
                    continue;

                var rect = ClampRect(DecodeRect(x, y, w, h, inputW, inputH, scale, padX, padY), origW, origH);
                boxes.Add(new DetBox(rect, score, cls));

                if (Math.Abs(angle) > 0.0001f)
                {
                    var quad = DecodeQuad(x, y, w, h, angle, inputW, inputH, scale, padX, padY);
                    polys.Add(new DetPoly(quad, score, cls));
                }
            }

            List<DetBox> kept;
            List<DetPoly> keptPolys;
            var isAlreadyNmsOutput = format == OutputFormat.NmsXyxy || format == OutputFormat.NmsXywha;
            if (isAlreadyNmsOutput)
            {
                kept = boxes;
                keptPolys = polys;

                if (maxDet > 0)
                {
                    if (kept.Count > maxDet)
                        kept = kept.Take(maxDet).ToList();
                    if (keptPolys.Count > maxDet)
                        keptPolys = keptPolys.Take(maxDet).ToList();
                }
            }
            else
            {
                var nms = classCount > 1
                    ? NmsBoxesByClass(boxes, iouThresh, maxDet)
                    : NmsBoxes(boxes, iouThresh, maxDet);
                kept = nms.Select(idx => boxes[idx]).ToList();
                keptPolys = NmsPolys(polys, iouThresh, maxDet);
            }

             return new DetOutput(kept, keptPolys);
        }

        private static void LogRecRawOutput(Tensor<float> output, int rows, Action<string> log, int classCount, bool useSigmoidRaw, int classStartOffset, bool hasObj, bool hasAngle)
        {
            var shape = output.Dimensions.ToArray();
            if (shape.Length != 3)
                return;

            var channelsFirst = shape[1] < shape[2];
            var n = channelsFirst ? shape[2] : shape[1];
            var c = channelsFirst ? shape[1] : shape[2];
            var count = Math.Min(rows, n);
            var objIndex = Math.Max(4, Math.Min(c - 1, 4 + classStartOffset));
            var clsStart = hasObj
                ? Math.Max(5, Math.Min(c - classCount, 5 + classStartOffset))
                : hasAngle
                    ? Math.Max(5, Math.Min(c - classCount, 5 + classStartOffset))
                    : Math.Max(4, Math.Min(c - classCount, 4 + classStartOffset));

            for (var i = 0; i < count; i++)
            {
                var obj = hasObj && c > objIndex ? GetValue(output, channelsFirst, i, objIndex) : 0f;
                var angle = hasAngle && c > 4 ? GetValue(output, channelsFirst, i, 4) : 0f;
                var cls0 = c > clsStart ? GetValue(output, channelsFirst, i, clsStart) : 0f;
                var best = classCount > 0 && c >= clsStart + classCount
                    ? BestClassScore(output, channelsFirst, i, clsStart, classCount)
                    : (cls0, 0);
                var objProb = hasObj ? (useSigmoidRaw ? Sigmoid(obj) : obj) : 1.0f;
                var clsProb = useSigmoidRaw ? Sigmoid(best.Item1) : best.Item1;
                var score = objProb * clsProb;
                var msg = $"rec raw[{i}]: obj={obj:F4}, angle={angle:F4}, cls0={cls0:F4}, best={best.Item1:F4}, cls={best.Item2}, objSig={objProb:F4}, clsSig={clsProb:F4}, score={score:F4}, offset={classStartOffset}, hasObj={hasObj}, hasAngle={hasAngle}";
                Console.WriteLine(msg);
                log?.Invoke(msg);
            }
        }

        private sealed class RecLayout
        {
            public RecLayout(int offset, bool hasObj)
            {
                Offset = offset;
                HasObj = hasObj;
            }

            public int Offset { get; }
            public bool HasObj { get; }
        }

        // 自动推断 rec 的 obj/cls 布局
        private static RecLayout DetectRecLayout(Tensor<float> output, int classCount, int probeCount, bool preferNoObj, Action<string> log)
        {
            var shape = output.Dimensions.ToArray();
            if (shape.Length != 3)
                return new RecLayout(0, !preferNoObj);

            var channelsFirst = shape[1] < shape[2];
            var n = channelsFirst ? shape[2] : shape[1];
            var c = channelsFirst ? shape[1] : shape[2];
            if (c < 4 + classCount)
                return new RecLayout(0, !preferNoObj);

            var probes = Math.Max(1, Math.Min(probeCount, n));
            var maxOffset = Math.Max(0, Math.Min(3, c - (5 + classCount)));
            var offsets = Enumerable.Range(0, maxOffset + 1).ToArray();

            LogTopKChannels(output, channelsFirst, probes, Math.Min(8, c), log);

            var bestScore = double.MinValue;
            var bestOffset = 0;
            var bestHasObj = true;

            foreach (var offset in offsets)
            {
                foreach (var hasObj in new[] { true, false })
                {
                    if (preferNoObj && hasObj)
                        continue;
                    var score = EvaluateRecLayoutScore(output, channelsFirst, probes, c, classCount, offset, hasObj);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestOffset = offset;
                        bestHasObj = hasObj;
                    }
                }
            }

            var msg = $"rec layout: offset={bestOffset}, hasObj={bestHasObj}, score={bestScore:F4}";
            Console.WriteLine(msg);
            log?.Invoke(msg);
            return new RecLayout(bestOffset, bestHasObj);
        }

        private static double EvaluateRecLayoutScore(Tensor<float> output, bool channelsFirst, int probes, int c, int classCount, int offset, bool hasObj)
        {
            var clsStart = hasObj
                ? Math.Max(5, Math.Min(c - classCount, 5 + offset))
                : Math.Max(4, Math.Min(c - classCount, 4 + offset));
            var objIndex = Math.Max(4, Math.Min(c - 1, 4 + offset));

            if (clsStart < 0 || clsStart + classCount > c)
                return double.MinValue;
            if (hasObj && (objIndex < 0 || objIndex >= c))
                return double.MinValue;

            double sumMargin = 0.0;
            var count = 0;
            for (var i = 0; i < probes; i++)
            {
                var top1 = double.MinValue;
                var top2 = double.MinValue;
                for (var k = 0; k < classCount; k++)
                {
                    var v = GetValue(output, channelsFirst, i, clsStart + k);
                    if (v > top1)
                    {
                        top2 = top1;
                        top1 = v;
                    }
                    else if (v > top2)
                    {
                        top2 = v;
                    }
                }

                var margin = top1 - top2;
                if (hasObj)
                {
                    var obj = GetValue(output, channelsFirst, i, objIndex);
                    margin += Math.Abs(obj) * 0.1;
                }
                sumMargin += margin;
                count++;
            }

            return count > 0 ? sumMargin / count : double.MinValue;
        }

        // 依据输出峰值推断 rec 通道偏移（方案B）
        private static int DetectRecChannelOffset(Tensor<float> output, int classCount, int probeCount, Action<string> log)
        {
            var shape = output.Dimensions.ToArray();
            if (shape.Length != 3)
                return 0;

            var channelsFirst = shape[1] < shape[2];
            var n = channelsFirst ? shape[2] : shape[1];
            var c = channelsFirst ? shape[1] : shape[2];
            if (c < 5 + classCount)
                return 0;

            var probes = Math.Max(1, Math.Min(probeCount, n));

            // 输出 Top-K 通道值，便于定位 obj/cls 的真实通道
            LogTopKChannels(output, channelsFirst, probes, Math.Min(8, c), log);

            // 自动尝试 0/1/2 offset，选择“区分度最高”的偏移
            var candidates = new[] { 0, 1, 2 };
            var bestOffset = 0;
            var bestScore = double.MinValue;
            foreach (var offset in candidates)
            {
                var score = EvaluateOffsetScore(output, channelsFirst, probes, c, classCount, offset);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestOffset = offset;
                }
            }

            var msg = $"rec offset score: best_offset={bestOffset}, score={bestScore:F4}";
            Console.WriteLine(msg);
            log?.Invoke(msg);

            return Math.Max(0, Math.Min(bestOffset, c - (5 + classCount)));
        }

        private static double EvaluateOffsetScore(Tensor<float> output, bool channelsFirst, int probes, int c, int classCount, int offset)
        {
            var objIndex = Math.Max(4, Math.Min(c - 1, 4 + offset));
            var clsStart = Math.Max(5, Math.Min(c - classCount, 5 + offset));
            double sumMargin = 0.0;
            var count = 0;

            for (var i = 0; i < probes; i++)
            {
                var top1 = double.MinValue;
                var top2 = double.MinValue;
                for (var k = 0; k < classCount; k++)
                {
                    var v = GetValue(output, channelsFirst, i, clsStart + k);
                    if (v > top1)
                    {
                        top2 = top1;
                        top1 = v;
                    }
                    else if (v > top2)
                    {
                        top2 = v;
                    }
                }

                var obj = GetValue(output, channelsFirst, i, objIndex);
                var margin = (top1 - top2) + Math.Abs(obj) * 0.1;
                sumMargin += margin;
                count++;
            }

            return count > 0 ? sumMargin / count : 0.0;
        }

        private static void LogTopKChannels(Tensor<float> output, bool channelsFirst, int probes, int topK, Action<string> log)
        {
            var k = Math.Max(1, topK);
            var c = channelsFirst ? output.Dimensions[1] : output.Dimensions[2];
            for (var i = 0; i < Math.Min(probes, 2); i++)
            {
                var values = new List<(int ch, float v)>(c);
                for (var ch = 0; ch < c; ch++)
                {
                    var v = GetValue(output, channelsFirst, i, ch);
                    values.Add((ch, v));
                }
                var top = values.OrderByDescending(v => Math.Abs(v.v)).Take(k).ToList();
                var msg = $"rec topK channels (i={i}): " + string.Join(", ", top.Select(t => $"ch{t.ch}={t.v:F3}"));
                Console.WriteLine(msg);
                log?.Invoke(msg);
            }
        }

        private static SessionOptions CreateSessionOptions(OcrOptions options)
        {
            var opts = new SessionOptions();
            if (options.NumThreads > 0)
            {
                opts.IntraOpNumThreads = options.NumThreads;
                opts.InterOpNumThreads = options.NumThreads;
            }

            if (options.UseGpu)
            {
                try
                {
                    opts.AppendExecutionProvider_CUDA();
                }
                catch
                {
                    options.Log?.Invoke("GPU provider not available, using CPU.");
                }
            }

            return opts;
        }

        // 读取包含中文路径的图片
        private static Mat ReadImageUnicode(string path
        )
        {
            var bytes = File.ReadAllBytes(path);
            return Cv2.ImDecode(bytes, ImreadModes.Color);
        }

        // 将输入缩放到方形并补边
        private static Mat PrepareInput(Mat src, int size, out float scale, out int padX, out int padY)
        {
            var w = src.Cols;
            var h = src.Rows;
            scale = Math.Min((float)size / w, (float)size / h);
            var newW = (int)Math.Round(w * scale);
            var newH = (int)Math.Round(h * scale);

            using (var resized = new Mat())
            {
                Cv2.Resize(src, resized, new Size(newW, newH));
                var outMat = new Mat(new Size(size, size), MatType.CV_8UC3, new Scalar(114, 114, 114));

                padX = (size - newW) / 2;
                padY = (size - newH) / 2;
                var roi = new Rect(padX, padY, newW, newH);
                resized.CopyTo(new Mat(outMat, roi));
                return outMat;
            }
        }

        // 不做 letterbox 的直输入（仅用于排查预处理问题）
        private static Mat PrepareInputNoLetterbox(Mat src, int size)
        {
            var outMat = new Mat();
            Cv2.Resize(src, outMat, new Size(size, size));
            return outMat;
        }

        // 计算 Tensor 的 min/max/mean
        private static void LogValueStats(string title, Tensor<float> output, Action<string> log)
        {
            var data = output.ToArray();
            if (data.Length == 0)
                return;

            var min = double.MaxValue;
            var max = double.MinValue;
            double sum = 0.0;
            foreach (var v in data)
            {
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }
            var mean = sum / data.Length;
            var msg = $"{title}: min={min:F6}, max={max:F6}, mean={mean:F6}";
            Console.WriteLine(msg);
            log?.Invoke(msg);
        }

        // 输出加密模型 SHA256，确认加载的是最新文件
        private static void LogModelSha256(string tag, string path, byte[] data, Action<string> log)
        {
            using (var sha = SHA256.Create())
            {
                var hash = sha.ComputeHash(data);
                var hex = BitConverter.ToString(hash).Replace("-", string.Empty).ToLowerInvariant();
                var msg = $"{tag} enc sha256: {hex} ({path})";
                Console.WriteLine(msg);
                log?.Invoke(msg);
            }
        }

        // 将 OpenCV Mat 转为 CHW Tensor
        private static DenseTensor<float> ToTensor(Mat input)
        {
            var chw = new float[1 * 3 * input.Rows * input.Cols];
            var idx = 0;
            for (var c = 0; c < 3; c++)
            {
                for (var y = 0; y < input.Rows; y++)
                {
                    for (var x = 0; x < input.Cols; x++)
                    {
                        var pixel = input.At<Vec3b>(y, x);
                        byte value;
                        switch (c)
                        {
                            case 0:
                                value = pixel.Item2;
                                break;
                            case 1:
                                value = pixel.Item1;
                                break;
                            default:
                                value = pixel.Item0;
                                break;
                        }
                        chw[idx++] = value / 255.0f;
                    }
                }
            }

            return new DenseTensor<float>(chw, new[] { 1, 3, input.Rows, input.Cols });
        }

        // 从 xyxy 格式转换为矩形
        private static Rect RectFromXyxy(float x1, float y1, float x2, float y2, int imgW, int imgH)
        {
            var left = (int)Math.Round(Math.Min(x1, x2));
            var top = (int)Math.Round(Math.Min(y1, y2));
            var right = (int)Math.Round(Math.Max(x1, x2));
            var bottom = (int)Math.Round(Math.Max(y1, y2));
            return ClampRect(new Rect(left, top, Math.Max(1, right - left), Math.Max(1, bottom - top)), imgW, imgH);
        }

        private static Rect ClampRect(Rect rect, int imgW, int imgH)
        {
            var x1 = Math.Max(0, rect.Left);
            var y1 = Math.Max(0, rect.Top);
            var x2 = Math.Min(imgW, rect.Right);
            var y2 = Math.Min(imgH, rect.Bottom);
            return new Rect(x1, y1, Math.Max(1, x2 - x1), Math.Max(1, y2 - y1));
        }

        private static bool IsMostlyInside(Rect rect, int imgW, int imgH)
        {
            if (rect.Width <= 0 || rect.Height <= 0)
                return false;
            var area = rect.Width * rect.Height;
            var clamped = ClampRect(rect, imgW, imgH);
            var insideArea = clamped.Width * clamped.Height;
            return area > 0 && insideArea / (double)area >= 0.7;
        }

        // 将四点框转换为轴对齐矩形
        private static Rect QuadToAxisAligned(Point2f[] quad)
        {
            var xs = quad.Select(p => p.X).ToArray();
            var ys = quad.Select(p => p.Y).ToArray();
            var x1 = xs.Min();
            var x2 = xs.Max();
            var y1 = ys.Min();
            var y2 = ys.Max();
            return new Rect((int)Math.Round(x1), (int)Math.Round(y1), (int)Math.Round(x2 - x1), (int)Math.Round(y2 - y1));
        }

        // 对矩形进行 padding（基于检测框）
        private static Rect PadAxisRect(DetBox box, int w, int h, double padRatio, int padPx)
        {
            return PadAxisRect(box.Rect, w, h, padRatio, padPx);
        }

        // 对矩形进行 padding（基于 rect）
        private static Rect PadAxisRect(Rect rect, int w, int h, double padRatio, int padPx)
        {
            var pad = Math.Max((int)Math.Round(Math.Max(rect.Width, rect.Height) * padRatio), padPx);
            var x1 = Math.Max(0, rect.Left - pad);
            var y1 = Math.Max(0, rect.Top - pad);
            var x2 = Math.Min(w, rect.Right + pad);
            var y2 = Math.Min(h, rect.Bottom + pad);
            return new Rect(x1, y1, Math.Max(1, x2 - x1), Math.Max(1, y2 - y1));
        }

        private static Rect DecodeRectFromXyxy(float x1, float y1, float x2, float y2, float scale, int padX, int padY, int imgW, int imgH)
        {
            var ux1 = (x1 - padX) / scale;
            var uy1 = (y1 - padY) / scale;
            var ux2 = (x2 - padX) / scale;
            var uy2 = (y2 - padY) / scale;
            var left = (int)Math.Round(Math.Min(ux1, ux2));
            var top = (int)Math.Round(Math.Min(uy1, uy2));
            var right = (int)Math.Round(Math.Max(ux1, ux2));
            var bottom = (int)Math.Round(Math.Max(uy1, uy2));
            return ClampRect(new Rect(left, top, Math.Max(1, right - left), Math.Max(1, bottom - top)), imgW, imgH);
        }

        // 计算中心点距离
        private static double CenterDistance(Rect a, Rect b)
        {
            var ax = a.X + a.Width / 2.0;
            var ay = a.Y + a.Height / 2.0;
            var bx = b.X + b.Width / 2.0;
            var by = b.Y + b.Height / 2.0;
            var dx = ax - bx;
            var dy = ay - by;
            return Math.Sqrt(dx * dx + dy * dy);
        }

        // 按中心距离过滤多边形（用于去重）
        private static List<DetPoly> FilterDistinctPolysByDistance(List<DetPoly> polys, int imgW, int imgH, double minCenterRatio, int maxDet)
        {
            if (polys.Count == 0)
                return polys;

            var minDist = Math.Max(4.0, Math.Min(imgW, imgH) * Math.Max(minCenterRatio, 0.0));
            var sorted = polys.OrderByDescending(p => p.Conf).ToList();
            var kept = new List<DetPoly>();
            foreach (var p in sorted)
            {
                var rect = QuadToAxisAligned(p.Quad);
                var isDistinct = kept.All(k => CenterDistance(rect, QuadToAxisAligned(k.Quad)) >= minDist);
                if (!isDistinct)
                    continue;
                kept.Add(p);
                if (maxDet > 0 && kept.Count >= maxDet)
                    break;
            }
            return kept;
        }

        // 按中心距离过滤矩形（用于去重）
        private static List<DetBox> FilterDistinctBoxesByDistance(List<DetBox> boxes, int imgW, int imgH, double minCenterRatio, int maxDet)
        {
            if (boxes.Count == 0)
                return boxes;

            var minDist = Math.Max(4.0, Math.Min(imgW, imgH) * Math.Max(minCenterRatio, 0.0));
            var sorted = boxes.OrderByDescending(b => b.Score).ToList();
            var kept = new List<DetBox>();
            foreach (var b in sorted)
            {
                var isDistinct = kept.All(k => CenterDistance(b.Rect, k.Rect) >= minDist);
                if (!isDistinct)
                    continue;
                kept.Add(b);
                if (maxDet > 0 && kept.Count >= maxDet)
                    break;
            }
            return kept;
        }

        // YOLO 输出值读取
        private static float GetValue(Tensor<float> output, bool channelsFirst, int index, int channel)
        {
            return channelsFirst ? output[0, channel, index] : output[0, index, channel];
        }

        // sigmoid
        private static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        // 计算最大类别分数与索引
        private static (float score, int index) BestClassScore(Tensor<float> output, bool channelsFirst, int index, int startChannel, int classCount)
        {
            var bestScore = float.MinValue;
            var bestIdx = 0;
            for (var c = 0; c < classCount; c++)
            {
                var val = GetValue(output, channelsFirst, index, startChannel + c);
                if (val > bestScore)
                {
                    bestScore = val;
                    bestIdx = c;
                }
            }
            return (bestScore, bestIdx);
        }


        // 估计是否需要 sigmoid（rec 专用入口）
        private static bool ShouldApplySigmoidForRec(Tensor<float> output, OcrOptions options, int classStartOffset, bool hasObj, bool hasAngle)
        {
            if (options.RecForceSigmoid)
                return true;

            var shape = output.Dimensions.ToArray();
            if (shape.Length != 3)
                return true;

            var channelsFirst = shape[1] < shape[2];
            var n = channelsFirst ? shape[2] : shape[1];
            var c = channelsFirst ? shape[1] : shape[2];

            var clsStart = hasObj
                ? 5 + classStartOffset
                : hasAngle
                    ? 5 + classStartOffset
                    : 4 + classStartOffset;

            if (clsStart < 0 || clsStart + options.RecClassCount > c)
                return true;

            return ShouldApplySigmoidByCls(output, channelsFirst, n, clsStart, options.RecClassCount);
        }

        // 按类别通道判断是否需要 sigmoid（避免 bbox 通道干扰）
        private static bool ShouldApplySigmoidByCls(Tensor<float> output, bool channelsFirst, int n, int clsStart, int classCount)
        {
            var sample = Math.Min(n, 50);
            var min = float.MaxValue;
            var max = float.MinValue;
            for (var i = 0; i < sample; i++)
            {
                for (var k = 0; k < classCount; k++)
                {
                    var v = GetValue(output, channelsFirst, i, clsStart + k);
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }
            // 若超出 [0,1]，说明需要 sigmoid
            return min < -0.001f || max > 1.001f;
        }

        // 估计是否需要 sigmoid（输出已在 0~1 范围则不再 sigmoid）
        private static bool ShouldApplySigmoid(Tensor<float> output, bool channelsFirst, int n, int c, int classCount)
        {
            var sample = Math.Min(n, 50);
            var min = float.MaxValue;
            var max = float.MinValue;
            for (var i = 0; i < sample; i++)
            {
                for (var k = 0; k < Math.Min(c, 8); k++)
                {
                    var v = GetValue(output, channelsFirst, i, k);
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }
            // 若超出 [0,1]，说明需要 sigmoid
            return min < -0.001f || max > 1.001f;
        }

        // 判断 NMS 输出是否为 xywha
        private static bool LooksLikeXywha(Tensor<float> output, bool channelsFirst, int n)
        {
            var sample = Math.Min(n, 3);
            for (var i = 0; i < sample; i++)
            {
                var angle = GetValue(output, channelsFirst, i, 6);
                if (Math.Abs(angle) > 6.5f)
                    return false;
            }
            return true;
        }

        // XYWH 解码到原图坐标（考虑 letterbox）
        private static Rect DecodeRect(float x, float y, float w, float h, int inputW, int inputH, float scale, int padX, int padY)
        {
            var cx = (x - padX) / scale;
            var cy = (y - padY) / scale;
            var ww = w / scale;
            var hh = h / scale;
            var left = (int)Math.Round(cx - ww / 2.0f);
            var top = (int)Math.Round(cy - hh / 2.0f);
            return new Rect(left, top, Math.Max(1, (int)Math.Round(ww)), Math.Max(1, (int)Math.Round(hh)));
        }

        // XYWH 解码（不做 letterbox 反变换）
        private static Rect DecodeRectRaw(float x, float y, float w, float h)
        {
            var left = (int)Math.Round(x - w / 2.0f);
            var top = (int)Math.Round(y - h / 2.0f);
            return new Rect(left, top, Math.Max(1, (int)Math.Round(w)), Math.Max(1, (int)Math.Round(h)));
        }

        // OBB 解码到原图坐标（考虑 letterbox）
        private static Point2f[] DecodeQuad(float x, float y, float w, float h, float angle, int inputW, int inputH, float scale, int padX, int padY)
        {
            var cx = (x - padX) / scale;
            var cy = (y - padY) / scale;
            var ww = w / scale;
            var hh = h / scale;
            return BuildQuad(cx, cy, ww, hh, angle);
        }

        // OBB 解码（不做 letterbox 反变换）
        private static Point2f[] DecodeQuadRaw(float x, float y, float w, float h, float angle)
        {
            return BuildQuad(x, y, w, h, angle);
        }

        private static Point2f[] BuildQuad(float cx, float cy, float w, float h, float angle)
        {
            var cos = (float)Math.Cos(angle);
            var sin = (float)Math.Sin(angle);
            var hw = w / 2.0f;
            var hh = h / 2.0f;

            var dx1 = -hw;
            var dy1 = -hh;
            var dx2 = hw;
            var dy2 = -hh;
            var dx3 = hw;
            var dy3 = hh;
            var dx4 = -hw;
            var dy4 = hh;

            return new[]
            {
                new Point2f(cx + dx1 * cos - dy1 * sin, cy + dx1 * sin + dy1 * cos),
                new Point2f(cx + dx2 * cos - dy2 * sin, cy + dx2 * sin + dy2 * cos),
                new Point2f(cx + dx3 * cos - dy3 * sin, cy + dx3 * sin + dy3 * cos),
                new Point2f(cx + dx4 * cos - dy4 * sin, cy + dx4 * sin + dy4 * cos),
            };
        }

        // 多边形 NMS（按外接矩形 IoU）
        private static List<DetPoly> NmsPolys(List<DetPoly> polys, double iouThresh, int maxDet)
        {
            if (polys.Count == 0)
                return new List<DetPoly>();

            var order = polys.Select((p, i) => new { Score = p.Conf, Index = i })
                .OrderByDescending(x => x.Score)
                .Select(x => x.Index)
                .ToList();

            var keep = new List<int>();
            while (order.Count > 0)
            {
                var idx = order[0];
                order.RemoveAt(0);
                keep.Add(idx);

                if (maxDet > 0 && keep.Count >= maxDet)
                    break;

                var rectA = QuadToAxisAligned(polys[idx].Quad);
                order = order.Where(i => IoU(rectA, QuadToAxisAligned(polys[i].Quad)) < iouThresh).ToList();
            }

            return keep.Select(i => polys[i]).ToList();
        }
    }

    public sealed class OcrOptions
    {
        public int DetImgSize { get; set; } = 640;
        public double DetConf { get; set; } = 0.25;
        public double DetIou { get; set; } = 0.7;
        public int DetMaxDet { get; set; } = 3;
        public int DetRawMaxDet { get; set; } = 30;
        public int DetClassCount { get; set; } = 1;
        public double DetDistinctIou { get; set; } = 0.7;
        public double DetDistinctMinCenterRatio { get; set; } = 0.0;
        public bool DetFallbackToBoxesWhenPolys { get; set; } = false;
        public OutputFormat DetOutputFormat { get; set; } = OutputFormat.NmsXyxy;

        public int RecImgSize { get; set; } = 320;
        public bool RecAutoImgSize { get; set; } = true;
        public int RecImgSizeMin { get; set; } = 60;
        public int RecImgSizeMax { get; set; } = 640;
        public double RecConf { get; set; } = 0.25;
        public double RecIou { get; set; } = 0.7;
        public int RecTopN { get; set; } = 14;
        public int RecMinBox { get; set; } = 4;
        public double RecMinScore { get; set; } = 0.5;
        public bool RecFlipEnable { get; set; } = false;
        public double RecFlipMinScore { get; set; } = 0.5;
        public int RecMaxDet { get; set; } = 300;
        public int RecClassCount { get; set; } = 13;
        public OutputFormat RecOutputFormat { get; set; } = OutputFormat.NmsXyxy;
        public RecOutputMode RecMode { get; set; } = RecOutputMode.YoloLike;
        public int RecCtcBlankIndex { get; set; } = -1;
        public int RecCtcClassCount { get; set; } = 0;
        public bool RecUseObjectness { get; set; } = false;
        public bool RecUseSigmoid { get; set; } = false;
        public ImageChannelOrder RecChannelOrder { get; set; } = ImageChannelOrder.Rgb;
        public bool RecForceSigmoid { get; set; } = false;
        public bool RecAutoChannelOffset { get; set; } = true;
        public bool RecAutoLayout { get; set; } = true;
        public bool RecForceNoObj { get; set; } = false;
        public bool RecHasAngle { get; set; } = true;
        public int RecClassStartOffset { get; set; } = 0;
        public int RecAutoChannelProbe { get; set; } = 8;

        public bool RecLogValueStats { get; set; } = true;
        public bool RecTestNoLetterbox { get; set; } = false;

        public double RoiPadRatio { get; set; } = 0.06;// 与 Qt 解码测试工具保持一致
        public int RoiPadPx { get; set; } = 4;// 与 Qt 解码测试工具保持一致

        /// <summary>
        /// 识别阶段的 ROI 额外上下文扩边（不影响定位回显）。
        /// 例如 0.20 表示在 ROI 四周额外补 20% 的边，再送入 rec；
        /// 最终字符框会反投回原 ROI 范围，不会越过定位回显框。
        /// </summary>
        public double RecRoiPadRatio { get; set; } = 0.0;
        public int RecRoiPadPx { get; set; } = 0;

        /// <summary>
        /// 每个 ROI 单独的识别上下文扩边比例（优先级高于 RecRoiPadRatio）。
        /// 推荐：Roi1/2 保持 0，Roi3 设 0.20~0.35。
        /// 例："0,0,0.25"。 
        /// </summary>
        public double[] RecRoiPadRatios { get; set; } = { 0.0, 0.0, 0.45 };
        public string RecRoiPadRatiosText
        {
            get => RecRoiPadRatios == null ? string.Empty : string.Join(",", RecRoiPadRatios.Select(v => v.ToString(System.Globalization.CultureInfo.InvariantCulture)));
            set => RecRoiPadRatios = ParseDoubleList(value);
        }

        /// <summary>
        /// 每个 ROI 单独的 pre-warp 扩边比例（相对长边）。例如 "0.06,0.06,0.06"
        /// 表示 Roi1=6%, Roi2=6%, Roi3=6%。负值或越界则回退到全局 <see cref="RoiPadRatio"/>。
        /// </summary>
        public double[] RoiPadRatios { get; set; } = { 0.06, 0.06, 0.06 };
        public string RoiPadRatiosText
        {
            get => RoiPadRatios == null ? string.Empty : string.Join(",", RoiPadRatios.Select(v => v.ToString(System.Globalization.CultureInfo.InvariantCulture)));
            set => RoiPadRatios = ParseDoubleList(value);
        }

        /// <summary>
        /// rec 解码出的字符框在 ROI 局部坐标系下额外向外扩展的比例。0 表示不调整。
        /// 用于补偿大/粗字符上 YOLO 回归头的"过紧"倾向（典型：Roi3 的 2 位数字）。
        /// </summary>
        public double RecCharPadRatio { get; set; } = 0.0;

        /// <summary>
        /// 每个 ROI 单独的字符框扩边比例。负值或越界回退到 <see cref="RecCharPadRatio"/>。
        /// 例如 "0,0,0.20" → 仅对 Roi3 的 char box 外扩 20%。
        /// 经验值：Roi3 的 "1" 用较小比例就够，但 "0"/"8" 等宽字形需要 ~0.20 才能完整包住；
        /// 超出部分由 ExpandCharBoxesInRoi 内的 quad clamp 截断在 ROI 局部边界，
        /// 不会越界踩进相邻 Roi2 的区域。
        /// </summary>
        public double[] RecCharPadRatios { get; set; } = { 0.0, 0.0, 0.30 };
        public string RecCharPadRatiosText
        {
            get => RecCharPadRatios == null ? string.Empty : string.Join(",", RecCharPadRatios.Select(v => v.ToString(System.Globalization.CultureInfo.InvariantCulture)));
            set => RecCharPadRatios = ParseDoubleList(value);
        }

        public double RecRowThresh { get; set; } = 0.6;
        public int[] RoiExpectedLengths { get; set; } = { 8, 14, 2 };
        public string RoiExpectedLengthsText
        {
            get => RoiExpectedLengths == null ? string.Empty : string.Join(",", RoiExpectedLengths);
            set => RoiExpectedLengths = ParseRoiExpectedLengths(value);
        }

        public bool UseGpu { get; set; } = false;
        public int NumThreads { get; set; } = 0;
        public SortMode SortBy { get; set; } = SortMode.X;

        public string VocabPath { get; set; } = "vocab_rec.txt";
        public Action<string> Log { get; set; }

        // 解析 ROI 期望长度字符串
        public static int[] ParseRoiExpectedLengths(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return null;
            text = text.Replace("，", ",");
            var parts = text.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            var values = new List<int>();
            foreach (var p in parts)
            {
                if (int.TryParse(p.Trim(), out var v) && v > 0)
                    values.Add(v);
            }
            return values.Count > 0 ? values.ToArray() : null;
        }

        // 解析逗号分隔的浮点列表（用于 RoiPadRatios / RecCharPadRatios）
        public static double[] ParseDoubleList(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return null;
            text = text.Replace("，", ",");
            var parts = text.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            var values = new List<double>();
            foreach (var p in parts)
            {
                if (double.TryParse(p.Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v))
                    values.Add(v);
            }
            return values.Count > 0 ? values.ToArray() : null;
        }
    }
    public enum OutputFormat
    {
        Auto,
        NmsXyxy,
        NmsXywha,
    }

    public enum SortMode
    {
        X,
        Y,
        Line,
    }

    public enum RecOutputMode
    {
        Ctc,
        YoloLike,
    }

    public enum ImageChannelOrder
    {
        Rgb,
        Bgr,
    }

    public sealed class OcrBox
    {
        public OcrBox(Rect rect, Point2f[] quad, string text, double score, int roiIndex, List<CharBox> charBoxes = null)
        {
            Rect = rect;
            Quad = quad;
            Text = text;
            Score = score;
            RoiIndex = roiIndex;
            CharBoxes = charBoxes;
        }

        public Rect Rect { get; }
        public Point2f[] Quad { get; }
        public string Text { get; }
        public double Score { get; }
        public int RoiIndex { get; }
        public List<CharBox> CharBoxes { get; }
    }

    public sealed class OcrResult
    {
        public OcrResult(List<OcrBox> boxes, double detTimeMs, double recTimeMs, double totalTimeMs, double meanScore, List<Mat> roiPreviews)
        {
            Boxes = boxes;
            DetTimeMs = detTimeMs;
            RecTimeMs = recTimeMs;
            TotalTimeMs = totalTimeMs;
            MeanScore = meanScore;
            RoiPreviews = roiPreviews;
        }

        public List<OcrBox> Boxes { get; }
        public double DetTimeMs { get; }
        public double RecTimeMs { get; }
        public double TotalTimeMs { get; }
        public double MeanScore { get; }
        public List<Mat> RoiPreviews { get; }
    }

    internal sealed class DetBox
    {
        public DetBox(Rect rect, double score, int cls)
        {
            Rect = rect;
            Score = score;
            Cls = cls;
        }

        public Rect Rect { get; }
        public double Score { get; }
        public int Cls { get; }
        public Point2f Center => new Point2f(Rect.X + Rect.Width / 2f, Rect.Y + Rect.Height / 2f);
        public float Height => Rect.Height;
    }

    internal sealed class DetPoly
    {
        public DetPoly(Point2f[] quad, double conf, int cls)
        {
            Quad = quad;
            Conf = conf;
            Cls = cls;
        }

        public Point2f[] Quad { get; }
        public double Conf { get; }
        public int Cls { get; }
        public Point2f Center => new Point2f(Quad.Average(p => p.X), Quad.Average(p => p.Y));
        public float Height => (float)(Quad.Max(p => p.Y) - Quad.Min(p => p.Y));
    }

    internal sealed class DetOutput
    {
        public DetOutput(List<DetBox> boxes, List<DetPoly> polys)
        {
            Boxes = boxes;
            Polys = polys;
        }

        public List<DetBox> Boxes { get; }
        public List<DetPoly> Polys { get; }
    }

    internal sealed class RecOutput
    {
        public RecOutput(List<DetBox> boxes, List<DetPoly> polys)
        {
            Boxes = boxes ?? new List<DetBox>();
            Polys = polys ?? new List<DetPoly>();
        }

        public List<DetBox> Boxes { get; }
        public List<DetPoly> Polys { get; }
    }

    internal sealed class RecResult
    {
        public RecResult(string text, double score, List<CharBox> charBoxes)
        {
            Text = text;
            Score = score;
            CharBoxes = charBoxes;
        }

        public string Text { get; }
        public double Score { get; }
        public List<CharBox> CharBoxes { get; }
    }

    public sealed class CharBox
    {
        public CharBox(Rect rect, Point2f[] quad, string text, double score)
        {
            Rect = rect;
            Quad = quad;
            Text = text;
            Score = score;
        }

        public Rect Rect { get; }
        public Point2f[] Quad { get; }
        public string Text { get; }
        public double Score { get; }
    }
}

