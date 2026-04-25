using System;
using Python.Runtime;

namespace ShuyunPgOcrDll
{
    /// <summary>
    /// OCR DLL 桥接类：通过 Python.NET 调用 shuyun_pg_ocr.py。
    /// </summary>
    public sealed class ShuyunPgOcrBridge
    {
        private static bool _initialized;

        /// <summary>
        /// 初始化 Python 运行时（建议在程序启动时调用一次）。
        /// </summary>
        /// <param name="pythonHome">Python 环境根目录（如 Conda 环境路径）。</param>
        /// <param name="pythonDll">Python DLL 完整路径（如 python310.dll）。</param>
        /// <param name="pythonPath">PythonPath（包含项目根目录）。</param>
        public static void Initialize(string? pythonHome, string? pythonDll, string? pythonPath)
        {
            if (_initialized)
            {
                return;
            }

            if (!string.IsNullOrWhiteSpace(pythonHome))
            {
                PythonEngine.PythonHome = pythonHome;
            }

            if (!string.IsNullOrWhiteSpace(pythonDll))
            {
                Runtime.PythonDLL = pythonDll;
            }

            if (!string.IsNullOrWhiteSpace(pythonPath))
            {
                Environment.SetEnvironmentVariable("PYTHONPATH", pythonPath);
            }

            PythonEngine.Initialize();
            _initialized = true;
        }

        /// <summary>
        /// 使用嵌入式 Python 目录初始化（免安装环境）。
        /// </summary>
        /// <param name="embeddedRoot">嵌入式 Python 根目录（包含 python.exe / python310.dll）。</param>
        /// <param name="appRoot">Python 代码根目录（包含 qt_app、src）。</param>
        /// <param name="pythonDllName">Python DLL 文件名（默认 python310.dll）。</param>
        public static void InitializeEmbedded(string embeddedRoot, string appRoot, string pythonDllName = "python310.dll")
        {
            if (_initialized)
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(embeddedRoot))
            {
                throw new ArgumentException("embeddedRoot 不能为空", nameof(embeddedRoot));
            }

            if (string.IsNullOrWhiteSpace(appRoot))
            {
                throw new ArgumentException("appRoot 不能为空", nameof(appRoot));
            }

            var pythonHome = embeddedRoot;
            var pythonDll = System.IO.Path.Combine(embeddedRoot, pythonDllName);
            var pythonPath = string.Join(
                System.IO.Path.PathSeparator,
                new[]
                {
                    embeddedRoot,
                    appRoot,
                }
            );

            Initialize(pythonHome, pythonDll, pythonPath);
        }

        /// <summary>
        /// 使用默认目录初始化嵌入式 Python。
        /// </summary>
        /// <param name="pythonDllName">Python DLL 文件名（默认 python310.dll）。</param>
        public static void InitializeEmbeddedDefault(string pythonDllName = "python310.dll")
        {
            var baseDir = AppContext.BaseDirectory.TrimEnd('\u005c', '/');
            var embeddedRoot = System.IO.Path.Combine(baseDir, "pyembed");
            var appRoot = System.IO.Path.Combine(embeddedRoot, "app");
            InitializeEmbedded(embeddedRoot, appRoot, pythonDllName);
        }

        /// <summary>
        /// 调用 Python OCR 并返回 JSON 字符串。
        /// </summary>
        public static string RunOcrJson(string imagePath, string detModel, string recModel, string? saveDir = null)
        {
            using (Py.GIL())
            {
                dynamic module = Py.Import("qt_app.shuyun_pg_ocr_dll.shuyun_pg_ocr");
                if (saveDir == null)
                {
                    return module.run_ocr_by_path_json(imagePath, detModel, recModel).ToString();
                }

                return module.run_ocr_by_path_json(imagePath, detModel, recModel, null, saveDir).ToString();
            }
        }
    }
}
