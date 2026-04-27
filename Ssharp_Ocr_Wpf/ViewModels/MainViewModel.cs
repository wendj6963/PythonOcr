using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using OpenCvSharp;

namespace Ssharp_Ocr_Wpf.ViewModels
{
    public sealed class MainViewModel : ViewModelBase
    {
        private readonly HashSet<string> _imageExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"
        };

        private List<string> _imageFiles = new List<string>();
        private int _currentIndex = -1;
        private BitmapImage _currentImage;
        private double _imagePixelWidth;
        private double _imagePixelHeight;
        private string _logText;
        private string _detModelPath = AppDomain.CurrentDomain.BaseDirectory + "onnx\\det.onnx.enc";
        private string _recModelPath = AppDomain.CurrentDomain.BaseDirectory + "onnx\\rec.onnx.enc";
        private string _ctcModelPath = AppDomain.CurrentDomain.BaseDirectory + "onnx\\rec_ctc.onnx.enc";
        private readonly List<BitmapImage> _roiPreviews = new List<BitmapImage>();
        private int _roiPreviewIndex = -1;
        private BitmapImage _currentRoiPreview;

        public MainViewModel()
        {
            Options = new OcrOptionsViewModel();
            Logs = new ObservableCollection<string>();
            Boxes = new ObservableCollection<DisplayBox>();
            RecBoxes = new ObservableCollection<DisplayBox>();

            BrowseFolderCommand = new RelayCommand(BrowseFolder);
            BrowseDetModelCommand = new RelayCommand(BrowseDetModelFile);
            BrowseRecModelCommand = new RelayCommand(BrowseRecModelFile);
            BrowseCtcModelCommand = new RelayCommand(BrowseCtcModelFile);
            PrevCommand = new RelayCommand(PrevImage, CanGoPrev);
            NextCommand = new RelayCommand(NextImage, CanGoNext);
            PrevRoiCommand = new RelayCommand(PrevRoi, CanGoPrevRoi);
            NextRoiCommand = new RelayCommand(NextRoi, CanGoNextRoi);
        }

        public OcrOptionsViewModel Options { get; }

        public ObservableCollection<string> Logs { get; }

        public ObservableCollection<DisplayBox> Boxes { get; }

        public ObservableCollection<DisplayBox> RecBoxes { get; }

        public string LogText
        {
            get => _logText;
            private set
            {
                _logText = value;
                OnPropertyChanged();
            }
        }

        public RelayCommand BrowseFolderCommand { get; }

        public RelayCommand BrowseDetModelCommand { get; }

        public RelayCommand BrowseRecModelCommand { get; }

        public RelayCommand BrowseCtcModelCommand { get; }

        public RelayCommand PrevCommand { get; }

        public RelayCommand NextCommand { get; }

        public RelayCommand PrevRoiCommand { get; }

        public RelayCommand NextRoiCommand { get; }

        public string DetModelPath
        {
            get => _detModelPath;
            set
            {
                _detModelPath = value;
                OnPropertyChanged();
            }
        }

        public string RecModelPath
        {
            get => _recModelPath;
            set
            {
                _recModelPath = value;
                OnPropertyChanged();
            }
        }

        public string CtcModelPath
        {
            get => _ctcModelPath;
            set
            {
                _ctcModelPath = value;
                OnPropertyChanged();
            }
        }

        public string Passphrase { get; set; } = "PG&shuyun@568.com";

        public BitmapImage CurrentImage
        {
            get => _currentImage;
            private set
            {
                _currentImage = value;
                OnPropertyChanged();
            }
        }

        public BitmapImage CurrentRoiPreview
        {
            get => _currentRoiPreview;
            private set
            {
                _currentRoiPreview = value;
                OnPropertyChanged();
            }
        }

        public double ImagePixelWidth
        {
            get => _imagePixelWidth;
            private set
            {
                _imagePixelWidth = value;
                OnPropertyChanged();
            }
        }

        public double ImagePixelHeight
        {
            get => _imagePixelHeight;
            private set
            {
                _imagePixelHeight = value;
                OnPropertyChanged();
            }
        }

        public string ProgressText
        {
            get
            {
                if (_imageFiles.Count == 0)
                {
                    return "0/0";
                }

                return string.Format("{0}/{1}", _currentIndex + 1, _imageFiles.Count);
            }
        }

        public string CurrentFileName
        {
            get
            {
                if (_currentIndex < 0 || _currentIndex >= _imageFiles.Count)
                {
                    return string.Empty;
                }

                return Path.GetFileName(_imageFiles[_currentIndex]);
            }
        }

        private void BrowseFolder()
        {
            var dialog = new OpenFileDialog
            {
                Title = "选择图片文件夹中的任意图片",
                Filter = "图片文件|*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tif;*.tiff",
                CheckFileExists = true,
                Multiselect = false
            };

            if (dialog.ShowDialog() != true)
            {
                return;
            }

            var folder = Path.GetDirectoryName(dialog.FileName);
            if (!string.IsNullOrWhiteSpace(folder))
            {
                LoadImages(folder);
            }
        }

        private void BrowseDetModelFile()
        {
            var selectedPath = BrowseModelFile("选择检测模型文件", DetModelPath);
            if (!string.IsNullOrWhiteSpace(selectedPath))
            {
                DetModelPath = selectedPath;
            }
        }

        private void BrowseRecModelFile()
        {
            var selectedPath = BrowseModelFile("选择识别模型文件", RecModelPath);
            if (!string.IsNullOrWhiteSpace(selectedPath))
            {
                RecModelPath = selectedPath;
            }
        }

        private void BrowseCtcModelFile()
        {
            var selectedPath = BrowseModelFile("选择CTC模型文件", CtcModelPath);
            if (!string.IsNullOrWhiteSpace(selectedPath))
            {
                CtcModelPath = selectedPath;
            }
        }

        private static string BrowseModelFile(string title, string currentPath)
        {
            var dialog = new OpenFileDialog
            {
                Title = title,
                CheckFileExists = true,
                Multiselect = false,
                Filter = "ONNX模型|*.onnx;*.enc|所有文件|*.*"
            };

            try
            {
                if (!string.IsNullOrWhiteSpace(currentPath))
                {
                    var initialDir = Path.GetDirectoryName(currentPath);
                    if (!string.IsNullOrWhiteSpace(initialDir) && Directory.Exists(initialDir))
                    {
                        dialog.InitialDirectory = initialDir;
                    }

                    dialog.FileName = Path.GetFileName(currentPath);
                }
            }
            catch
            {
                // Ignore invalid pre-filled path and keep default dialog location.
            }

            return dialog.ShowDialog() == true ? dialog.FileName : string.Empty;
        }

        private void LoadImages(string folderPath)
        {
            var files = Directory.EnumerateFiles(folderPath)
                .Where(file => _imageExtensions.Contains(Path.GetExtension(file)))
                .OrderBy(file => file)
                .ToList();

            _imageFiles = files;
            _currentIndex = _imageFiles.Count > 0 ? 0 : -1;

            Boxes.Clear();
            _roiPreviews.Clear();
            _roiPreviewIndex = -1;
            CurrentRoiPreview = null;
            AddLog(string.Format("已加载 {0} 张图片。", _imageFiles.Count));

            UpdateCurrentImage();
            UpdateProgress();
            UpdateCommands();
        }

        private void PrevImage()
        {
            if (_currentIndex <= 0)
            {
                return;
            }

            _currentIndex--;
            UpdateCurrentImage();
            UpdateProgress();
            UpdateCommands();
        }

        private void NextImage()
        {
            if (_currentIndex >= _imageFiles.Count - 1)
            {
                return;
            }

            _currentIndex++;
            UpdateCurrentImage();
            UpdateProgress();
            UpdateCommands();
        }

        private bool CanGoPrev()
        {
            return _currentIndex > 0;
        }

        private bool CanGoNext()
        {
            return _currentIndex >= 0 && _currentIndex < _imageFiles.Count - 1;
        }

        private void UpdateCurrentImage()
        {
            Boxes.Clear();
            RecBoxes.Clear();

            if (_currentIndex < 0 || _currentIndex >= _imageFiles.Count)
            {
                CurrentImage = null;
                ImagePixelWidth = 0;
                ImagePixelHeight = 0;
                return;
            }

            var filePath = _imageFiles[_currentIndex];
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.UriSource = new Uri(filePath);
            bitmap.EndInit();
            bitmap.Freeze();

            CurrentImage = bitmap;
            ImagePixelWidth = bitmap.PixelWidth;
            ImagePixelHeight = bitmap.PixelHeight;

            RecognizeCurrent(filePath);
        }

        private void RecognizeCurrent(string imagePath)
        {
            var recModelPath = ResolveRecModelPath();
            if (string.IsNullOrWhiteSpace(DetModelPath) || string.IsNullOrWhiteSpace(recModelPath))
            {
                AddLog("请先配置检测模型和识别模型路径。");
                return;
            }

            try
            {
                AddLog(string.Format("开始识别: {0}", Path.GetFileName(imagePath)));
                var options = Options.ToOptions(AddLog);
                AddLog(string.Format("识别后端: {0}", options.RecBackend));

                using (var runner = new OcrRunner(DetModelPath, recModelPath, Passphrase ?? string.Empty, options))
                {
                    var result = runner.Run(imagePath);
                    Boxes.Clear();
                    RecBoxes.Clear();
                    _roiPreviews.Clear();
                    foreach (var box in result.Boxes)
                    {
                        var displayText = string.Format("Roi{0}:{1}", box.RoiIndex, box.Text);
                        if (box.Quad != null && box.Quad.Length == 4)
                        {
                            var points = box.Quad.Select(p => new System.Windows.Point(p.X, p.Y));
                            Boxes.Add(new DisplayBox(points, displayText, box.Score));
                        }
                        else
                        {
                            var rect = box.Rect;
                            var points = new[]
                            {
                                new System.Windows.Point(rect.Left, rect.Top),
                                new System.Windows.Point(rect.Right, rect.Top),
                                new System.Windows.Point(rect.Right, rect.Bottom),
                                new System.Windows.Point(rect.Left, rect.Bottom)
                            };
                            Boxes.Add(new DisplayBox(points, displayText, box.Score));
                        }
                    }

                    foreach (var ocrBox in result.Boxes)
                    {
                        if (ocrBox.CharBoxes == null || ocrBox.CharBoxes.Count == 0)
                            continue;

                        foreach (var charBox in ocrBox.CharBoxes)
                        {
                            System.Windows.Point[] points;
                            if (charBox.Quad != null && charBox.Quad.Length == 4)
                            {
                                points = charBox.Quad.Select(p => new System.Windows.Point(p.X, p.Y)).ToArray();
                            }
                            else
                            {
                                var rect = charBox.Rect;
                                points = new[]
                                {
                                    new System.Windows.Point(rect.Left, rect.Top),
                                    new System.Windows.Point(rect.Right, rect.Top),
                                    new System.Windows.Point(rect.Right, rect.Bottom),
                                    new System.Windows.Point(rect.Left, rect.Bottom)
                                };
                            }
                            RecBoxes.Add(new DisplayBox(points, charBox.Text, charBox.Score));
                        }
                    }

                    foreach (var roi in result.RoiPreviews)
                    {
                        var preview = ConvertMatToBitmapImage(roi);
                        if (preview != null)
                        {
                            _roiPreviews.Add(preview);
                        }
                    }

                    if (_roiPreviews.Count > 0)
                    {
                        _roiPreviewIndex = 0;
                        CurrentRoiPreview = _roiPreviews[_roiPreviewIndex];
                    }
                    else
                    {
                        _roiPreviewIndex = -1;
                        CurrentRoiPreview = null;
                    }

                    OnPropertyChanged("RoiProgressText");
                    PrevRoiCommand.RaiseCanExecuteChanged();
                    NextRoiCommand.RaiseCanExecuteChanged();

                    AddLog(string.Format("识别完成，均值置信度: {0:F2}", result.MeanScore));
                }
            }
            catch (Exception ex)
            {
                AddLog(string.Format("识别失败: {0}", ex.Message));
            }
        }

        private string ResolveRecModelPath()
        {
            if (Options.RecBackend == RecBackend.Ctc)
            {
                if (!string.IsNullOrWhiteSpace(CtcModelPath))
                {
                    return CtcModelPath;
                }

                // 兼容旧配置：未单独配置 CTC 时回退到识别模型路径。
                return RecModelPath;
            }

            return RecModelPath;
        }

        private void UpdateProgress()
        {
            OnPropertyChanged("ProgressText");
            OnPropertyChanged("CurrentFileName");
        }

        private void UpdateCommands()
        {
            PrevCommand.RaiseCanExecuteChanged();
            NextCommand.RaiseCanExecuteChanged();
            PrevRoiCommand.RaiseCanExecuteChanged();
            NextRoiCommand.RaiseCanExecuteChanged();
        }

        private void PrevRoi()
        {
            if (_roiPreviewIndex <= 0)
            {
                return;
            }

            _roiPreviewIndex--;
            CurrentRoiPreview = _roiPreviews[_roiPreviewIndex];
            OnPropertyChanged("RoiProgressText");
            PrevRoiCommand.RaiseCanExecuteChanged();
            NextRoiCommand.RaiseCanExecuteChanged();
        }

        private void NextRoi()
        {
            if (_roiPreviewIndex >= _roiPreviews.Count - 1)
            {
                return;
            }

            _roiPreviewIndex++;
            CurrentRoiPreview = _roiPreviews[_roiPreviewIndex];
            OnPropertyChanged("RoiProgressText");
            PrevRoiCommand.RaiseCanExecuteChanged();
            NextRoiCommand.RaiseCanExecuteChanged();
        }

        private bool CanGoPrevRoi()
        {
            return _roiPreviewIndex > 0;
        }

        private bool CanGoNextRoi()
        {
            return _roiPreviewIndex >= 0 && _roiPreviewIndex < _roiPreviews.Count - 1;
        }

        private static BitmapImage ConvertMatToBitmapImage(Mat mat)
        {
            if (mat == null || mat.Empty())
            {
                return null;
            }

            var data = mat.ToBytes(".png");
            var bitmap = new BitmapImage();
            using (var stream = new MemoryStream(data))
            {
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.StreamSource = stream;
                bitmap.EndInit();
                bitmap.Freeze();
            }

            return bitmap;
        }

        private void AddLog(string message)
        {
            var entry = string.Format("[{0:HH:mm:ss}] {1}", DateTime.Now, message);
            Logs.Add(entry);
            LogText = string.Join(Environment.NewLine, Logs);
        }
    }
}
