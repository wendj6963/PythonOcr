using System;
using System.Windows;
using System.Windows.Input;
using Ssharp_Ocr_Wpf.ViewModels;

namespace Ssharp_Ocr_Wpf
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        private bool _isPanning;
        private Point _panStart;
        private double _scale = 1.0;

        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainViewModel();
        }

        private void ImageScrollViewer_OnPreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            const double zoomStep = 0.1;
            var delta = e.Delta > 0 ? zoomStep : -zoomStep;
            _scale = Math.Max(0.1, Math.Min(5.0, _scale + delta));
            ImageScale.ScaleX = _scale;
            ImageScale.ScaleY = _scale;
            e.Handled = true;
        }

        private void ImageScrollViewer_OnPreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            _isPanning = true;
            _panStart = e.GetPosition(ImageScrollViewer);
            ImageScrollViewer.CaptureMouse();
        }

        private void ImageScrollViewer_OnPreviewMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            _isPanning = false;
            ImageScrollViewer.ReleaseMouseCapture();
        }

        private void ImageScrollViewer_OnPreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (!_isPanning)
            {
                return;
            }

            var position = e.GetPosition(ImageScrollViewer);
            var dx = _panStart.X - position.X;
            var dy = _panStart.Y - position.Y;

            ImageScrollViewer.ScrollToHorizontalOffset(ImageScrollViewer.HorizontalOffset + dx);
            ImageScrollViewer.ScrollToVerticalOffset(ImageScrollViewer.VerticalOffset + dy);

            _panStart = position;
        }
    }
}
