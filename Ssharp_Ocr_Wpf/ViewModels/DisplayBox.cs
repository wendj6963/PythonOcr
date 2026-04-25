using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Media;

namespace Ssharp_Ocr_Wpf.ViewModels
{
    public sealed class DisplayBox
    {
        public DisplayBox(IEnumerable<Point> points, string text, double score)
        {
            var pointList = points?.ToList() ?? new List<Point>();
            Points = new PointCollection(pointList);
            Text = text;
            Score = score;

            if (pointList.Count == 0)
            {
                LabelX = 0;
                LabelY = 0;
                return;
            }

            LabelX = pointList.Min(p => p.X);
            LabelY = pointList.Min(p => p.Y);
        }

        public PointCollection Points { get; }
        public string Text { get; }
        public double Score { get; }
        public double LabelX { get; }
        public double LabelY { get; }
    }
}
